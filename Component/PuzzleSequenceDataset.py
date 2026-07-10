import math

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

CLOCK_CAP_SECONDS = 600.0  # deve combaciare con GraphBuilder.CLOCK_CAP_SECONDS


def _clock_seconds_from_norm(clock_norm: float) -> float:
    denom = math.log1p(CLOCK_CAP_SECONDS)
    return math.expm1(clock_norm * denom)


def group_puzzle_sequences(puzzle_data_list):
    by_puzzle = {}
    for d in puzzle_data_list:
        pid = getattr(d, "puzzle_id", None)
        if pid is None:
            continue  # non e' una posizione-puzzle (es. proviene da games.pt)
        by_puzzle.setdefault(pid, []).append(d)

    sequences = []
    for plies in by_puzzle.values():
        plies_sorted = sorted(plies, key=lambda d: d.x[0, 3].item())
        sequences.append(plies_sorted)
    return sequences


class PuzzleSequenceDataset(Dataset):

    def __init__(self, puzzle_data_list):
        self.sequences = group_puzzle_sequences(puzzle_data_list)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def timed_collate_fn(batch_of_sequences):
    flat_positions = []
    chain_src, chain_dst, chain_dt = [], [], []

    running_idx = 0
    for sequence in batch_of_sequences:
        for i, d in enumerate(sequence):
            flat_positions.append(d)
            if i > 0:
                t_prev = _clock_seconds_from_norm(sequence[i - 1].x[0, 3].item())
                t_curr = _clock_seconds_from_norm(d.x[0, 3].item())
                chain_src.append(running_idx + i - 1)
                chain_dst.append(running_idx + i)
                chain_dt.append(max(t_curr - t_prev, 0.0))  # difesa: mai negativo
        running_idx += len(sequence)

    inner_batch = Batch.from_data_list(flat_positions)

    if chain_src:
        chain_edge_index = torch.tensor([chain_src, chain_dst], dtype=torch.long)
        chain_edge_attr = torch.tensor(chain_dt, dtype=torch.float)
    else:
        chain_edge_index = torch.zeros((2, 0), dtype=torch.long)
        chain_edge_attr = torch.zeros((0,), dtype=torch.float)

    return inner_batch, chain_edge_index, chain_edge_attr
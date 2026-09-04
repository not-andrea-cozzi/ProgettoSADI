import math
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

CLOCK_CAP_SECONDS = 600.0

# Chiavi non-tensor / metadati: NON devono finire nel Batch collate
EXCLUDE_KEYS = [
    "game_id", "puzzle_id", "problem_id",
    "best_move_uci", "best_move_idx",
    "fen", "clock_source", "clock_is_real",
    "ply", "move_idx", "source",
    "clock_seconds",  # letto per-oggetto nel collate, non serve nel batch
]


def _clock_seconds_from_norm(clock_norm: float) -> float:
    denom = math.log1p(CLOCK_CAP_SECONDS)
    if denom <= 0:
        return 0.0
    return max(0.0, math.expm1(clock_norm * denom))


def group_puzzle_sequences(puzzle_data_list):
    """Raggruppa per (game_id, problem_id) se disponibili, altrimenti puzzle_id."""
    by_puzzle = {}
    for d in puzzle_data_list:
        gid = getattr(d, "game_id", None)
        pid = getattr(d, "problem_id", None)
        if gid is not None and pid is not None:
            key = (gid, pid) if not isinstance(gid, list) else (tuple(gid), pid)
        else:
            key = getattr(d, "puzzle_id", None) or gid or pid
            if isinstance(key, list):
                key = tuple(key) if len(key) > 1 else key[0]
        if key is None:
            continue
        by_puzzle.setdefault(key, []).append(d)

    sequences = []
    for plies in by_puzzle.values():
        plies_sorted = sorted(
            plies,
            key=lambda d: getattr(d, "ply", getattr(d, "move_idx", 0))
        )
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
                if hasattr(d, "clock_seconds") and d.clock_seconds is not None:
                    dt = float(d.clock_seconds)
                else:
                    dt = _clock_seconds_from_norm(d.x[0, 3].item())
                chain_src.append(running_idx + i - 1)
                chain_dst.append(running_idx + i)
                chain_dt.append(max(0.0, dt))
        running_idx += len(sequence)

    # exclude_keys: evita KeyError su chiavi eterogenee/non-tensor
    inner_batch = Batch.from_data_list(flat_positions, exclude_keys=EXCLUDE_KEYS)

    if chain_src:
        chain_edge_index = torch.tensor([chain_src, chain_dst], dtype=torch.long)
        chain_edge_attr = torch.tensor(chain_dt, dtype=torch.float)
    else:
        chain_edge_index = torch.zeros((2, 0), dtype=torch.long)
        chain_edge_attr = torch.zeros((0,), dtype=torch.float)

    return inner_batch, chain_edge_index, chain_edge_attr

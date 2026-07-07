import random
import chess
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from Model.graph_builder import board_to_pyg_data


def merge_and_split(puzzle_pt_list, games_pt_path, out_dir, seed=42):
    """Unisce Data list da puzzle (gia' processati) + partite (.pt da ChessAnalysisPipeline)
    e risalva 3 file train/val/test.pt coerenti col resto della pipeline."""
    import os
    games_data = torch.load(games_pt_path) if games_pt_path else []
    all_data = list(puzzle_pt_list) + list(games_data)
    random.Random(seed).shuffle(all_data)

    n = len(all_data)
    i_train, i_val = int(n * 0.8), int(n * 0.9)
    splits = {
        "train": all_data[:i_train],
        "val": all_data[i_train:i_val],
        "test": all_data[i_val:],
    }
    os.makedirs(out_dir, exist_ok=True)
    for name, dlist in splits.items():
        torch.save(dlist, f"{out_dir}/merged_{name}.pt")
    return splits


class PuzzleGraphDataset(InMemoryDataset):
    """Costruisce dataset PyG da lichess_db_puzzle.csv filtrato per mateIn1-5."""

    def __init__(self, csv_path, root, split="train", mate_range=(1, 5),
                 max_puzzles=None, seed=42, avg_time_by_rating=None):
        self.csv_path = csv_path
        self.mate_range = mate_range
        self.max_puzzles = max_puzzles
        self.seed = seed
        self.split = split  # train | val | test
        self.avg_time_by_rating = avg_time_by_rating or {}
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"puzzle_{self.split}.pt"]

    def process(self):
        df = pd.read_csv(self.csv_path)
        lo, hi = self.mate_range
        mask = df["Themes"].str.contains(
            "|".join(f"mateIn{n}" for n in range(lo, hi + 1)), na=False
        )
        df = df[mask]
        if self.max_puzzles:
            df = df.head(self.max_puzzles)

        rows = df.to_dict("records")
        random.Random(self.seed).shuffle(rows)
        n = len(rows)
        i_train, i_val = int(n * 0.8), int(n * 0.9)
        split_rows = {
            "train": rows[:i_train],
            "val": rows[i_train:i_val],
            "test": rows[i_val:],
        }[self.split]

        data_list = []
        for row in split_rows:
            uci_moves = row["Moves"].split()
            board = chess.Board(row["FEN"])
            mate_n = self._extract_mate_n(row["Themes"])
            clock = self._simulated_clock(row["Rating"])

            # una posizione-grafo per ogni mossa della soluzione (posizione -> mossa corretta)
            for ply_idx, uci in enumerate(uci_moves):
                move = chess.Move.from_uci(uci)
                legal = list(board.legal_moves)
                try:
                    best_idx = legal.index(move)
                except ValueError:
                    board.push(move)
                    continue

                label = {
                    "mate_n": mate_n,
                    "best_move_idx": best_idx,
                }
                d = board_to_pyg_data(board, clock_seconds=clock * (1 + 0.1 * ply_idx), label=label)
                d.puzzle_id = row["PuzzleId"]
                d.rating = float(row["Rating"])
                data_list.append(d)
                board.push(move)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def _extract_mate_n(themes: str) -> int:
        for t in themes.split():
            if t.startswith("mateIn"):
                return int(t.replace("mateIn", ""))
        return 0

    def _simulated_clock(self, rating: int) -> float:
        if self.avg_time_by_rating:
            bucket = round(rating / 100) * 100
            return self.avg_time_by_rating.get(bucket, 15.0)
        return 5.0 + (rating / 3000.0) * 55.0
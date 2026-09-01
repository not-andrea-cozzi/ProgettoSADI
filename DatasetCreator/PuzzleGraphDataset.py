import os
import random
from DatasetCreator.GraphBuilder import GraphBuilder
import chess
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset


def merge_and_split(puzzle_splits: dict, games_splits: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    result = {}
    for name in ("train", "val", "test"):
        merged = list(puzzle_splits.get(name, [])) + list(games_splits.get(name, []))
        random.Random(42 + hash(name) % 1000).shuffle(merged)
        result[name] = merged
        torch.save(merged, os.path.join(out_dir, f"merged_{name}.pt"))
    return result


class PuzzleGraphDataset(InMemoryDataset):
    def __init__(self, csv_path, root, split="train", mate_range=(1, 5),
                 max_puzzles=None, seed=42, avg_time_by_rating=None, chunksize=50_000):
        self.csv_path = csv_path
        self.mate_range = mate_range
        self.max_puzzles = max_puzzles
        self.seed = seed
        self.split = split
        self.avg_time_by_rating = avg_time_by_rating or {}
        self.chunksize = chunksize
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f"puzzle_{self.split}.pt"]

    def _load_filtered_rows(self) -> list[dict]:
        lo, hi = self.mate_range
        theme_pattern = "|".join(f"mateIn{n}" for n in range(lo, hi + 1))
        rows = []
        reader = pd.read_csv(self.csv_path, chunksize=self.chunksize)
        pbar = tqdm(desc=f"Lettura CSV puzzle [pool completo]", unit=" righe valide")
        
        for chunk in reader:
            mask = chunk["Themes"].str.contains(theme_pattern, na=False)
            filtered = chunk[mask]
            rows.extend(filtered.to_dict("records"))
            pbar.update(len(filtered))

            if self.max_puzzles and len(rows) >= self.max_puzzles:
                rows = rows[: self.max_puzzles]
                break
        pbar.close()
        return rows

    def _rows_for_split(self, rows: list[dict]) -> list[dict]:
        rows_sorted = sorted(rows, key=lambda r: r["PuzzleId"])
        random.Random(self.seed).shuffle(rows_sorted)
        n = len(rows_sorted)
        i_train, i_val = int(n * 0.8), int(n * 0.9)
        return {
            "train": rows_sorted[:i_train],
            "val": rows_sorted[i_train:i_val],
            "test": rows_sorted[i_val:],
        }[self.split]

    def process(self):
        all_rows = self._load_filtered_rows()
        split_rows = self._rows_for_split(all_rows)

        data_list = []
        for row in tqdm(split_rows, desc=f"Costruzione grafi puzzle [{self.split}]"):
            uci_moves = row["Moves"].split()
            if not uci_moves:
                continue
                
            board = chess.Board(row["FEN"])
            mate_n_iniziale = self._extract_mate_n(row["Themes"])
            clock = self._simulated_clock(row["Rating"])

            first_move = chess.Move.from_uci(uci_moves[0])
            if first_move in board.legal_moves:
                board.push(first_move)
            else:
                continue

            for ply_idx, uci in enumerate(uci_moves[1:], start=1):
                move = chess.Move.from_uci(uci)
                
                if ply_idx % 2 == 0:
                    if move in board.legal_moves:
                        board.push(move)
                    continue

                legal = list(board.legal_moves)
                if move not in legal:
                    break
                    
                best_idx = legal.index(move)
                current_mate_n = max(1, mate_n_iniziale - (ply_idx // 2))
                
                label = {
                    "mate_n": current_mate_n,
                    "best_move_idx": best_idx,
                }
                
                d = GraphBuilder.board_to_pyg_data(
                    board, 
                    clock_seconds=clock * (1 + 0.1 * ply_idx), 
                    label=label
                )
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
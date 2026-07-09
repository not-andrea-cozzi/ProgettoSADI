import os
import random
from Model.graph_builder import GraphBuilder
import chess
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset


def merge_and_split(puzzle_splits: dict, games_splits: dict, out_dir: str):
    """Unisce, PER-SPLIT, i Data list gia' splittati a monte (per puzzle_id/game_id)
    da PuzzleGraphDataset e ChessAnalysisPipeline. NON ri-shuffla e NON ri-splitta:
    farlo introdurrebbe leakage perche' mosse dello stesso puzzle/game finirebbero
    in split diversi. L'unico shuffle qui e' interno al singolo split (ordine dei
    batch), mai tra split.

    puzzle_splits / games_splits: dict con chiavi "train","val","test" -> list[Data]
    """
    os.makedirs(out_dir, exist_ok=True)
    result = {}
    for name in ("train", "val", "test"):
        merged = list(puzzle_splits.get(name, [])) + list(games_splits.get(name, []))
        random.Random(42 + hash(name) % 1000).shuffle(merged)  # shuffle intra-split, non cross-split
        result[name] = merged
        torch.save(merged, os.path.join(out_dir, f"merged_{name}.pt"))
    return result


class PuzzleGraphDataset(InMemoryDataset):
    """Costruisce dataset PyG da lichess_db_puzzle.csv filtrato per mateIn1-5.

    FIX split-leakage: lo split train/val/test avviene sui PuzzleId (una riga CSV =
    un puzzle = una sequenza di mosse), PRIMA di espandere ogni puzzle nelle sue
    posizioni-mossa. In precedenza lo split veniva fatto dopo l'espansione a livello
    di singola posizione/mossa in merge_and_split, cosi' mosse dello stesso puzzle
    potevano finire sia in train che in val/test (data leakage: il modello vede
    posizioni della stessa linea tattica sia in training che in validazione).

    Il CSV Lichess ha ~6M righe: legge a CHUNK e si ferma non appena ha raccolto
    max_puzzles righe valide (che matchano il tema), invece di caricare tutto il file."""

    def __init__(self, csv_path, root, split="train", mate_range=(1, 5),
                 max_puzzles=None, seed=42, avg_time_by_rating=None, chunksize=50_000):
        self.csv_path = csv_path
        self.mate_range = mate_range
        self.max_puzzles = max_puzzles
        self.seed = seed
        self.split = split  # train | val | test
        self.avg_time_by_rating = avg_time_by_rating or {}
        self.chunksize = chunksize
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f"puzzle_{self.split}.pt"]

    def _load_filtered_rows(self) -> list[dict]:
        """Legge il CSV a chunk, tiene solo righe con tema mateIn{lo..hi}, si ferma
        appena raggiunge max_puzzles righe valide (se specificato).

        NB: il filtro qui avviene su TUTTO il pool di puzzle (non ancora splittato).
        Lo split train/val/test per PuzzleId avviene subito dopo, in process()."""
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
        """Split 80/10/10 sui PuzzleId (a livello di riga CSV = puzzle intero),
        con lo stesso seed per garantire che train/val/test chiamati separatamente
        (split="train", poi split="val", ecc.) producano partizioni disgiunte e
        deterministiche sullo STESSO pool di rows."""
        rows_sorted = sorted(rows, key=lambda r: r["PuzzleId"])  # ordine deterministico pre-shuffle
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
            board = chess.Board(row["FEN"])
            mate_n = self._extract_mate_n(row["Themes"])
            clock = self._simulated_clock(row["Rating"])

            # una posizione-grafo per ogni mossa della soluzione (posizione -> mossa corretta)
            # tutte le mosse di QUESTO puzzle restano nello split assegnato sopra: nessuna
            # mossa dello stesso PuzzleId puo' finire in uno split diverso.
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
                d = GraphBuilder.board_to_pyg_data(board, clock_seconds=clock * (1 + 0.1 * ply_idx), label=label)
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
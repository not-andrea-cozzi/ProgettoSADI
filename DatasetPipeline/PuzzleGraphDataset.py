"""
PuzzleGraphDataset.py

Costruisce il dataset puzzle (schema compresso, vedi DatasetCreator/
GraphBuilder.py) a partire dal CSV Lichess puzzle standard:

    PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays,
    Themes, GameUrl, OpeningTags, DailyDate

Solo i campi utili alla predizione della best move (+ mate_n ausiliario,
+ game_id/ply per la ricostruzione delle sequenze) finiscono nel tensor
.pt: PuzzleId/GameUrl/OpeningTags/DailyDate/RatingDeviation/Popularity/
NbPlays NON entrano nel sample compresso (non servono al modello) —
vengono pero' preservati nel .jsonl di debug per audit/tracciabilita',
insieme a fen/best_move_uci.

NOTA game_id/ply (fix rispetto alla versione precedente):
Ogni puzzle riceve un game_id NAMESPACED (vedi GraphBuilder.make_game_id,
source_tag=SOURCE_TAG_PUZZLE) progressivo, assegnato in process() PRIMA
di costruire i Data del puzzle stesso: tutti i ply dello stesso puzzle
condividono lo stesso game_id. Questo campo vive DENTRO l'oggetto Data
(non solo nel .jsonl di debug), quindi sopravvive a qualunque merge/
shuffle a valle (MergeSplitter) senza bisogno di allineare file esterni.
Il game_id progressivo e' locale allo split corrente (self.split): non e'
il PuzzleId originale del CSV, che resta comunque nel .jsonl per audit.
"""
import os
import random
from typing import Tuple, Optional, Dict, Any, List

import chess
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data

from DatasetPipeline.GraphBuilder import GraphBuilder, make_game_id, SOURCE_TAG_PUZZLE


def merge_and_split(puzzle_splits: dict, games_splits: dict, out_dir: str,
                    ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Unisce puzzle_splits e games_splits, quindi suddivide in train/val/test
    bilanciando per mate_n (schema compresso: sempre un uint8 tensor
    scalare data.mate_n, nessuna ambiguita' di formato da gestire come
    nella versione precedente con tuple/liste/attributi alternativi).

    NOTA: per uno split stratificato coerente su tutto il progetto, usare
    MergeSplitter.MateNMergeSplitter invece di questa funzione legacy:
    quella classe fa lo stesso lavoro con logging della distribuzione e
    scrittura atomica (tmp + os.replace). Questa funzione resta per
    retrocompatibilita' con chiamanti esistenti.
    """
    from collections import defaultdict

    os.makedirs(out_dir, exist_ok=True)

    all_data: List[Data] = []
    for name in ("train", "val", "test"):
        all_data.extend(puzzle_splits.get(name, []))
        all_data.extend(games_splits.get(name, []))

    def get_mate_value(item: Data) -> int:
        if not hasattr(item, "mate_n"):
            raise ValueError(f"Oggetto senza attributo 'mate_n': {item}")
        val = item.mate_n
        if isinstance(val, torch.Tensor):
            return int(val.item())
        return int(val)

    groups: Dict[int, List[Data]] = defaultdict(list)
    for item in all_data:
        groups[get_mate_value(item)].append(item)

    torch.manual_seed(seed)
    train_ratio, val_ratio, _test_ratio = ratios
    train_list, val_list, test_list = [], [], []

    for _mate, items in groups.items():
        n = len(items)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_train = max(0, n_train)
            n_val = max(0, n_val)
            n_test = n - n_train - n_val

        perm = torch.randperm(n)
        shuffled = [items[i] for i in perm.tolist()]

        train_list.extend(shuffled[:n_train])
        val_list.extend(shuffled[n_train:n_train + n_val])
        test_list.extend(shuffled[n_train + n_val:])

    def shuffle_list(lst):
        if not lst:
            return lst
        perm = torch.randperm(len(lst))
        return [lst[i] for i in perm.tolist()]

    train_list = shuffle_list(train_list)
    val_list = shuffle_list(val_list)
    test_list = shuffle_list(test_list)

    for name, data in zip(("train", "val", "test"), (train_list, val_list, test_list)):
        out_path = os.path.join(out_dir, f"merged_{name}.pt")
        torch.save(data, out_path)
        print(f"Salvato {out_path} con {len(data)} elementi")

    return {"train": train_list, "val": val_list, "test": test_list}


class PuzzleGraphDataset(InMemoryDataset):
    """Dataset puzzle in schema compresso. Ogni riga del CSV con un tema
    mateInN nel range richiesto genera una sequenza di sample (uno per ogni
    ply del lato che deve dare matto), ciascuno compresso secondo lo
    schema di GraphBuilder.board_to_pyg_data. Tutti i sample dello stesso
    puzzle condividono lo stesso game_id namespaced (SOURCE_TAG_PUZZLE) e
    portano il proprio ply assoluto, cosi' la sequenza e' ricostruibile a
    valle anche dopo merge/shuffle.

    Il .jsonl di debug (fen, best_move_uci, puzzle_id, mate_n, rating,
    ply_idx, game_id) viene scritto accanto al processed_paths[0], stesso
    ordine dei sample nel .pt, per audit senza portare stringhe nel tensor
    dataset. Il game_id compare anche nel .jsonl (decodificabile con
    GraphBuilder.decode_game_id) per poter incrociare i due file durante
    il debug, ma NON e' piu' l'unica fonte di verita': se il .jsonl si
    corrompe, le sequenze restano ricostruibili dal solo .pt."""

    def __init__(self, csv_path, root, split="train", mate_range=(1, 5),
                 max_puzzles=None, seed=42, avg_time_by_rating=None, chunksize=50_000,
                 split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        self.csv_path = csv_path
        self.mate_range = mate_range
        self.max_puzzles = max_puzzles
        self.seed = seed
        self.split = split
        self.avg_time_by_rating = avg_time_by_rating or {}
        self.chunksize = chunksize

        if len(split_ratios) != 3:
            raise ValueError("split_ratios deve contenere train, val e test.")
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios deve sommare a 1.0.")
        self.split_ratios = split_ratios

        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f"puzzle_{self.split}.pt"]

    @property
    def debug_jsonl_path(self) -> str:
        return os.path.join(self.processed_dir, f"puzzle_{self.split}.jsonl")

    def _load_filtered_rows(self) -> List[dict]:
        lo, hi = self.mate_range
        theme_pattern = "|".join(f"mateIn{n}" for n in range(lo, hi + 1))
        rows = []
        reader = pd.read_csv(self.csv_path, chunksize=self.chunksize)
        pbar = tqdm(desc="Lettura CSV puzzle [pool completo]", unit=" righe valide")

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

    def _rows_for_split(self, rows: List[dict]) -> List[dict]:
        rows_sorted = sorted(rows, key=lambda r: r["PuzzleId"])
        random.Random(self.seed).shuffle(rows_sorted)
        n = len(rows_sorted)

        train_ratio, val_ratio, _ = self.split_ratios
        i_train = int(n * train_ratio)
        i_val = int(n * (train_ratio + val_ratio))

        return {
            "train": rows_sorted[:i_train],
            "val": rows_sorted[i_train:i_val],
            "test": rows_sorted[i_val:],
        }[self.split]

    def process(self):
        all_rows = self._load_filtered_rows()
        split_rows = self._rows_for_split(all_rows)

        data_list: List[Data] = []
        debug_records: List[Dict[str, Any]] = []

        # Contatore progressivo LOCALE allo split corrente: un puzzle = una
        # sequenza = un game_id. Namespaced con SOURCE_TAG_PUZZLE cosi' non
        # collide con i game_id assegnati da GamesBuilder/ClubGamesTimedBuilder
        # (source_tag diversi) quando tutto finisce nello stesso
        # merged_{split}.pt via MergeSplitter.
        local_game_counter = 0

        for row in tqdm(split_rows, desc=f"Costruzione grafi puzzle [{self.split}]"):
            uci_moves = row["Moves"].split()
            if not uci_moves:
                continue

            board = chess.Board(row["FEN"])
            mate_n_iniziale = self._extract_mate_n(row["Themes"])
            clock = self._simulated_clock(row["Rating"])
            puzzle_rating = float(row["Rating"]) if pd.notna(row.get("Rating")) else None

            first_move = chess.Move.from_uci(uci_moves[0])
            if first_move in board.legal_moves:
                board.push(first_move)
            else:
                continue

            # ply assoluto: la mossa iniziale (uci_moves[0], gia' giocata
            # sopra) e' il ply 0 della sequenza FEN; i ply successivi
            # incrementano di 1 per ogni mezza-mossa, esattamente come
            # l'indice board.ply() interno di python-chess.
            puzzle_game_id = make_game_id(SOURCE_TAG_PUZZLE, local_game_counter)
            has_any_sample = False

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

                label = {"mate_n": current_mate_n, "best_move_idx": best_idx}
                clock_seconds = clock * (1 + 0.1 * ply_idx)

                d = GraphBuilder.board_to_pyg_data(
                    board,
                    clock_seconds=clock_seconds,
                    label=label,
                    legal_moves=legal,
                    rating=puzzle_rating,
                    game_id=puzzle_game_id,
                    ply=ply_idx,
                )
                data_list.append(d)
                has_any_sample = True

                debug_records.append({
                    "problem_id": f"puzzle_{row['PuzzleId']}_{ply_idx}",
                    "puzzle_id": row["PuzzleId"],
                    "game_id": puzzle_game_id,
                    "ply": ply_idx,
                    "fen": board.fen(),
                    "best_move_uci": move.uci(),
                    "mate_n": current_mate_n,
                    "ply_idx": ply_idx,
                    "rating": puzzle_rating,
                    "rating_deviation": row.get("RatingDeviation"),
                    "popularity": row.get("Popularity"),
                    "nb_plays": row.get("NbPlays"),
                    "game_url": row.get("GameUrl"),
                    "opening_tags": row.get("OpeningTags"),
                    "clock_seconds": float(clock_seconds),
                    "source": "puzzle",
                })

                board.push(move)

            # Il contatore avanza solo se il puzzle ha prodotto almeno un
            # sample: evita di "consumare" game_id per puzzle scartati
            # subito (nessun impatto funzionale, ma tiene il namespace
            # locale piu' denso/compatto per audit).
            if has_any_sample:
                local_game_counter += 1

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        GraphBuilder.write_debug_jsonl(debug_records, self.debug_jsonl_path)

    @staticmethod
    def _extract_mate_n(themes: str) -> int:
        for t in themes.split():
            if t.startswith("mateIn"):
                return int(t.replace("mateIn", ""))
        return 0

    def _simulated_clock(self, rating: float) -> float:
        if self.avg_time_by_rating:
            bucket = round(rating / 100) * 100
            return self.avg_time_by_rating.get(bucket, 15.0)
        return 5.0 + (rating / 3000.0) * 55.0
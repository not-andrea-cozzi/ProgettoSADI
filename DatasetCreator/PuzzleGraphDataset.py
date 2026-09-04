import os
import random
from typing import Tuple
from DatasetCreator.GraphBuilder import GraphBuilder
import chess
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset


def merge_and_split(puzzle_splits: dict, games_splits: dict, out_dir: str,
                    mate_attr: str = "mate_n", ratios=(0.8, 0.1, 0.1),
                    seed=42, mate_extractor=None):
    """
    Unisce puzzle_splits e games_splits, quindi suddivide in train/val/test
    bilanciando per l'attributo 'mate' (di default 'mate_n').

    Gestisce:
      - Oggetti che sono tuple/liste (estrae il primo elemento)
      - Attributi alternativi ('mate', 'mate_in', 'y')
      - Tensori convertiti a scalare
    """
    import torch
    from collections import defaultdict
    import os

    os.makedirs(out_dir, exist_ok=True)

    # 1. Raccogli tutti i dati da entrambi i tipi di split
    all_data = []
    for name in ("train", "val", "test"):
        all_data.extend(puzzle_splits.get(name, []))
        all_data.extend(games_splits.get(name, []))

    # 2. Funzione per estrarre il grafo (se l'elemento è una tupla/lista)
    def get_graph(item):
        if isinstance(item, (tuple, list)) and len(item) > 0:
            return item[0]          # prende il primo elemento (il Data)
        return item

    # 3. Funzione per ottenere il valore del mate
    def get_mate_value(item):
        if mate_extractor is not None:
            return mate_extractor(item)

        graph = get_graph(item)

        # Prova vari nomi di attributo
        for attr in [mate_attr, "mate_n", "mate", "mate_in", "y"]:
            if hasattr(graph, attr):
                val = getattr(graph, attr)
                # Se è un tensore, convertilo a scalare intero
                if isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        val = val.item()
                    elif val.dim() == 1 and val.numel() > 1:
                        # Se è un tensore 1D, prendiamo il primo valore? 
                        # Di solito mate è scalare, ma se abbiamo per nodo, potrebbe servire una aggregazione.
                        # Qui assumiamo che il valore sia lo stesso per tutti i nodi, quindi prendiamo il primo.
                        val = val[0].item()
                    else:
                        # Se tensore multidimensionale, non sappiamo come gestirlo: solleviamo eccezione
                        raise ValueError(f"Tensore '{attr}' con forma {val.shape} non gestito.")
                # Se il valore è intero, lo restituiamo
                if isinstance(val, (int, float)):
                    return int(val)
                # Se è un tensore ma già convertito, lo abbiamo già trasformato
                return val
        raise ValueError(
            f"Nessun attributo di mate trovato nell'oggetto {type(graph)}. "
            f"Attributi disponibili: {[a for a in dir(graph) if not a.startswith('_')]}"
        )

    # 4. Raggruppa per mate
    groups = defaultdict(list)
    for item in all_data:
        mate = get_mate_value(item)
        if mate is None:
            # Se nonostante tutto il mate è None, scartiamo l'elemento (o solleviamo eccezione)
            # Qui scegliamo di sollevare per individuare il problema
            raise ValueError(f"Impossibile estrarre mate per oggetto: {item}")
        groups[mate].append(item)

    # 5. Suddivisione stratificata per gruppi
    torch.manual_seed(seed)
    train_ratio, val_ratio, test_ratio = ratios
    train_list, val_list, test_list = [], [], []

    for mate, items in groups.items():
        n = len(items)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        n_test = n - n_train - n_val
        # Correzione per arrotondamenti
        if n_test < 0:
            n_train = max(0, n_train)
            n_val = max(0, n_val)
            n_test = n - n_train - n_val

        perm = torch.randperm(n)
        shuffled = [items[i] for i in perm.tolist()]

        train_list.extend(shuffled[:n_train])
        val_list.extend(shuffled[n_train:n_train + n_val])
        test_list.extend(shuffled[n_train + n_val:])

    # 6. Mescolanza globale degli split
    def shuffle_list(lst):
        if not lst:
            return lst
        perm = torch.randperm(len(lst))
        return [lst[i] for i in perm.tolist()]

    train_list = shuffle_list(train_list)
    val_list = shuffle_list(val_list)
    test_list = shuffle_list(test_list)

    # 7. Salvataggio
    for name, data in zip(("train", "val", "test"), (train_list, val_list, test_list)):
        out_path = os.path.join(out_dir, f"merged_{name}.pt")
        torch.save(data, out_path)
        print(f"Salvato {out_path} con {len(data)} elementi")

    return {"train": train_list, "val": val_list, "test": test_list}

class PuzzleGraphDataset(InMemoryDataset):
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

        # Validazione: stessa logica usata altrove nella pipeline
        # (vedi validate_config in MainDatasetCreator.py), per coerenza
        # e per evitare split silenziosamente sbagliati.
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

        # Usa split_ratios configurabile invece dei valori 0.8/0.9
        # precedentemente hardcoded, per coerenza con split_cfg
        # applicato al resto della pipeline (games_pipeline, merge).
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
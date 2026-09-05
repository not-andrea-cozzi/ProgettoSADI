from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data

logger = logging.getLogger("merge_splitter")


class BaseMergeSplitter(ABC):
    """Classe astratta: definisce lo scheletro dell'algoritmo merge+split
    (Template Method in run()/save()) e lascia alle sottoclassi SOLO la
    scelta della chiave di stratificazione, tramite _group_key().

    Le sottoclassi non devono toccare run()/save()/_log_distribution():
    implementano unicamente _group_key(item) -> int, che determina come i
    sample vengono raggruppati prima di essere splittati proporzionalmente
    all'interno di ogni gruppo (cosi' ogni valore della chiave e'
    rappresentato nella stessa proporzione in train/val/test)."""

    def __init__(
        self,
        out_dir: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed: int = 42,
    ):
        if len(split_ratios) != 3:
            raise ValueError("split_ratios deve contenere train, val e test.")
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios deve sommare a 1.0.")
        self.out_dir = out_dir
        self.split_ratios = split_ratios
        self.seed = seed

    # ------------------------------------------------------------------
    # UNICO metodo che le sottoclassi devono implementare.
    # ------------------------------------------------------------------
    @abstractmethod
    def _group_key(self, item: Data) -> int:
        """Ritorna la chiave di stratificazione per un sample (es. il
        valore intero di mate_n). Deve essere deterministica e definita
        per ogni sample valido prodotto da GraphBuilder.board_to_pyg_data."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # TEMPLATE METHOD: algoritmo comune, le sottoclassi non lo sovrascrivono.
    # ------------------------------------------------------------------
    def run(self, *sources: Dict[str, List[Data]]) -> Dict[str, List[Data]]:
        """sources: uno o piu' dict {"train": [...], "val": [...], "test": [...]}
        (tipicamente puzzle_splits e games_splits). Vengono flattenati
        insieme e ri-splittati da zero secondo _group_key, IGNORANDO lo
        split originale di ciascuna sorgente (vedi trade-off nel docstring
        di modulo)."""
        all_data: List[Data] = []
        for source in sources:
            for split_name in ("train", "val", "test"):
                all_data.extend(source.get(split_name, []))

        if not all_data:
            raise ValueError("Nessun sample da splittare: tutte le sorgenti sono vuote.")

        groups: Dict[int, List[Data]] = defaultdict(list)
        for item in all_data:
            groups[self._group_key(item)].append(item)

        generator = torch.Generator().manual_seed(self.seed)
        train_ratio, val_ratio, _test_ratio = self.split_ratios
        train_list: List[Data] = []
        val_list: List[Data] = []
        test_list: List[Data] = []

        for key in sorted(groups.keys()):
            items = groups[key]
            n = len(items)

            n_train = min(int(train_ratio * n), n)
            n_val = min(int(val_ratio * n), n - n_train)
            n_test = n - n_train - n_val  # >= 0 per costruzione

            perm = torch.randperm(n, generator=generator)
            shuffled = [items[i] for i in perm.tolist()]

            train_list.extend(shuffled[:n_train])
            val_list.extend(shuffled[n_train:n_train + n_val])
            test_list.extend(shuffled[n_train + n_val:])

        result = {"train": train_list, "val": val_list, "test": test_list}

        # Shuffle finale per split: senza questo, ogni split resterebbe
        # ordinato a blocchi per chiave (tutti i mate_n=1, poi tutti i
        # mate_n=2, ...), il che puo' influenzare la composizione dei
        # batch se il DataLoader a valle non fa shuffle=True.
        for split_name, data_list in result.items():
            if not data_list:
                continue
            perm = torch.randperm(len(data_list), generator=generator)
            result[split_name] = [data_list[i] for i in perm.tolist()]

        self._log_distribution(result)
        return result

    def _log_distribution(self, result: Dict[str, List[Data]]) -> None:
        for split_name, data_list in result.items():
            counts: Dict[int, int] = defaultdict(int)
            for item in data_list:
                counts[self._group_key(item)] += 1
            total = len(data_list)
            logger.info(f"[{self.__class__.__name__}] split={split_name} totale={total:,}")
            for key in sorted(counts.keys()):
                c = counts[key]
                pct = 100 * c / total if total else 0.0
                logger.info(f"    chiave={key}: {c:,} ({pct:.1f}%)")

    def save(self, result: Dict[str, List[Data]]) -> Dict[str, str]:
        """Salva ogni split come merged_{split}.pt in out_dir, con lo
        stesso pattern write-tmp+os.replace usato altrove nella pipeline
        per evitare file parziali in caso di crash a meta' scrittura."""
        os.makedirs(self.out_dir, exist_ok=True)
        paths: Dict[str, str] = {}
        for split_name, data_list in result.items():
            out_path = os.path.join(self.out_dir, f"merged_{split_name}.pt")
            tmp_path = out_path + ".tmp"
            torch.save(data_list, tmp_path)
            os.replace(tmp_path, out_path)
            paths[split_name] = out_path
        return paths


class MateNMergeSplitter(BaseMergeSplitter):
    """Stratifica per mate_n: ogni valore di mate_n viene splittato
    train/val/test separatamente secondo split_ratios, cosi' la
    proporzione di ciascun mate_n resta la stessa nei tre split finali.
    E' il criterio richiesto dal progetto (vedi docstring di modulo),
    NON il rating."""

    def _group_key(self, item: Data) -> int:
        if not hasattr(item, "mate_n"):
            raise ValueError(f"Sample senza attributo 'mate_n', impossibile stratificare: {item}")
        val = item.mate_n
        return int(val.item()) if isinstance(val, torch.Tensor) else int(val)
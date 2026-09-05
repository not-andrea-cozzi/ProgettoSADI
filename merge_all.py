from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from DatasetPipeline.MergeSplitter import MateNMergeSplitter
from DatasetPipeline.PipelineState import file_ready

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("merge_all")

_TRIPLE_RE = re.compile(r"^(?P<prefix>.+)_(train|val|test)\.pt$")


def discover_split_triples(directory: str) -> Dict[str, Dict[str, str]]:
    """Trova tutte le triple {prefix}_train.pt/_val.pt/_test.pt in una
    directory, raggruppate per prefix. Una tripla incompleta (solo 1 o 2
    dei tre split presenti) viene ignorata con un warning esplicito,
    invece di essere inclusa parzialmente in silenzio."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory non trovata: {directory}")

    grouped: Dict[str, Dict[str, str]] = defaultdict(dict)
    for fname in sorted(os.listdir(directory)):
        m = _TRIPLE_RE.match(fname)
        if not m:
            continue
        grouped[m.group("prefix")][m.group(2)] = os.path.join(directory, fname)

    complete: Dict[str, Dict[str, str]] = {}
    for prefix, paths in grouped.items():
        if len(paths) == 3:
            complete[prefix] = paths
        else:
            logger.warning(
                f"{directory}: tripla incompleta per prefix '{prefix}' "
                f"({sorted(paths.keys())}), ignorata."
            )
    return complete


def load_pool(paths: Dict[str, str]) -> Dict[str, List]:
    pool: Dict[str, List] = {}
    for split, path in paths.items():
        if not file_ready(path):
            logger.warning(f"{path} assente o vuoto, split '{split}' trattato come [].")
            pool[split] = []
            continue
        pool[split] = torch.load(path, weights_only=False)
    return pool


def collect_pools(directories: List[str], out_dir: str) -> Tuple[List[Dict[str, List]], List[str]]:
    """Scansiona tutte le directory indicate e ritorna la lista dei pool
    trovati + la lista human-readable delle sorgenti (per il log)."""
    out_dir_abs = os.path.abspath(out_dir)
    pools: List[Dict[str, List]] = []
    labels: List[str] = []

    for directory in directories:
        triples = discover_split_triples(directory)
        if not triples:
            logger.warning(f"Nessuna tripla train/val/test trovata in {directory}, saltata.")
            continue

        if os.path.abspath(directory) == out_dir_abs:
            logger.warning(
                f"ATTENZIONE: la sorgente '{directory}' coincide esattamente con la "
                f"directory di output '{out_dir}'. Se questo merge era gia' stato "
                f"eseguito in passato con lo stesso --out-dir, questi sample "
                f"potrebbero essere conteggiati due volte. Vedi il docstring del modulo."
            )

        for prefix, paths in triples.items():
            pools.append(load_pool(paths))
            labels.append(f"{directory}/{prefix}_*.pt")

    return pools, labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combina piu' pool di .pt (train/val/test) in uno solo, "
                     "ri-splittando stratificato per mate_n."
    )
    p.add_argument(
        "directories", nargs="+",
        help="Una o piu' directory da cui pescare triple *_train.pt/_val.pt/_test.pt "
             "(es. Dataset/Train Dataset/Games).",
    )
    p.add_argument(
        "--out-dir", required=True,
        help="Directory di output per merged_train.pt / merged_val.pt / merged_test.pt. "
             "Puo' coincidere con una directory di input (sovrascrittura intenzionale), "
             "ma vedi l'avviso sul doppio conteggio nel docstring del modulo.",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

    pools, labels = collect_pools(args.directories, args.out_dir)
    if not pools:
        logger.error("Nessun pool trovato nelle directory indicate, niente da fare.")
        sys.exit(2)

    logger.info("Sorgenti trovate:")
    for label, pool in zip(labels, pools):
        n = sum(len(v) for v in pool.values())
        logger.info(f"  {label}: {n:,} campioni totali (train/val/test)")

    splitter = MateNMergeSplitter(out_dir=args.out_dir, split_ratios=split_ratios, seed=args.seed)
    merged = splitter.run(*pools)
    paths = splitter.save(merged)

    for split, path in paths.items():
        n = len(merged[split])
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"Salvato {split}: {n:,} campioni in {path} ({size_mb:.2f} MB)")

    logger.info("Merge completato.")


if __name__ == "__main__":
    main()
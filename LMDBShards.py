"""
LMDBShards.py

Converte merged_{split}.pt in un DB LMDB memory-mapped e fornisce il
Dataset per leggerlo. Elimina il collo di bottiglia di caricamento per
dataset grandi (2.6GB+): mmap, no torch.load() completo in RAM, random
access O(1), page cache condivisa tra worker.

Build:
    python LMDBShards.py build --data-dir Dataset/Train --splits train val

Uso in training:
    from LMDBShards import PuzzleLMDBDataset
    train_dataset = PuzzleLMDBDataset("Dataset/Train/lmdb_train")
    loader = DataLoader(train_dataset, batch_size=64, num_workers=4,
                         collate_fn=timed_collate_fn, persistent_workers=True)
"""
import argparse
import io
import os
import pickle
import time
from typing import Optional

import lmdb
import torch
from torch.utils.data import Dataset


# ============================================================
# BUILD
# ============================================================
def _serialize(seq) -> bytes:
    buf = io.BytesIO()
    torch.save(seq, buf)
    return buf.getvalue()


def build_lmdb(src_pt: str, out_dir: str, map_size_gb: float = 8.0):
    if not os.path.exists(src_pt):
        print(f"[SKIP] {src_pt} non trovato")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"Caricamento {src_pt}...")
    t0 = time.time()
    data_list = torch.load(src_pt, weights_only=False, map_location="cpu")
    print(f"Caricato in {time.time()-t0:.1f}s ({len(data_list)} sequenze)")

    map_size = int(map_size_gb * (1024 ** 3))
    env = lmdb.open(out_dir, map_size=map_size, subdir=True, readonly=False,
                     meminit=False, map_async=True)

    t0 = time.time()
    with env.begin(write=True) as txn:
        for idx, seq in enumerate(data_list):
            key = f"{idx:010d}".encode("ascii")
            txn.put(key, _serialize(seq))
            if (idx + 1) % 5000 == 0:
                print(f"  {idx+1}/{len(data_list)} scritti ({time.time()-t0:.1f}s)")
        txn.put(b"__len__", pickle.dumps(len(data_list)))

    env.sync()
    env.close()
    print(f"Fatto: {len(data_list)} sequenze -> {out_dir} ({time.time()-t0:.1f}s)")


# ============================================================
# DATASET
# ============================================================
class PuzzleLMDBDataset(Dataset):
    def __init__(self, lmdb_dir: str):
        self.lmdb_dir = lmdb_dir
        self._env: Optional[lmdb.Environment] = None

        env = lmdb.open(lmdb_dir, readonly=True, lock=False, subdir=True,
                         readahead=False, meminit=False)
        with env.begin() as txn:
            self._len = pickle.loads(txn.get(b"__len__"))
        env.close()

    def _get_env(self) -> lmdb.Environment:
        # Lazy-open: ogni worker apre il proprio handle mmap dopo il fork.
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_dir, readonly=True, lock=False, subdir=True,
                readahead=True, meminit=False, max_readers=256,
            )
        return self._env

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        env = self._get_env()
        key = f"{idx:010d}".encode("ascii")
        with env.begin(buffers=True) as txn:
            raw = txn.get(key)
            if raw is None:
                raise IndexError(f"Indice {idx} non trovato in {self.lmdb_dir}")
            buf = io.BytesIO(bytes(raw))
        return torch.load(buf, weights_only=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--data-dir", default="Dataset/Train")
    b.add_argument("--splits", nargs="+", default=["train", "val"])
    b.add_argument("--map-size-gb", type=float, default=8.0)

    args = ap.parse_args()

    if args.cmd == "build":
        for split in args.splits:
            src = os.path.join(args.data_dir, f"merged_{split}.pt")
            out = os.path.join(args.data_dir, f"lmdb_{split}")
            build_lmdb(src, out, map_size_gb=args.map_size_gb)
        print("\nCompletato.")


if __name__ == "__main__":
    main()
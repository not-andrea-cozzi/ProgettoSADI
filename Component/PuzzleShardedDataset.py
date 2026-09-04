"""
Dataset shardato per sequenze di puzzle.
Ogni __getitem__ restituisce UNA sequenza (lista di plies), esattamente come
PuzzleSequenceDataset originale. Il timed_collate_fn esistente funziona senza modifiche.

Uso:
    from Component.PuzzleShardedDataset import PuzzleShardedDataset
    train_dataset = PuzzleShardedDataset("Dataset/Train/shards_train", cache_size=3)
"""
import os
from collections import OrderedDict
from typing import Optional

import torch
from torch.utils.data import Dataset


class PuzzleShardedDataset(Dataset):
    """
    Legge sequenze di puzzle da shard su disco.
    Ogni worker mantiene una cache LRU di 'cache_size' shard in RAM.

    Con seqs_per_shard=500 e cache_size=3:
      ~30-70MB × 3 shard × N workers  ≪  2.2GB × N workers.
    """

    def __init__(self, shard_dir: str, cache_size: int = 3):
        self.shard_dir = shard_dir
        self.cache_size = max(1, cache_size)

        index_path = os.path.join(shard_dir, "index.pt")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"{index_path} non trovato. Hai lanciato ShardDataset.py?"
            )
        meta = torch.load(index_path, weights_only=False, map_location="cpu")
        # index[i] = (shard_id, offset_in_shard), dove ogni elemento è UNA sequenza
        self._index = meta["index"]
        self._n_sequences = meta["n_sequences"]
        self._n_shards = meta["n_shards"]

        # Cache LRU per worker (inizializzata lazy → niente pickle di roba grossa allo spawn)
        self._cache: Optional[OrderedDict] = None

    def __len__(self) -> int:
        return self._n_sequences

    def _get_cache(self) -> OrderedDict:
        if self._cache is None:
            self._cache = OrderedDict()
        return self._cache

    def _load_shard(self, shard_id: int):
        cache = self._get_cache()
        if shard_id in cache:
            cache.move_to_end(shard_id)  # LRU: marca come recente
            return cache[shard_id]

        shard_path = os.path.join(self.shard_dir, f"shard_{shard_id:05d}.pt")
        shard = torch.load(shard_path, weights_only=False, map_location="cpu")
        cache[shard_id] = shard

        while len(cache) > self.cache_size:
            cache.popitem(last=False)  # rimuove il meno recente

        return shard

    def __getitem__(self, idx: int):
        shard_id, offset = self._index[idx]
        shard = self._load_shard(shard_id)
        # shard[offset] è UNA sequenza (lista di plies), stesso formato di
        # PuzzleSequenceDataset.__getitem__(idx) → compatibile con timed_collate_fn
        return shard[offset]

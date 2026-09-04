"""
Splitta merged_train.pt / merged_val.pt in shard di SEQUENZE, normalizzando
gli schemi eterogenei di games e puzzle Lichess.

Normalizzazione unificata:
  1. best_move_idx -> y (per games)
  2. y e mate_n forzati a shape (1,) 1D per evitare mismatch di dimensioni nel Batch collate
  3. clock_seconds: se manca (puzzle) -> stimato dal rating via avg_time_by_rating.json
  4. x[:, 3] = clock_norm ricalcolato coerentemente (log1p(sec) / log1p(600))

Uso:
    python ShardDataset.py
    python ShardDataset.py --seqs-per-shard 5000
"""
import argparse
import json
import math
import os
import time
from collections import Counter

import torch

MAX_MATE_N = 10
CLOCK_CAP_SECONDS = 600.0
_LOG_DENOM = math.log1p(CLOCK_CAP_SECONDS)


# =========================================================
# Rating -> seconds (LUT con interpolazione lineare)
# =========================================================
class RatingTimeLUT:
    def __init__(self, json_path):
        self.buckets = None
        if not json_path or not os.path.exists(json_path):
            print(f"[LUT] {json_path} non trovato: fallback a 10.0s costanti.")
            return
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            self.buckets = sorted((int(k), float(v)) for k, v in data.items())
        except Exception as e:
            print(f"[LUT] Errore parsing {json_path} ({e}): fallback a 10.0s costanti.")
            self.buckets = None
            return
        print(f"[LUT] Caricati {len(self.buckets)} bucket rating da {json_path} "
              f"(range {self.buckets[0][0]}..{self.buckets[-1][0]}, "
              f"tempi {self.buckets[0][1]:.1f}s..{self.buckets[-1][1]:.1f}s)")

    def seconds_for(self, rating):
        if self.buckets is None:
            return 10.0
        if rating is None:
            return self.buckets[len(self.buckets) // 2][1]
        r = float(rating)
        if r <= self.buckets[0][0]:
            return self.buckets[0][1]
        if r >= self.buckets[-1][0]:
            return self.buckets[-1][1]
        for i in range(len(self.buckets) - 1):
            r0, s0 = self.buckets[i]
            r1, s1 = self.buckets[i + 1]
            if r0 <= r <= r1:
                if r1 == r0:
                    return s0
                t = (r - r0) / (r1 - r0)
                return s0 + t * (s1 - s0)
        return self.buckets[-1][1]


# =========================================================
# Helpers
# =========================================================
def _seconds_to_clock_norm(seconds: float) -> float:
    if _LOG_DENOM <= 0:
        return 0.0
    return max(0.0, min(1.0, math.log1p(max(0.0, seconds)) / _LOG_DENOM))


def _to_scalar_tensor_1d(val, dtype=torch.long):
    """Forza qualsiasi valore (int, float, tensor) a tensor shape (1,) 1D."""
    if not torch.is_tensor(val):
        val = torch.tensor(val, dtype=dtype)
    val = val.reshape(-1)  # appiattisce qualsiasi shape
    if val.numel() == 0:
        val = torch.zeros(1, dtype=dtype)
    elif val.numel() > 1:
        val = val[:1]  # tieni solo il primo
    return val.to(dtype)


def _get_rating(d):
    r = getattr(d, "rating", None)
    if r is None:
        return None
    if torch.is_tensor(r):
        r = r.item() if r.numel() == 1 else r.flatten()[0].item()
    return float(r)


def _has_valid_clock(d):
    if not hasattr(d, "clock_seconds") or d.clock_seconds is None:
        return False
    val = d.clock_seconds
    if torch.is_tensor(val):
        val = val.item() if val.numel() == 1 else val.flatten()[0].item()
    return float(val) > 0.0


# =========================================================
# Normalizzazione (con shape uniforme!)
# =========================================================
def _normalize(d, lut: RatingTimeLUT, stats: dict):
    # 1. Target y: unifica shape a (1,)
    y_source = None
    if hasattr(d, "y") and d.y is not None:
        y_source = d.y
    elif hasattr(d, "best_move_idx"):
        y_source = d.best_move_idx
    if y_source is not None:
        d.y = _to_scalar_tensor_1d(y_source, dtype=torch.long)

    # 2. mate_n a shape (1,)
    if hasattr(d, "mate_n") and d.mate_n is not None:
        d.mate_n = _to_scalar_tensor_1d(d.mate_n, dtype=torch.long)

    # 3. clock_seconds
    if _has_valid_clock(d):
        stats["clock_real"] += 1
        cs = d.clock_seconds
        sec = cs.item() if torch.is_tensor(cs) and cs.numel() == 1 \
              else (cs.flatten()[0].item() if torch.is_tensor(cs) else float(cs))
    else:
        rating = _get_rating(d)
        sec = lut.seconds_for(rating)
        if rating is None:
            stats["clock_default"] += 1
        else:
            stats["clock_simulated"] += 1
    # clock_seconds anche lui shape (1,) float
    d.clock_seconds = torch.tensor([sec], dtype=torch.float)

    # 4. aggiorna x[:, 3] (clock_norm) coerente
    if hasattr(d, "x") and d.x is not None and d.x.numel() > 0 and d.x.size(1) > 3:
        d.x[:, 3] = _seconds_to_clock_norm(sec)

    # 5. rating (se presente e usato) shape (1,) float
    if hasattr(d, "rating") and d.rating is not None:
        d.rating = _to_scalar_tensor_1d(d.rating, dtype=torch.float)

    return d


# =========================================================
# Raggruppamento in sequenze
# =========================================================
def _get_group_key(d):
    gid = getattr(d, "game_id", None)
    pid = getattr(d, "problem_id", None)
    if gid is not None and pid is not None:
        if isinstance(gid, list):
            gid = tuple(gid) if len(gid) > 1 else gid[0]
        return ("game", gid, pid)

    puzz = getattr(d, "puzzle_id", None)
    if puzz is not None:
        if isinstance(puzz, list):
            puzz = tuple(puzz) if len(puzz) > 1 else puzz[0]
        return ("puzzle", puzz)

    return None


def _group_sequences(data_list):
    by_key = {}
    skipped = 0
    for d in data_list:
        key = _get_group_key(d)
        if key is None:
            skipped += 1
            continue
        by_key.setdefault(key, []).append(d)

    sequences = []
    for plies in by_key.values():
        plies_sorted = sorted(
            plies,
            key=lambda d: getattr(d, "ply", getattr(d, "move_idx", 0))
        )
        sequences.append(plies_sorted)

    return sequences, skipped


def _count_types(data_list):
    n_games, n_puzzles, n_other = 0, 0, 0
    for d in data_list:
        if hasattr(d, "game_id") and hasattr(d, "best_move_idx"):
            n_games += 1
        elif hasattr(d, "puzzle_id"):
            n_puzzles += 1
        else:
            n_other += 1
    return n_games, n_puzzles, n_other


# =========================================================
# Main routine
# =========================================================
def shard_split(data_dir: str, name: str, seqs_per_shard: int,
                filter_mate: bool, lut: RatingTimeLUT):
    src_path = os.path.join(data_dir, f"merged_{name}.pt")
    if not os.path.exists(src_path):
        print(f"[SKIP] {src_path} non trovato")
        return

    out_dir = os.path.join(data_dir, f"shards_{name}")
    os.makedirs(out_dir, exist_ok=True)

    # Pulisce shard vecchi
    for f in os.listdir(out_dir):
        if f.startswith("shard_") and f.endswith(".pt"):
            os.remove(os.path.join(out_dir, f))
    idx_old = os.path.join(out_dir, "index.pt")
    if os.path.exists(idx_old):
        os.remove(idx_old)

    print(f"\n[{name}] Caricamento {src_path}...")
    t0 = time.time()
    data_list = torch.load(src_path, weights_only=False, map_location="cpu")
    print(f"[{name}] Caricato in {time.time()-t0:.1f}s ({len(data_list)} posizioni totali)")

    ng, np_, no = _count_types(data_list)
    print(f"[{name}] Composizione: {ng} games, {np_} puzzle Lichess, {no} altro")

    # Filtra mate_n troppo alto (prima di normalizzare per evitare cast inutili)
    if filter_mate:
        before = len(data_list)
        def _mate_ok(d):
            mn = getattr(d, "mate_n", 0)
            if torch.is_tensor(mn):
                mn = mn.item() if mn.numel() == 1 else mn.flatten()[0].item()
            return int(mn) <= MAX_MATE_N
        data_list = [d for d in data_list if _mate_ok(d)]
        print(f"[{name}] Filtro mate_n > {MAX_MATE_N}: {before} -> {len(data_list)}")

    # Normalizza (y, mate_n, clock_seconds, x[:,3], rating)
    print(f"[{name}] Normalizzazione (shape uniforme + timing simulato)...")
    stats = {"clock_real": 0, "clock_simulated": 0, "clock_default": 0}
    t0 = time.time()
    data_list = [_normalize(d, lut, stats) for d in data_list]
    print(f"[{name}] Normalizzato in {time.time()-t0:.1f}s. "
          f"clock reale={stats['clock_real']}, "
          f"simulato da rating={stats['clock_simulated']}, "
          f"default (no rating)={stats['clock_default']}")

    # Sanity check: verifica shape uniformi dopo normalizzazione
    if data_list:
        d0 = data_list[0]
        print(f"[{name}] Sanity check primo campione:")
        print(f"    y.shape={tuple(d0.y.shape) if hasattr(d0,'y') and d0.y is not None else 'N/A'} "
              f"mate_n.shape={tuple(d0.mate_n.shape) if hasattr(d0,'mate_n') else 'N/A'} "
              f"clock_seconds.shape={tuple(d0.clock_seconds.shape) if hasattr(d0,'clock_seconds') else 'N/A'}")

    # Raggruppa in sequenze
    print(f"[{name}] Raggruppamento in sequenze...")
    t0 = time.time()
    sequences, skipped = _group_sequences(data_list)
    print(f"[{name}] {len(sequences)} sequenze da {len(data_list)} posizioni "
          f"({skipped} scartate senza id) in {time.time()-t0:.1f}s")

    len_dist = Counter(len(s) for s in sequences)
    print(f"[{name}] Distribuzione lunghezza sequenze (top 5):")
    for L, n in sorted(len_dist.items())[:5]:
        print(f"    len={L}: {n} sequenze")
    total_multi = sum(n for L, n in len_dist.items() if L > 1)
    print(f"[{name}] Sequenze con >1 ply (usano chain temporale): {total_multi}")

    del data_list

    # Scrivi shard
    n_shards = (len(sequences) + seqs_per_shard - 1) // seqs_per_shard
    print(f"[{name}] Scrittura di {n_shards} shard da ~{seqs_per_shard} sequenze in {out_dir}...")

    t0 = time.time()
    for shard_id in range(n_shards):
        start = shard_id * seqs_per_shard
        end = min(start + seqs_per_shard, len(sequences))
        shard = sequences[start:end]
        shard_path = os.path.join(out_dir, f"shard_{shard_id:05d}.pt")
        torch.save(shard, shard_path)
        if (shard_id + 1) % 10 == 0 or shard_id == n_shards - 1:
            print(f"  {shard_id+1}/{n_shards} shard scritti ({time.time()-t0:.1f}s)")

    # Indice
    index = []
    for shard_id in range(n_shards):
        start = shard_id * seqs_per_shard
        end = min(start + seqs_per_shard, len(sequences))
        for offset in range(end - start):
            index.append((shard_id, offset))

    torch.save(
        {
            "index": index,
            "n_sequences": len(sequences),
            "n_shards": n_shards,
            "seqs_per_shard": seqs_per_shard,
        },
        os.path.join(out_dir, "index.pt"),
    )
    print(f"[{name}] Fatto.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="Dataset/Train")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--seqs-per-shard", type=int, default=5000)
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--time-stats-json", default="Dataset/avg_time_by_rating.json")
    args = ap.parse_args()

    lut = RatingTimeLUT(args.time_stats_json)

    for split in args.splits:
        shard_split(args.data_dir, split, args.seqs_per_shard,
                    filter_mate=not args.no_filter, lut=lut)

    print("\nShardatura completata.")


if __name__ == "__main__":
    main()

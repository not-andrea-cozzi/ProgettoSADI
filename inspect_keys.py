"""
inspect_keys.py

Verifica se TUTTI gli elementi di merged_{split}.pt (ognuno un singolo
oggetto torch_geometric.data.Data, cioe' una singola posizione) condividono
lo stesso set di keys/attributi, o se ci sono elementi con keys diverse
(spesso sintomo di dati mescolati da fonti diverse, es. puzzle vs games,
che il merge non ha normalizzato).

IMPORTANTE: merged_{split}.pt e' una lista PIATTA di oggetti Data singoli
(una posizione per elemento), non una lista di sequenze/ply. Il
raggruppamento in sequenze (per game_id/problem_id) avviene solo dopo, in
Component/PuzzleSequenceDataset.group_puzzle_sequences.

Poi stampa per intero un elemento di esempio.

Uso:
    python inspect_keys.py Dataset/Train/merged_train.pt
    python inspect_keys.py Dataset/Train/merged_train.pt --sample 50000
"""
import argparse
import io
import time
from collections import Counter
from typing import Any, FrozenSet

import torch


def _keyset(data: Any) -> FrozenSet[str]:
    """Set di chiavi PyG Data + attributi extra non-tensor attaccati
    all'oggetto (es. game_id, problem_id, fen, ...), come frozenset cosi'
    e' hashable e confrontabile/contabile con Counter."""
    try:
        pyg_keys = set(data.keys())
    except Exception:
        return frozenset({f"__NOT_A_DATA_OBJECT__:{type(data).__name__}"})

    extra_attrs = set()
    for a in dir(data):
        if a.startswith("_"):
            continue
        if a in pyg_keys:
            continue
        try:
            val = getattr(data, a)
        except Exception:
            continue
        if callable(val):
            continue
        extra_attrs.add(a)

    return frozenset(pyg_keys | extra_attrs)


def _serialized_size(obj: Any) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.tell()


def _describe_data(data: Any, indent: str = "  ") -> None:
    try:
        pyg_keys = list(data.keys())
    except Exception:
        print(f"{indent}[non e' un oggetto PyG Data] type={type(data)} value={data!r}")
        return

    for k in pyg_keys:
        v = data[k]
        if torch.is_tensor(v):
            print(f"{indent}{k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"{indent}{k}: {type(v).__name__} = {v!r}")

    extra_attrs = sorted(
        a for a in dir(data)
        if not a.startswith("_") and a not in pyg_keys and not callable(getattr(data, a, None))
    )
    for a in extra_attrs:
        val = getattr(data, a)
        if torch.is_tensor(val):
            print(f"{indent}[extra] {a}: tensor shape={tuple(val.shape)} dtype={val.dtype}")
        else:
            print(f"{indent}[extra] {a}: {type(val).__name__} = {val!r}")


def inspect(pt_path: str, sample: int = None) -> None:
    print(f"Caricamento {pt_path}...")
    t0 = time.time()
    data_list = torch.load(pt_path, weights_only=False, map_location="cpu")
    total = len(data_list)
    print(f"Caricato in {time.time() - t0:.1f}s ({total} elementi totali, "
          f"ognuno un singolo Data = una posizione)\n")

    indices = range(total) if sample is None else range(min(sample, total))
    n_checked = len(indices)

    keyset_counts: Counter = Counter()
    first_idx_with_keyset = {}
    sizes = []

    t0 = time.time()
    for i in indices:
        data = data_list[i]
        ks = _keyset(data)
        keyset_counts[ks] += 1
        if ks not in first_idx_with_keyset:
            first_idx_with_keyset[ks] = i

        size_bytes = _serialized_size(data)
        sizes.append((i, size_bytes))

        if (i + 1) % 20000 == 0:
            print(f"  {i+1}/{n_checked} elementi controllati ({time.time()-t0:.1f}s)")

    print(f"Controllo completato in {time.time()-t0:.1f}s su {n_checked} elementi.\n")

    print("=" * 70)
    print(f"KEYSET DEGLI ELEMENTI ({len(keyset_counts)} varianti distinte trovate)")
    print("=" * 70)

    if len(keyset_counts) == 1:
        print("Tutti gli elementi hanno esattamente lo stesso set di keys. Nessuna inconsistenza.")
    else:
        print("ATTENZIONE: trovate keyset diverse tra gli elementi. Dettaglio:\n")

    for ks, count in sorted(keyset_counts.items(), key=lambda x: -x[1]):
        example_idx = first_idx_with_keyset[ks]
        print(f"  {count:>8} elementi con questo set ({len(ks)} keys) | "
              f"primo trovato in idx={example_idx}")
        print(f"    keys: {sorted(ks)}")
        print()

    if len(keyset_counts) > 1:
        all_keysets = list(keyset_counts.keys())
        common = set.intersection(*[set(k) for k in all_keysets])
        union = set.union(*[set(k) for k in all_keysets])
        print(f"  Keys comuni a TUTTE le varianti: {sorted(common)}")
        print(f"  Keys presenti in ALMENO una variante ma non in tutte: {sorted(union - common)}")
        print()

    print("=" * 70)
    print("DIMENSIONE ELEMENTI (serializzati singolarmente)")
    print("=" * 70)
    sizes_sorted = sorted(sizes, key=lambda x: -x[1])
    mean_size = sum(s for _, s in sizes) / len(sizes)
    median_size = sorted(s for _, s in sizes)[len(sizes) // 2]
    print(f"  media: {mean_size/1024:.2f} KB | mediana: {median_size/1024:.2f} KB | "
          f"max: {sizes_sorted[0][1]/1024:.2f} KB (idx={sizes_sorted[0][0]})")
    print(f"  Top 5 piu' grandi:")
    for idx, size_bytes in sizes_sorted[:5]:
        print(f"    idx={idx}: {size_bytes/1024:.2f} KB")
    print()

    # --------------------------------------------------------------
    # Stampa per intero un elemento di esempio (il primo).
    # --------------------------------------------------------------
    print("=" * 70)
    print("ESEMPIO COMPLETO: elemento idx=0 (un singolo Data)")
    print("=" * 70)
    _describe_data(data_list[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pt_path")
    ap.add_argument("--sample", type=int, default=None,
                     help="Controlla solo i primi N elementi invece di tutto il file (piu' veloce)")
    args = ap.parse_args()
    inspect(args.pt_path, args.sample)
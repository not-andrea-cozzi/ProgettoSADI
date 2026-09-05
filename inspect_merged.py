"""
inspect_merged.py

Diagnostica su merged_{split}.pt: trova sequenze anomale per dimensione e
ne stampa la struttura dettagliata (shape/dtype per ogni tensore, per ogni
ply), per capire se il rigonfiamento (es. 0.688 GB per una sola sequenza)
e' un bug a monte (duplicazione, shape sbagliata, ecc.) o dati legittimi.

Uso:
    python inspect_merged.py Dataset/Train/merged_train.pt
    python inspect_merged.py Dataset/Train/merged_train.pt --top 5
    python inspect_merged.py Dataset/Train/merged_train.pt --idx 12345
"""
import argparse
import io
import time
from typing import Any, List

import torch


def _serialized_size(obj: Any) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.tell()


def _describe_ply(ply: Any, indent: str = "    ") -> List[str]:
    lines = []
    try:
        keys = list(ply.keys())
    except Exception:
        lines.append(f"{indent}[non e' un oggetto PyG Data / non ha .keys()] type={type(ply)}")
        return lines

    for k in keys:
        v = ply[k]
        if torch.is_tensor(v):
            lines.append(
                f"{indent}{k}: tensor shape={tuple(v.shape)} dtype={v.dtype} "
                f"numel={v.numel()} bytes\u2248{v.numel() * v.element_size()}"
            )
        else:
            lines.append(f"{indent}{k}: {type(v).__name__} = {v!r}"[:200])

    extra_attrs = [
        a for a in dir(ply)
        if not a.startswith("_")
        and a not in keys
        and not callable(getattr(ply, a, None))
    ]
    known_props = {
        "num_nodes", "num_edges", "num_edge_features", "num_node_features",
        "num_features", "is_directed", "is_undirected", "is_coalesced",
    }
    for a in extra_attrs:
        if a in known_props:
            continue
        try:
            val = getattr(ply, a)
        except Exception:
            continue
        if torch.is_tensor(val):
            lines.append(
                f"{indent}[extra] {a}: tensor shape={tuple(val.shape)} dtype={val.dtype}"
            )
        else:
            lines.append(f"{indent}[extra] {a}: {type(val).__name__} = {val!r}"[:200])

    return lines


def inspect(pt_path: str, top: int, explicit_idx: int = None) -> None:
    print(f"Caricamento {pt_path}...")
    t0 = time.time()
    data_list = torch.load(pt_path, weights_only=False, map_location="cpu")
    print(f"Caricato in {time.time() - t0:.1f}s ({len(data_list)} sequenze totali)\n")

    if explicit_idx is not None:
        targets = [explicit_idx]
    else:
        print(f"Misurazione dimensione di tutte le {len(data_list)} sequenze "
              f"(puo' richiedere tempo)...")
        t0 = time.time()
        sizes = []
        for i, seq in enumerate(data_list):
            sizes.append((i, _serialized_size(seq), len(seq) if hasattr(seq, "__len__") else -1))
            if (i + 1) % 20000 == 0:
                print(f"  {i+1}/{len(data_list)} misurate ({time.time()-t0:.1f}s)")
        print(f"Misurazione completata in {time.time()-t0:.1f}s\n")

        sizes.sort(key=lambda x: x[1], reverse=True)

        print(f"{'idx':>8} {'size_GB':>10} {'n_ply':>8}")
        for idx, size_bytes, n_ply in sizes[:top]:
            print(f"{idx:>8} {size_bytes/(1024**3):>10.4f} {n_ply:>8}")

        mean_size = sum(s for _, s, _ in sizes) / len(sizes)
        median_size = sorted(s for _, s, _ in sizes)[len(sizes) // 2]
        print(f"\nDimensione media: {mean_size/(1024**2):.2f} MB | "
              f"mediana: {median_size/(1024**2):.2f} MB | "
              f"max: {sizes[0][1]/(1024**2):.2f} MB")

        targets = [sizes[0][0]]
        print(f"\nDettaglio della sequenza piu' grande (idx={targets[0]}):\n")

    for idx in targets:
        seq = data_list[idx]
        n_ply = len(seq) if hasattr(seq, "__len__") else "?"
        seq_bytes = _serialized_size(seq)
        print(f"=== Sequenza idx={idx} | {n_ply} ply | {seq_bytes/(1024**3):.4f} GB ===")

        if not hasattr(seq, "__len__"):
            print(f"  [non e' una lista/sequenza di ply] type={type(seq)}")
            print(_describe_ply(seq))
            continue

        # Mostra il primo, un centrale, e l'ultimo ply per non stampare
        # migliaia di righe se la sequenza e' molto lunga.
        sample_positions = sorted(set([0, len(seq) // 2, len(seq) - 1]))
        for pos in sample_positions:
            ply = seq[pos]
            ply_bytes = _serialized_size(ply)
            print(f"  --- ply[{pos}] ({ply_bytes/(1024**2):.3f} MB) ---")
            for line in _describe_ply(ply):
                print(line)
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pt_path")
    ap.add_argument("--top", type=int, default=10, help="Quante sequenze piu' grandi elencare")
    ap.add_argument("--idx", type=int, default=None, help="Ispeziona un indice specifico invece di scansionare tutto")
    args = ap.parse_args()
    inspect(args.pt_path, args.top, args.idx)
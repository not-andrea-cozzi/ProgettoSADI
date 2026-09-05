"""
ShardDataset.py

Converte merged_{split}.pt in shard .pt di dimensione controllata, piu' un
index.pt che mappa ogni indice globale di sequenza a (shard_id, offset).

Formato di output compatibile con Component/PuzzleShardedDataset.py:

    shard_dir/
        index.pt          -> {"index": [(shard_id, offset), ...],
                               "n_sequences": int, "n_shards": int}
        shard_00000.pt    -> lista di sequenze (ogni sequenza e' una lista di ply,
                              stesso formato di merged_{split}.pt)
        shard_00001.pt
        ...

Nessun numero fisso di sequenze per shard: come in LMDBShards.py, ogni shard
viene chiuso quando la sua dimensione stimata su disco supera --target-shard-gb
(eccezione: una singola sequenza piu' grande del target viene comunque scritta
da sola, per non bloccare il build).

Build:

    python ShardDataset.py build \
        --data-dir Dataset/Train \
        --splits train val

Parametri opzionali:

    --target-shard-gb 0.5   (default)

Uso nel training (gia' presente in Component/PuzzleShardedDataset.py):

    from Component.PuzzleShardedDataset import PuzzleShardedDataset

    train_dataset = PuzzleShardedDataset(
        "Dataset/Train/shards_train", cache_size=3
    )
"""
from __future__ import annotations

import argparse
import io
import os
import time
from typing import Any, List, Tuple

import torch

DEFAULT_TARGET_SHARD_GB = 0.5


def _item_size_bytes(seq: Any) -> int:
    """Dimensione approssimata della singola sequenza serializzata da sola.

    E' una stima (torch.save di N sequenze insieme non e' esattamente la
    somma delle singole, per via di header/metadata condivisi), ma e' O(1)
    per sequenza invece di O(dimensione buffer accumulato): sommandola in
    modo incrementale evitiamo di riserializzare l'intero buffer ad ogni
    controllo, che sarebbe quadratico sull'intero file.
    """
    buf = io.BytesIO()
    torch.save(seq, buf)
    return buf.tell()


def build_shards(
    src_pt: str,
    out_dir: str,
    target_shard_gb: float = DEFAULT_TARGET_SHARD_GB,
    overwrite: bool = False,
) -> None:
    """Costruisce shard .pt + index.pt a partire da un merged_{split}.pt.

    Args:
        src_pt: percorso al file merged_{split}.pt sorgente.
        out_dir: cartella di output (verra' creata se assente).
        target_shard_gb: dimensione approssimativa desiderata per shard.
        overwrite: se True, rimuove shard/index preesistenti senza chiedere
            conferma. Se False e la cartella contiene gia' shard, chiede
            conferma interattiva (stesso comportamento di LMDBShards.py).
    """
    if not os.path.exists(src_pt):
        print(f"[SKIP] {src_pt} non trovato")
        return

    if target_shard_gb <= 0:
        raise ValueError("--target-shard-gb deve essere > 0")

    target_bytes = int(target_shard_gb * (1024 ** 3))

    print(f"Caricamento {src_pt}...")
    t0 = time.time()
    data_list: List[Any] = torch.load(src_pt, weights_only=False, map_location="cpu")
    total = len(data_list)
    print(f"Caricato in {time.time() - t0:.1f}s ({total} sequenze)")

    if total == 0:
        print("[SKIP] Dataset vuoto")
        return

    os.makedirs(out_dir, exist_ok=True)

    existing_shards = sorted(
        name
        for name in os.listdir(out_dir)
        if name.startswith("shard_") and name.endswith(".pt")
    )
    index_path = os.path.join(out_dir, "index.pt")

    if existing_shards or os.path.exists(index_path):
        print()
        print(f"Trovati {len(existing_shards)} shard esistenti in {out_dir}.")
        if not overwrite:
            answer = input("Vuoi eliminarli e ricostruire? [y/N]: ").strip().lower()
            if answer != "y":
                print("[STOP] Build annullato.")
                return

        for name in existing_shards:
            path = os.path.join(out_dir, name)
            print(f"  Rimozione {path}")
            os.remove(path)
        if os.path.exists(index_path):
            os.remove(index_path)

    print()
    print("Configurazione:")
    print(f"  Sequenze totali : {total}")
    print(f"  Target shard    : {target_shard_gb:.2f} GB")
    print()

    index: List[Tuple[int, int]] = []
    shard_id = 0
    shard_buffer: List[Any] = []
    shard_bytes = 0  # somma incrementale delle dimensioni stimate per-item
    global_idx = 0
    build_start = time.time()

    def flush_shard() -> None:
        nonlocal shard_id, shard_buffer, shard_bytes
        if not shard_buffer:
            return

        shard_path = os.path.join(out_dir, f"shard_{shard_id:05d}.pt")
        torch.save(shard_buffer, shard_path)

        elapsed = time.time() - build_start
        print(
            f"  \u2713 shard_{shard_id:05d}.pt completato | "
            f"sequenze={len(shard_buffer)} | "
            f"dati\u2248{shard_bytes / (1024 ** 3):.3f} GB | "
            f"tempo={elapsed:.1f}s"
        )

        shard_id += 1
        shard_buffer = []
        shard_bytes = 0

    while global_idx < total:
        seq = data_list[global_idx]
        item_size = _item_size_bytes(seq)

        # Chiudi lo shard PRIMA di inserire questa sequenza solo se il buffer
        # ha gia' contenuto e aggiungerla lo farebbe superare il target.
        # Se il buffer e' vuoto, la sequenza entra comunque (anche se da sola
        # supera il target: non possiamo spezzare una singola sequenza).
        # Questo permette a poche sequenze grandi di condividere legittimamente
        # uno shard (es. 3 sequenze = 780MB sotto un target di 1GB) invece di
        # essere isolate una per shard.
        if shard_buffer and shard_bytes + item_size > target_bytes:
            flush_shard()
            continue

        offset = len(shard_buffer)
        shard_buffer.append(seq)
        shard_bytes += item_size
        index.append((shard_id, offset))
        global_idx += 1

        if global_idx % 5000 == 0:
            print(
                f"    {global_idx}/{total} sequenze processate | "
                f"shard corrente: {len(shard_buffer)} seq, "
                f"\u2248{shard_bytes / (1024 ** 3):.3f} GB"
            )

    flush_shard()
    n_shards = shard_id

    torch.save(
        {"index": index, "n_sequences": total, "n_shards": n_shards},
        index_path,
    )

    elapsed = time.time() - build_start
    print()
    print("=" * 60)
    print("BUILD COMPLETATO")
    print("=" * 60)
    print(f"Dataset : {src_pt}")
    print(f"Output  : {out_dir}")
    print(f"Sequenze: {total}")
    print(f"Shard   : {n_shards}")
    print(f"Tempo   : {elapsed:.1f}s")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Costruisce shard .pt + index.pt da merged_{split}.pt."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Costruisce gli shard.")
    b.add_argument(
        "--data-dir",
        default="Dataset/Train",
        help="Directory contenente merged_train.pt / merged_val.pt",
    )
    b.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Split da convertire.",
    )
    b.add_argument(
        "--target-shard-gb",
        type=float,
        default=DEFAULT_TARGET_SHARD_GB,
        help=f"Dimensione target di ogni shard in GB. Default: {DEFAULT_TARGET_SHARD_GB}",
    )
    b.add_argument(
        "--overwrite",
        action="store_true",
        help="Sovrascrive shard esistenti senza chiedere conferma.",
    )

    args = ap.parse_args()

    if args.cmd == "build":
        for split in args.splits:
            src = os.path.join(args.data_dir, f"merged_{split}.pt")
            out = os.path.join(args.data_dir, f"shards_{split}")

            print()
            print("#" * 60)
            print(f"BUILD SPLIT: {split}")
            print("#" * 60)

            build_shards(
                src_pt=src,
                out_dir=out,
                target_shard_gb=args.target_shard_gb,
                overwrite=args.overwrite,
            )

        print()
        print("Completato.")


if __name__ == "__main__":
    main()
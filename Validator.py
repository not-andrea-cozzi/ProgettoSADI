import logging
import os
import random
import sys
from types import SimpleNamespace
from typing import Dict, List, Set

import pandas as pd
import torch
from tqdm import tqdm
import zstandard as zstd

from DatasetCreator.GraphBuilder import GraphBuilder
import chess
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_model_validator")


def load_config(config_path: str = "validator.yaml") -> SimpleNamespace:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File di configurazione {config_path} non trovato.")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)


CONFIG = load_config("Yaml/validator.yaml") 


def resolve_puzzle_csv(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Se path e' un .csv.zst lo decomprime accanto a se' (skip se gia' fatto),
    altrimenti ritorna path invariato (deve essere un .csv gia' pronto)."""
    if not path.endswith(".zst"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} non trovato.")
        return path

    out_csv = path[: -len(".zst")]
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        logger.info(f"{out_csv} gia' presente, decompressione saltata.")
        return out_csv

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} non trovato.")

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    tmp_out = out_csv + ".tmp"
    dctx = zstd.ZstdDecompressor()
    total_size = os.path.getsize(path)
    try:
        with open(path, "rb") as f_in, open(tmp_out, "wb") as f_out:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Decomprimo {os.path.basename(path)}") as pbar:
                reader = dctx.stream_reader(f_in)
                while True:
                    chunk = reader.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.n = f_in.tell()
                    pbar.refresh()
        os.replace(tmp_out, out_csv)
    except BaseException:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        raise
    return out_csv


def _extract_puzzle_ids_from_pt(path: str) -> Set[str]:
    """Estrae i PuzzleId gia' usati da un file .pt (lista di Data)."""
    ids: Set[str] = set()
    if not os.path.exists(path):
        logger.warning(f"{path} non trovato, nessun ID escluso da qui.")
        return ids
    data_list = torch.load(path, weights_only=False)
    for d in data_list:
        pid = getattr(d, "puzzle_id", None)
        if pid is not None:
            ids.add(str(pid))
        prob_id = getattr(d, "problem_id", None)
        if prob_id is not None:
            ids.add(str(prob_id))
    return ids


def collect_seen_puzzle_ids(dataset_dir: str, merged_subfolder: str, holdout_subfolder: str,
                             holdout_filename: str) -> Set[str]:
    """Raccoglie tutti i PuzzleId gia' usati in train/val/test + holdout esterno."""
    seen: Set[str] = set()

    merged_dir = os.path.join(dataset_dir, merged_subfolder)
    for split in ("train", "val", "test"):
        p = os.path.join(merged_dir, f"merged_{split}.pt")
        found = _extract_puzzle_ids_from_pt(p)
        logger.info(f"  merged_{split}.pt -> {len(found)} id esclusi")
        seen |= found

    holdout_path = os.path.join(dataset_dir, holdout_subfolder, holdout_filename)
    found = _extract_puzzle_ids_from_pt(holdout_path)
    logger.info(f"  {holdout_filename} -> {len(found)} id esclusi")
    seen |= found

    logger.info(f"Totale PuzzleId gia' visti da escludere: {len(seen)}")
    return seen


def build_new_puzzle_test_set(
    csv_paths: List[str],
    seen_ids: Set[str],
    mate_range: tuple,
    target_samples: int,
    avg_time_by_rating: Dict[int, float],
    seed: int = 42,
    chunksize: int = 50_000,
) -> List:
    lo, hi = mate_range
    theme_pattern = "|".join(f"mateIn{n}" for n in range(lo, hi + 1))
    already_excluded = set(seen_ids)

    candidate_rows = []
    for csv_path in csv_paths:
        if len(candidate_rows) >= target_samples * 3:
            break
        reader = pd.read_csv(csv_path, chunksize=chunksize)
        pbar = tqdm(desc=f"Scansione {os.path.basename(csv_path)} [solo mai visti]", unit=" righe")
        for chunk in reader:
            mask = chunk["Themes"].str.contains(theme_pattern, na=False)
            mask &= ~chunk["PuzzleId"].astype(str).isin(already_excluded)
            filtered = chunk[mask]
            candidate_rows.extend(filtered.to_dict("records"))
            already_excluded.update(filtered["PuzzleId"].astype(str).tolist())
            pbar.update(len(filtered))
            if len(candidate_rows) >= target_samples * 3:
                break
        pbar.close()
    logger.info(f"Righe candidate (mai viste, mate {lo}-{hi}, {len(csv_paths)} fonti): {len(candidate_rows)}")

    random.Random(seed).shuffle(candidate_rows)

    data_list = []
    for row in tqdm(candidate_rows, desc="Costruzione grafi test set"):
        if len(data_list) >= target_samples:
            break

        uci_moves = str(row["Moves"]).split()
        if not uci_moves:
            continue

        try:
            board = chess.Board(row["FEN"])
        except Exception:
            continue

        mate_n_iniziale = _extract_mate_n(str(row["Themes"]))
        if mate_n_iniziale == 0:
            continue

        # Clock simulato
        if avg_time_by_rating:
            rating = float(row["Rating"])
            bucket = round(rating / 100) * 100
            clock = avg_time_by_rating.get(bucket, 15.0)
        else:
            clock = 5.0 + (float(row["Rating"]) / 3000.0) * 55.0

        first_move_uci = uci_moves[0]
        try:
            first_move = chess.Move.from_uci(first_move_uci)
        except Exception:
            continue
        if first_move not in board.legal_moves:
            continue
        board.push(first_move)

        for ply_idx, uci in enumerate(uci_moves[1:], start=1):
            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                break

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
            try:
                d = GraphBuilder.board_to_pyg_data(
                    board,
                    clock_seconds=clock * (1 + 0.1 * ply_idx),
                    label=label,
                )
            except Exception:
                break

            d.puzzle_id = row["PuzzleId"]
            d.rating = float(row["Rating"])
            data_list.append(d)

            board.push(move)

            if len(data_list) >= target_samples:
                break

    return data_list


def _extract_mate_n(themes: str) -> int:
    for t in themes.split():
        if t.startswith("mateIn"):
            return int(t.replace("mateIn", ""))
    return 0


def main():
    cfg = CONFIG

    # Usa i campi del YAML (alcuni sono opzionali, con default)
    # Percorso del CSV dei puzzle (deve essere presente nel YAML)
    puzzle_csv_path = getattr(cfg, "puzzle_csv_path", None)
    if puzzle_csv_path is None:
        # Se non esplicitato, prova a usare 'csv_path' ma avvisa
        logger.warning("'puzzle_csv_path' non specificato, uso 'csv_path' (potrebbe essere sbagliato).")
        puzzle_csv_path = cfg.csv_path
    puzzle_csv_path = resolve_puzzle_csv(puzzle_csv_path)

    if not os.path.exists(puzzle_csv_path):
        raise FileNotFoundError(
            f"{puzzle_csv_path} non trovato. Assicurati che il file esista e che sia un CSV di puzzle Lichess."
        )

    # Percorso delle statistiche dei tempi (opzionale)
    time_stats_path = getattr(cfg, "time_stats_path", None)
    avg_time_by_rating = {}
    if time_stats_path and os.path.exists(time_stats_path):
        import json
        with open(time_stats_path) as f:
            avg_time_by_rating = {int(k): float(v) for k, v in json.load(f).items()}
    else:
        logger.warning(f"time_stats_path non trovato o non specificato: uso fallback lineare per il clock.")

    # Parametri dal YAML
    dataset_dir = getattr(cfg, "dataset_dir", "Dataset")
    merged_subfolder = getattr(cfg, "merged_subfolder", "Train")
    holdout_subfolder = getattr(cfg, "holdout_subfolder", "Holdout")
    holdout_filename = getattr(cfg, "holdout_filename", "external_holdout.pt")

    # Range mate: usa mate_range_min/max se presenti, altrimenti mate_min/max
    mate_min = getattr(cfg, "mate_min", getattr(cfg, "mate_range_min", 1))
    mate_max = getattr(cfg, "mate_max", getattr(cfg, "mate_range_max", 5))

    target_samples = getattr(cfg, "target_samples", 50000)
    seed = getattr(cfg, "seed", 42)
    chunksize = getattr(cfg, "chunksize", 50000)

    out_subfolder = getattr(cfg, "out_subfolder", "Validation")
    out_filename = getattr(cfg, "out_filename", "validator_test.pt")
    # Se il YAML ha 'out_pt' e i precedenti non sono specificati, usa quello
    if hasattr(cfg, "out_pt") and not hasattr(cfg, "out_subfolder") and not hasattr(cfg, "out_filename"):
        out_path = cfg.out_pt
    else:
        out_dir = os.path.join(dataset_dir, out_subfolder)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_filename)

    logger.info("Raccolta PuzzleId gia' usati (train/val/test/holdout)...")
    seen_ids = collect_seen_puzzle_ids(dataset_dir, merged_subfolder, holdout_subfolder, holdout_filename)

    logger.info(f"Costruzione nuovo test set: target={target_samples}, mate={mate_min}-{mate_max}")
    data_list = build_new_puzzle_test_set(
        csv_paths=[puzzle_csv_path],
        seen_ids=seen_ids,
        mate_range=(mate_min, mate_max),
        target_samples=target_samples,
        avg_time_by_rating=avg_time_by_rating,
        seed=seed,
        chunksize=chunksize,
    )

    tmp_path = out_path + ".tmp"
    torch.save(data_list, tmp_path)
    os.replace(tmp_path, out_path)

    logger.info(f"Salvato {out_path}: {len(data_list)} campioni (target {target_samples}).")
    if len(data_list) < target_samples:
        logger.warning(
            "Campioni ottenuti inferiori al target: il pool di puzzle mai visti "
            "nel range mate richiesto potrebbe essere esaurito. Considera aumentare mate_max."
        )


if __name__ == "__main__":
    main()
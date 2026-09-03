import logging
import os
import sys
from ModelUtils.Utils import (
    load_config_as_namespace,
    resolve_puzzle_csv,
    collect_seen_puzzle_ids,
    build_validator_dataset,
)
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("generate_dataset")

CONFIG_PATH = "Yaml/generate.yaml"


def main():
    cfg = load_config_as_namespace(CONFIG_PATH)

    puzzle_csv_path = resolve_puzzle_csv(cfg.puzzle_csv_path)

    # Statistiche dei tempi (opzionale)
    avg_time_by_rating = {}
    if hasattr(cfg, "time_stats_path") and cfg.time_stats_path and os.path.exists(cfg.time_stats_path):
        import json
        with open(cfg.time_stats_path) as f:
            avg_time_by_rating = {int(k): float(v) for k, v in json.load(f).items()}
    else:
        logger.warning("time_stats_path non trovato: uso fallback lineare per il clock.")

    # Parametri dal YAML
    dataset_dir = cfg.dataset_dir
    merged_subfolder = cfg.merged_subfolder
    holdout_subfolder = cfg.holdout_subfolder
    holdout_filename = cfg.holdout_filename

    mate_min = cfg.mate_min
    mate_max = cfg.mate_max
    target_samples = cfg.target_samples
    seed = getattr(cfg, "seed", 42)
    chunksize = getattr(cfg, "chunksize", 50000)

    out_path = cfg.out_pt  # percorso completo di output

    # 1. Raccolta ID già visti
    logger.info("Raccolta PuzzleId già usati (train/val/test/holdout)...")
    seen_ids = collect_seen_puzzle_ids(
        dataset_dir, merged_subfolder, holdout_subfolder, holdout_filename
    )

    # 2. Costruzione nuovo dataset
    logger.info(f"Costruzione nuovo test set: target={target_samples}, mate={mate_min}-{mate_max}")
    data_list = build_validator_dataset(
        csv_paths=[puzzle_csv_path],
        seen_ids=seen_ids,
        mate_range=(mate_min, mate_max),
        target_samples=target_samples,
        avg_time_by_rating=avg_time_by_rating,
        seed=seed,
        chunksize=chunksize,
    )

    # 3. Salvataggio
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
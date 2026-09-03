
import argparse
import logging
import os
import sys
from typing import Any, Dict

import yaml

from Component.ClubGamesTimedBuilder import ClubGamesTimedBuilder


def load_config(config_path: str) -> Dict[str, Any]:
    """Carica e restituisce il dizionario di configurazione dal file YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File di configurazione non trovato: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(log_file: str, log_level: str) -> None:
    """Inizializza il logger di sistema."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        filename=log_file,
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Esegue ClubGamesTimedBuilder caricando i parametri da un file YAML."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Yaml/club_games_timed.yaml",
        help="Percorso del file YAML di configurazione (default: club_games_timed.yaml)",
    )
    args = parser.parse_args()

    # Caricamento configurazione
    cfg = load_config(args.config)

    pipeline_cfg = cfg.get("pipeline", {})
    engine_cfg = cfg.get("engine", {})
    raw_cfg = cfg.get("raw_data", {})
    builder_cfg = cfg.get("club_games_timed", {})

    # Setup Logging
    log_file = pipeline_cfg.get("log_file", "club_games_timed.log")
    log_level = pipeline_cfg.get("log_level", "INFO")
    setup_logger(log_file, log_level)

    print(f"--> Caricata configurazione da: {args.config}")

    # Costruzione dei percorsi di output
    dataset_dir = pipeline_cfg.get("dataset_dir", "Dataset")
    holdout_subfolder = pipeline_cfg.get("holdout_subfolder", "Holdout")
    out_dir = os.path.join(dataset_dir, holdout_subfolder)
    os.makedirs(out_dir, exist_ok=True)

    output_filename = builder_cfg.get("output_filename", "club_games_timed_holdout.pt")
    jsonl_filename = builder_cfg.get("jsonl_filename", "club_games_timed_holdout.jsonl")

    out_pt = os.path.join(out_dir, output_filename)
    jsonl_out_path = os.path.join(out_dir, jsonl_filename) if jsonl_filename else None

    # Mappatura e istanziazione di ClubGamesTimedBuilder
    builder = ClubGamesTimedBuilder(
        csv_path=raw_cfg.get("external_csv", "DatasetCreator/RawData/club_games_data.csv.zip"),
        stockfish_path=engine_cfg.get("stockfish_path", "/usr/games/stockfish"),
        out_pt=out_pt,
        mate_range=(
            int(builder_cfg.get("mate_range_min", 1)),
            int(builder_cfg.get("mate_range_max", 10)),
        ),
        analysis_time=float(builder_cfg.get("time_limit_seconds", 0.2)),
        pgn_col=builder_cfg.get("pgn_col", "pgn"),
        skip_games=int(builder_cfg.get("skip_games", 10000)),
        max_games_to_scan=builder_cfg.get("max_games_to_scan"),
        default_move_seconds=float(builder_cfg.get("default_move_seconds", 15.0)),
        require_clock=bool(builder_cfg.get("require_clock", False)),
        min_ply=int(builder_cfg.get("min_ply", 0)),
        ply_sample_step=int(builder_cfg.get("ply_sample_step", 3)),
        max_positions_per_game=builder_cfg.get("max_positions_per_game"),
        threads=int(engine_cfg.get("threads", 1)),
        hash_mb=int(engine_cfg.get("hash_mb", 128)),
        checkpoint_every=int(builder_cfg.get("checkpoint_every", 50)),
        workers=builder_cfg.get("workers"),
        chunk_games=int(builder_cfg.get("chunk_games", 200)),
        max_piece_count=builder_cfg.get("max_piece_count"),
        min_material_for_mate_attempt=int(builder_cfg.get("min_material_for_mate_attempt", 4)),
        candidate_min_legal_moves=int(builder_cfg.get("candidate_min_legal_moves", 1)),
        candidate_max_legal_moves=builder_cfg.get("candidate_max_legal_moves"),
        skip_if_in_check=bool(builder_cfg.get("skip_if_in_check", False)),
        only_decisive_games=bool(builder_cfg.get("only_decisive_games", True)),
        skip_time_forfeit=bool(builder_cfg.get("skip_time_forfeit", True)),
        min_game_plies=int(builder_cfg.get("min_game_plies", 20)),
        pool_join_timeout=float(builder_cfg.get("pool_join_timeout", 15.0)),
        jsonl_out_path=jsonl_out_path,
    )


    # Esecuzione del processo
    data = builder.run()

    print(f"\n--> Processo terminato! Posizioni totali estorte e salvate: {len(data)}")


if __name__ == "__main__":
    main()
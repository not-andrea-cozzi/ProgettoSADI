#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

from DatasetPipeline.GamesBuilder import GamesBuilder, SourceSpec
from DatasetPipeline.PipelineState import PipelineState, file_ready, torch_pt_ready
from DatasetPipeline.RatingStats import compute_rating_stats, save_rating_stats

logger = logging.getLogger("games_only")


class PipelineConfigError(Exception):
    pass


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def require_file(path: str, hint: str = ""):
    if not os.path.exists(path):
        msg = f"File richiesto non trovato: {path}."
        if hint:
            msg += f" {hint}"
        raise PipelineConfigError(msg)


def require_executable(path: str, hint: str = ""):
    if not (os.path.exists(path) and os.access(path, os.X_OK)):
        msg = f"Eseguibile non trovato o permessi di esecuzione mancanti: {path}."
        if hint:
            msg += f" {hint}"
        raise PipelineConfigError(msg)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    require_file(config_path, "Specificare un file YAML valido.")
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except Exception as e:
            raise PipelineConfigError(f"Errore nel parsing del YAML: {e}")
    if not isinstance(cfg, dict):
        raise PipelineConfigError("Il YAML deve definire un dizionario.")
    return cfg


def load_avg_time_by_rating(path: str) -> Dict[str, float]:
    if not file_ready(path):
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def main():
    config_path = "Yaml/club_games_timed.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    cfg = load_yaml_config(config_path)

    # Sezioni
    pipe_cfg = cfg.get("pipeline", {})
    engine_cfg = cfg.get("engine", {})
    raw_cfg = cfg.get("raw_data", {})
    games_cfg = cfg.get("games_pipeline", {}) or cfg.get("club_games_timed", {})

    # Sezione splits (se non presente, default 80/10/10)
    split_cfg = cfg.get("splits", {})
    split_ratios = (
        split_cfg.get("train_ratio", 0.65),
        split_cfg.get("val_ratio", 0.15),
        split_cfg.get("test_ratio", 0.20),
    )

    # Setup logging
    log_level = pipe_cfg.get("log_level", "INFO")
    log_file = pipe_cfg.get("log_file")
    setup_logging(log_level, log_file)

    # Percorsi
    dataset_dir = pipe_cfg.get("dataset_dir", "Dataset")
    games_subfolder = pipe_cfg.get("games_subfolder", "Games")
    merged_subfolder = pipe_cfg.get("merged_subfolder", "Train")
    state_file = pipe_cfg.get("state_file", "pipeline_state.json")
    force = pipe_cfg.get("force_recompute", False)

    games_dir = os.path.join(dataset_dir, games_subfolder)
    merged_dir = os.path.join(dataset_dir, merged_subfolder)
    os.makedirs(games_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    state_path = os.path.join(dataset_dir, state_file)
    if force and os.path.exists(state_path):
        logger.warning("Flag FORCE attivo: reset dello stato.")
        os.remove(state_path)
    state = PipelineState(state_path)

    stockfish_path = engine_cfg.get("stockfish_path", "/usr/games/stockfish")
    require_executable(stockfish_path, "Stockfish non trovato.")

    # --- Costruzione delle sorgenti ---
    sources = []

    # Sorgente Lichess (PGN .zst)
    lichess_path = raw_cfg.get("games_zst")
    if lichess_path and os.path.exists(lichess_path):
        sources.append(SourceSpec(
            kind="lichess",
            path=lichess_path,
            skip_games=games_cfg.get("skip_games_part1", 0),
            max_games=games_cfg.get("max_games", 200000),
            tag="lichess",
        ))
    else:
        logger.info("Sorgente Lichess non configurata o assente.")

    # Sorgente FICS (PGN .bz2 o .pgn)
    fics_path = raw_cfg.get("fics_pgn") or raw_cfg.get("external_csv")
    if fics_path and os.path.exists(fics_path):
        sources.append(SourceSpec(
            kind="fics",
            path=fics_path,
            skip_games=games_cfg.get("fics_skip_games", 0),
            max_games=games_cfg.get("fics_max_games", 200000),
            tag="fics",
        ))
    else:
        logger.info("Sorgente FICS non configurata o assente.")

    # Sorgente Club (CSV)
    club_path = raw_cfg.get("club_csv")
    if club_path and os.path.exists(club_path):
        sources.append(SourceSpec(
            kind="club",
            path=club_path,
            pgn_col=games_cfg.get("club_pgn_col", "pgn"),
            skip_games=games_cfg.get("club_skip_games", 0),
            max_games=games_cfg.get("club_max_games", 200000),
            tag="club",
        ))
    else:
        logger.info("Sorgente Club non configurata o assente.")

    if not sources:
        raise PipelineConfigError("Nessuna sorgente valida trovata (Lichess, FICS o Club).")

    # --- Parametri per GamesBuilder ---
    avg_time_path = games_cfg.get("avg_time_by_rating_json")
    avg_time_by_rating = load_avg_time_by_rating(avg_time_path) if avg_time_path else {}

    output_base = games_cfg.get("output_filename", "games")
    if not output_base.endswith(".pt"):
        output_base += ".pt"
    output_base = os.path.join(games_dir, output_base)

    # Nomi finali dei tre split
    games_paths = {
        "train": f"{os.path.splitext(output_base)[0]}_train.pt",
        "val": f"{os.path.splitext(output_base)[0]}_val.pt",
        "test": f"{os.path.splitext(output_base)[0]}_test.pt",
    }

    # --- STEP 1: Generazione dei grafi da partite ---
    def step_games_pipeline():
        builder = GamesBuilder(
            sources=sources,
            stockfish_path=stockfish_path,
            output_pt=output_base,
            mate_range=(games_cfg.get("mate_range_min", 1), games_cfg.get("mate_range_max", 10)),
            search_depth=games_cfg.get("search_depth", 8),
            analysis_time=games_cfg.get("time_limit_seconds", 0.2),
            workers=games_cfg.get("workers"),
            threads=engine_cfg.get("threads", 1),
            hash_mb=engine_cfg.get("hash_mb", 128),
            multipv=1,
            seed=42,
            default_move_seconds=games_cfg.get("default_move_seconds", 15.0),
            avg_time_by_rating=avg_time_by_rating,
            require_clock=games_cfg.get("require_clock", False),
            min_ply=games_cfg.get("min_ply", 8),
            ply_sample_step=games_cfg.get("ply_sample_step", 3),
            split_ratios=split_ratios,
            min_game_plies=games_cfg.get("min_game_plies", 20),
            max_positions_per_game=games_cfg.get("max_positions_per_game", 20),
            candidate_min_legal_moves=games_cfg.get("candidate_min_legal_moves", 1),
            candidate_max_legal_moves=games_cfg.get("candidate_max_legal_moves"),
            skip_if_in_check=games_cfg.get("skip_if_in_check", False),
            max_piece_count=games_cfg.get("max_piece_count", 18),
            only_decisive_games=games_cfg.get("only_decisive_games", True),
            skip_time_forfeit=games_cfg.get("skip_time_forfeit", True),
            min_material_for_mate_attempt=games_cfg.get("min_material_for_mate_attempt", 4),
            drop_zero_clock=games_cfg.get("drop_zero_clock", True),
            min_rating=games_cfg.get("min_rating", 700),
            max_rating=games_cfg.get("max_rating"),
            min_material_diff_for_mate_attempt=games_cfg.get("min_material_diff_for_mate_attempt", 3),
            require_heavy_piece=games_cfg.get("require_heavy_piece", True),
            skip_forced_moves=games_cfg.get("skip_forced_moves", False),
            dedupe_positions=games_cfg.get("dedupe_positions", True),
            skip_trivial_endgame=games_cfg.get("skip_trivial_endgame", True),
            pool_join_timeout=games_cfg.get("pool_join_timeout", 20.0),
            syzygy_path=engine_cfg.get("syzygy_path"),
            checkpoint_every=games_cfg.get("checkpoint_every", 5000),
            config_error_cls=PipelineConfigError,
        )
        splits, paths = builder.run()
        return splits, paths

    # Se i file esistono già e non si forza, carica; altrimenti genera
    if all(torch_pt_ready(p) for p in games_paths.values()) and not force:
        logger.info("File games già presenti, caricamento in corso...")
        games_splits = {
            s: torch.load(games_paths[s], weights_only=False)
            for s in ("train", "val", "test")
        }
        logger.info(f"Caricati games: train={len(games_splits['train'])}, "
                    f"val={len(games_splits['val'])}, test={len(games_splits['test'])}")
    else:
        # Esecuzione dello step
        logger.info("[RUN] Generazione dei grafi da partite...")
        games_splits, _ = step_games_pipeline()
        state.mark_done("games_pipeline")
        logger.info("[DONE] Generazione completata.")

    # ========================================================================
    # STEP 2: MERGE + RATING STATS (sempre eseguito, anche se i file esistono)
    # ========================================================================
    merged_paths = {
        split: os.path.join(merged_dir, f"merged_{split}.pt")
        for split in ("train", "val", "test")
    }
    rating_stats_path = os.path.join(merged_dir, "rating_stats.json")

    logger.info("[RUN] Merge dei tre split e calcolo rating stats...")

    # (Qui non ci sono puzzle, ma la logica è identica a DatasetMain.py)
    # Se in futuro si volessero aggiungere puzzle, basterà caricare anche quelli
    puzzle_splits = {}  # per compatibilità, ma non usato

    for split in ("train", "val", "test"):
        combined = []
        # Nel nostro caso abbiamo solo games_splits
        if split in games_splits:
            combined.extend(games_splits[split])
        # Se ci fossero puzzle, li aggiungeremmo qui

        if not combined:
            logger.warning(f"Split {split} vuoto dopo il merge.")
            continue

        out_path = merged_paths[split]
        tmp_path = out_path + ".tmp"
        torch.save(combined, tmp_path)
        os.replace(tmp_path, out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        logger.info(f"Salvato {split}: {len(combined)} campioni in {out_path} ({size_mb:.2f} MB)")

    # Calcola rating_stats sul train
    train_data = torch.load(merged_paths["train"], weights_only=False) if file_ready(merged_paths["train"]) else []
    if not train_data:
        raise PipelineConfigError("Merge del train set vuoto: impossibile calcolare rating_stats.")

    stats = compute_rating_stats(train_data)
    save_rating_stats(stats, rating_stats_path)
    logger.info(
        f"Rating stats (train): mean={stats['mean']:.1f} std={stats['std']:.1f} "
        f"copertura={stats['coverage']*100:.1f}% ({stats['n_valid']}/{stats['n_total']})"
    )
    if stats["coverage"] < 0.5:
        logger.warning(
            "Copertura rating sotto il 50%%: la normalizzazione rating nel modello "
            "si affiderà spesso al fallback 'assente'."
        )

    state.mark_done("merge")
    logger.info("[DONE] Merge completato.")

    logger.info("Pipeline completata con successo.")
    logger.info(f"Dataset games finale disponibile in: {merged_dir}")


if __name__ == "__main__":
    try:
        main()
    except PipelineConfigError as e:
        logger.error(f"Errore di configurazione: {e}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Errore imprevisto: {e}")
        sys.exit(1)
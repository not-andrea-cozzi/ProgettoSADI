import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from Training_TestModel.timegnn.core import pipeline
import torch
import zstandard as zstd
from tqdm import tqdm
import yaml

from DatasetCreator.ChessAnalysisPipeline import ChessAnalysisPipeline
from DatasetCreator.ExternalHoldoutBuilder import ExternalHoldoutBuilder
from DatasetCreator.PuzzleGraphDataset import PuzzleGraphDataset, merge_and_split
from Component.TimeStatBuilder import TimeStatsBuilder, load_avg_time_by_rating
from DatasetCreator.PipelineState import PipelineState, retry, file_ready, torch_pt_ready

logger = logging.getLogger("main")


class PipelineConfigError(Exception):
    """Errore bloccante di configurazione o validazione parametri."""


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
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
    require_file(config_path, "Specificare un file YAML valido tramite --config.")
    if yaml is None:
        raise PipelineConfigError(
            "Modulo 'pyyaml' non installato. Esegui: pip install pyyaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except Exception as e:
            raise PipelineConfigError(f"Errore nel parsing del file YAML ({config_path}): {e}")
    if not isinstance(cfg, dict):
        raise PipelineConfigError("Il file di configurazione YAML deve definire un dizionario di primo livello.")
    return cfg


def validate_config(cfg: Dict[str, Any]):
    required_sections = [
        "pipeline", "engine", "raw_data", "time_stats",
        "games_pipeline", "puzzle_pipeline", "splits", "external_holdout"
    ]
    for section in required_sections:
        if section not in cfg:
            raise PipelineConfigError(f"Sezione mancante nel file YAML: '{section}'.")

    splits = cfg.get("splits", {})
    t_ratio = splits.get("train_ratio", 0.8)
    v_ratio = splits.get("val_ratio", 0.1)
    te_ratio = splits.get("test_ratio", 0.1)
    total = round(t_ratio + v_ratio + te_ratio, 5)
    if total != 1.0:
        raise PipelineConfigError(f"La somma degli split ratio deve essere 1.0 (attuale: {total}).")

    m_train = (cfg["games_pipeline"].get("mate_range_min", 1), cfg["games_pipeline"].get("mate_range_max", 5))
    if m_train[0] > m_train[1] or m_train[0] < 1:
        raise PipelineConfigError(f"Range mate non valido per train games: {m_train}")

    m_holdout = (cfg["external_holdout"].get("mate_range_min", 1), cfg["external_holdout"].get("mate_range_max", 10))
    if m_holdout[0] > m_holdout[1] or m_holdout[0] < 1:
        raise PipelineConfigError(f"Range mate non valido per external holdout: {m_holdout}")


@retry(max_attempts=3, base_delay=3.0, exceptions=(OSError, zstd.ZstdError))
def decompress_zst_csv(zst_path: str, out_csv: str, chunk_size: int = 1024 * 1024) -> str:
    require_file(zst_path, "Verifica il percorso del file compresso dei puzzle Lichess.")

    if file_ready(out_csv):
        logger.info(f"{out_csv} presente e valido, decompressione saltata.")
        return out_csv

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    tmp_out = out_csv + ".tmp"

    total_size = os.path.getsize(zst_path)
    dctx = zstd.ZstdDecompressor()

    try:
        with open(zst_path, "rb") as f_in, open(tmp_out, "wb") as f_out:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Decomprimo {os.path.basename(zst_path)}") as pbar:
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


def build_puzzle_pt(
    csv_path: str,
    root: str,
    mate_range: Tuple[int, int] = (1, 5),
    max_puzzles: Optional[int] = None,
    avg_time_by_rating: Optional[Dict[str, float]] = None,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Dict[str, List[Any]]:
    splits: Dict[str, List[Any]] = {}
    for split in ("train", "val", "test"):
        expected_path = os.path.join(root, "processed", f"puzzle_{split}.pt")
        if os.path.exists(expected_path) and not torch_pt_ready(expected_path):
            logger.warning(f"{expected_path} corrotto o incompleto, forzo la rigenerazione.")
            os.remove(expected_path)

        ds = PuzzleGraphDataset(
            csv_path,
            root,
            split=split,
            mate_range=mate_range,
            max_puzzles=max_puzzles,
            avg_time_by_rating=avg_time_by_rating,
            split_ratios=split_ratios,
        )
        splits[split] = list(ds)
        logger.info(f"Dataset puzzle [{split}]: {len(splits[split])} grafi caricati.")
    return splits


def run_step(state: PipelineState, step_name: str, is_ready_fn, do_fn):
    if state.is_done(step_name) and is_ready_fn():
        logger.info(f"[SKIP] Step '{step_name}' gia' completato e verificato.")
        return
    if state.is_done(step_name) and not is_ready_fn():
        logger.warning(f"[REDO] Step '{step_name}' marcato completato ma output mancante o non valido. Rieseguo.")

    logger.info(f"[RUN] Avvio step '{step_name}'...")
    try:
        do_fn()
    except PipelineConfigError:
        state.mark_failed(step_name, "Config Error")
        raise
    except Exception as e:
        state.mark_failed(step_name, str(e))
        raise
    state.mark_done(step_name)
    logger.info(f"[DONE] Step '{step_name}' terminato con successo.")


VALID_STEPS = [
    "time_stats", "games_pipeline",
    "decompress_puzzles", "build_puzzles", "merge_and_split", "external_holdout",
]

CONFIG_PATH = "config.yaml"


def main():
    cfg = load_yaml_config(CONFIG_PATH)
    validate_config(cfg)

    pipe_cfg = cfg["pipeline"]
    engine_cfg = cfg["engine"]
    raw_cfg = cfg["raw_data"]
    stats_cfg = cfg["time_stats"]
    games_cfg = cfg["games_pipeline"]
    puzzle_cfg = cfg["puzzle_pipeline"]
    split_cfg = cfg["splits"]
    holdout_cfg = cfg["external_holdout"]

    log_level = pipe_cfg.get("log_level", "INFO")
    log_file = pipe_cfg.get("log_file")
    setup_logging(log_level, log_file)

    step = pipe_cfg.get("step")
    if step is not None and step not in VALID_STEPS:
        raise PipelineConfigError(
            f"'pipeline.step' non valido: '{step}'. Valori ammessi: {VALID_STEPS} oppure null/assente per l'intera pipeline."
        )

    stockfish_path = engine_cfg["stockfish_path"]
    dataset_dir = pipe_cfg.get("dataset_dir", "Dataset")
    puzzles_dir = os.path.join(dataset_dir, pipe_cfg.get("puzzles_subfolder", "puzzles"))
    merged_dir = os.path.join(dataset_dir, pipe_cfg.get("merged_subfolder", "Train"))
    holdout_dir = os.path.join(dataset_dir, pipe_cfg.get("holdout_subfolder", "Holdout"))
    games_dir = os.path.join(dataset_dir, pipe_cfg.get("games_subfolder", "Games"))

    for d in (dataset_dir, puzzles_dir, merged_dir, holdout_dir, games_dir):
        os.makedirs(d, exist_ok=True)

    state_file = pipe_cfg.get("state_file", "pipeline_state.json")
    state_path = os.path.join(dataset_dir, state_file)
    force = pipe_cfg.get("force_recompute", False)
    if force and os.path.exists(state_path):
        logger.warning("Flag FORCE attivo: azzeramento stato precedente.")
        os.remove(state_path)
    state = PipelineState(state_path)

    time_stats_path = os.path.join(dataset_dir, stats_cfg.get("output_filename", "avg_time_by_rating.json"))
    puzzle_csv_path = os.path.join(dataset_dir, puzzle_cfg.get("decompressed_csv_filename", "lichess_puzzles.csv"))
    games_output_base = os.path.join(dataset_dir, games_cfg.get("output_base_filename", "games.pt"))
    holdout_path = os.path.join(holdout_dir, holdout_cfg.get("output_filename", "external_holdout.pt"))

    mate_train_range = (games_cfg.get("mate_range_min", 1), games_cfg.get("mate_range_max", 5))
    mate_holdout_range = (holdout_cfg.get("mate_range_min", 1), holdout_cfg.get("mate_range_max", 10))
    split_ratios = (split_cfg.get("train_ratio", 0.8), split_cfg.get("val_ratio", 0.1), split_cfg.get("test_ratio", 0.1))

    ctx: Dict[str, Any] = {}

    # Step 1: Statistiche tempo medio per rating
    def _step_time_stats():
        require_file(raw_cfg["games_zst"], "File PGN compresso necessario per il calcolo delle durate mosse.")
        builder = TimeStatsBuilder(
            zst_path=raw_cfg["games_zst"],
            max_games=stats_cfg.get("max_games", 50000),
            bucket_size=stats_cfg.get("bucket_size", 100),
        )
        stats = builder.build_and_save(time_stats_path)
        ctx["avg_time_by_rating"] = stats

    if step is None or step == "time_stats":
        run_step(state, "time_stats", is_ready_fn=lambda: file_ready(time_stats_path), do_fn=_step_time_stats)

    if file_ready(time_stats_path) and "avg_time_by_rating" not in ctx:
        ctx["avg_time_by_rating"] = load_avg_time_by_rating(time_stats_path)

    # Step 2: Pipeline partite Lichess -> grafi posizioni matto 1-5
    
    games_paths = {
        "train": f"{os.path.splitext(games_output_base)[0]}_train.pt",
        "val": f"{os.path.splitext(games_output_base)[0]}_val.pt",
        "test": f"{os.path.splitext(games_output_base)[0]}_test.pt",
    }
    """
    def _step_games_pipeline():
        require_file(raw_cfg["games_zst"], "Archivio PGN mancante per l'analisi partite.")
        require_executable(stockfish_path, "Eseguibile Stockfish mancante o non avviabile.")
        pipeline = ChessAnalysisPipeline(
            zst_path=raw_cfg["games_zst"],
            stockfish_path=stockfish_path,
            output_pt=games_output_base,

            mate_range=mate_train_range,

            search_depth=games_cfg.get("search_depth", 6),
            analysis_time=games_cfg.get("time_limit_seconds", 0.15),

            max_games=games_cfg.get("max_games", 180000),
            skip_games=games_cfg.get("skip_games", 0),

            workers=games_cfg.get("workers", 8),
            threads=1,

            hash_mb=engine_cfg.get("hash_mb", 128),
            multipv=1,

            split_ratios=split_ratios,

            default_move_seconds=15.0,
            require_clock=False,
            min_ply=16,

            max_piece_count=games_cfg.get("max_piece_count", 18),
            only_decisive_games=games_cfg.get("only_decisive_games", True),
            skip_time_forfeit=games_cfg.get("skip_time_forfeit", True),
            ply_sample_step=games_cfg.get("ply_sample_step", 3),
            max_positions_per_game=games_cfg.get("max_positions_per_game", 20),

            syzygy_path=engine_cfg.get("syzygy_path"),
            min_material_for_mate_attempt=games_cfg.get("min_material_for_mate_attempt", 4),
        )
        splits, paths = pipeline.run()
        ctx["games_splits"] = splits
        ctx["games_paths"] = paths

    if step is None or step == "games_pipeline":
        run_step(
            state,
            "games_pipeline",
            is_ready_fn=lambda: all(torch_pt_ready(p) for p in games_paths.values()),
            do_fn=_step_games_pipeline,
        )
    """

    # Step 3: Decompressione puzzle Lichess
    def _step_decompress_puzzles():
        decompress_zst_csv(
            zst_path=raw_cfg["puzzles_zst"],
            out_csv=puzzle_csv_path,
            chunk_size=puzzle_cfg.get("chunk_size_bytes", 1024 * 1024),
        )

    if step is None or step == "decompress_puzzles":
        run_step(
            state,
            "decompress_puzzles",
            is_ready_fn=lambda: file_ready(puzzle_csv_path),
            do_fn=_step_decompress_puzzles,
        )

    # Step 4: Costruzione grafi puzzle con tempo sintetico e split
    puzzle_processed_paths = {
        split: os.path.join(puzzles_dir, "processed", f"puzzle_{split}.pt")
        for split in ("train", "val", "test")
    }

    def _step_build_puzzles():
        if "avg_time_by_rating" not in ctx:
            ctx["avg_time_by_rating"] = load_avg_time_by_rating(time_stats_path)
        splits = build_puzzle_pt(
            csv_path=puzzle_csv_path,
            root=puzzles_dir,
            mate_range=mate_train_range,
            max_puzzles=puzzle_cfg.get("max_puzzles", 100000),
            avg_time_by_rating=ctx["avg_time_by_rating"],
            split_ratios=split_ratios,
        )
        ctx["puzzle_splits"] = splits

    if step is None or step == "build_puzzles":
        run_step(
            state,
            "build_puzzles",
            is_ready_fn=lambda: all(torch_pt_ready(p) for p in puzzle_processed_paths.values()),
            do_fn=_step_build_puzzles,
        )

    # Step 5: Merge puzzle + games per ogni split (80/10/10)
    merged_paths = {
        split: os.path.join(merged_dir, f"merged_{split}.pt")
        for split in ("train", "val", "test")
    }

    def _step_merge():
        if "puzzle_splits" not in ctx:
            ctx["puzzle_splits"] = {
                s: torch.load(puzzle_processed_paths[s], weights_only=False)
                for s in ("train", "val", "test")
            }
        if "games_splits" not in ctx:
            ctx["games_splits"] = {
                s: torch.load(games_paths[s], weights_only=False)
                for s in ("train", "val", "test")
            }
        merge_and_split(
            puzzle_splits=ctx["puzzle_splits"],
            games_splits=ctx["games_splits"],
            out_dir=merged_dir,
        )

    if step is None or step == "merge_and_split":
        run_step(
            state,
            "merge_and_split",
            is_ready_fn=lambda: all(torch_pt_ready(p) for p in merged_paths.values()),
            do_fn=_step_merge,
        )

    # Step 6: Held-out esterno stratificato (n=1..10 per valutazione comparativa e profondita)
    def _step_holdout():
        builder = ExternalHoldoutBuilder(
            external_csv=raw_cfg["external_csv"],
            stockfish_path=stockfish_path,
            out_pt=holdout_path,
            mate_range=mate_holdout_range,
            time_limit=holdout_cfg.get("time_limit_seconds", 0.3),
            pgn_col=holdout_cfg.get("pgn_col", "pgn"),
            max_games_to_scan=holdout_cfg.get("max_games_to_scan", 1200),
            target_total_problems=holdout_cfg.get("target_total_problems", 200),
            stratification_config=holdout_cfg.get("stratification"),
            threads=engine_cfg.get("threads", 2),
            hash_mb=engine_cfg.get("hash_mb", 256),
            require_move_match=holdout_cfg.get("require_move_match", True),
            checkpoint_every=holdout_cfg.get("checkpoint_every", 50),
            workers=holdout_cfg.get("workers"),
            chunk_games=holdout_cfg.get("chunk_games", 200),
            config_error_cls=PipelineConfigError,
            min_ply=holdout_cfg.get("min_ply", 0),
            ply_sample_step=holdout_cfg.get("ply_sample_step", 3),
            max_piece_count=holdout_cfg.get("max_piece_count", 18),
            candidate_min_legal_moves=holdout_cfg.get("candidate_min_legal_moves", 1),
            candidate_max_legal_moves=holdout_cfg.get("candidate_max_legal_moves"),
            skip_if_in_check=holdout_cfg.get("skip_if_in_check", False),
            pool_join_timeout=holdout_cfg.get("pool_join_timeout", 15.0),
        )
        builder.run()

    if step is None or step == "external_holdout":
        run_step(
            state,
            "external_holdout",
            is_ready_fn=lambda: torch_pt_ready(holdout_path),
            do_fn=_step_holdout,
        )

    logger.info("Pipeline completata con successo.")
    logger.info(f"Dataset train/val/test pronti in: {merged_dir}")
    logger.info(f"Held-out set di validazione esterna pronto in: {holdout_dir}")


if __name__ == "__main__":
    try:
        main()
    except PipelineConfigError as e:
        logger.error(f"Errore di configurazione pipeline: {e}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Interruzione imprevista della pipeline: {e}")
        logger.error("Rilanciare lo script per riprendere dall'ultimo checkpoint valido.")
        sys.exit(1)
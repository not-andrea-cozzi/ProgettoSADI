import argparse
import io
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import pandas as pd
import torch
import zstandard as zstd
from tqdm import tqdm
import yaml

from DatasetCreator.GraphBuilder import GraphBuilder
from ChessAnalysisPipeline import ChessAnalysisPipeline
from PuzzleGraphDataset import PuzzleGraphDataset, merge_and_split
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


@retry(max_attempts=3, base_delay=5.0, exceptions=(chess.engine.EngineError, OSError, BrokenPipeError))
def open_engine(stockfish_path: str, threads: int = 2, hash_mb: int = 256) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": threads, "Hash": hash_mb})
    return engine


def build_external_holdout(
    external_csv: str,
    stockfish_path: str,
    out_pt: str,
    mate_range: Tuple[int, int] = (1, 10),
    time_limit: float = 0.3,
    pgn_col: str = "pgn",
    max_games_to_scan: Optional[int] = None,
    target_total_problems: int = 200,
    stratification_config: Optional[Dict[str, int]] = None,
    threads: int = 2,
    hash_mb: int = 256,
    require_move_match: bool = True,
    checkpoint_every: int = 50,
) -> List[Any]:
    require_file(external_csv, "Verificare il file CSV esterno per l'held-out.")
    require_executable(stockfish_path, "Verificare il percorso del binario Stockfish.")

    df = pd.read_csv(external_csv)
    if max_games_to_scan:
        df = df.head(max_games_to_scan)
    if pgn_col not in df.columns:
        raise PipelineConfigError(
            f"Colonna '{pgn_col}' assente in {external_csv}. Colonne disponibili: {list(df.columns)}"
        )

    lo, hi = mate_range
    strat_targets: Dict[int, int] = {}
    cfg_strat = stratification_config or {}
    t_1_5 = cfg_strat.get("n_1_to_5_target_each", 30)
    t_6_10 = cfg_strat.get("n_6_to_10_target_each", 10)

    for n in range(lo, hi + 1):
        strat_targets[n] = t_1_5 if n <= 5 else t_6_10

    counts_by_n: Dict[int, int] = defaultdict(int)
    test_data: List[Any] = []
    skipped_no_match = 0
    tmp_out = out_pt + ".tmp"

    def save_checkpoint():
        os.makedirs(os.path.dirname(os.path.abspath(out_pt)) or ".", exist_ok=True)
        torch.save(test_data, tmp_out)
        os.replace(tmp_out, out_pt)

    def targets_satisfied() -> bool:
        if len(test_data) >= target_total_problems:
            return True
        return all(counts_by_n[n] >= strat_targets[n] for n in range(lo, hi + 1))

    engine = open_engine(stockfish_path, threads=threads, hash_mb=hash_mb)

    try:
        for game_idx, pgn_text in enumerate(tqdm(df[pgn_col].dropna(), desc="Estrazione Held-Out")):
            if targets_satisfied():
                break

            try:
                game = chess.pgn.read_game(io.StringIO(pgn_text))
            except Exception as e:
                logger.debug(f"Partita {game_idx}: PGN illeggibile ({e}), salto.")
                continue

            if game is None:
                continue

            node = game
            while node.variations:
                if targets_satisfied():
                    break

                nxt = node.variation(0)
                board = node.board()

                try:
                    info = engine.analyse(board, chess.engine.Limit(time=time_limit, mate=hi), multipv=1)
                except chess.engine.EngineTerminatedError:
                    logger.warning(f"Stockfish terminato alla partita {game_idx}, riavvio del processo motore.")
                    try:
                        engine.quit()
                    except Exception:
                        pass
                    engine = open_engine(stockfish_path, threads=threads, hash_mb=hash_mb)
                    node = nxt
                    continue
                except Exception as e:
                    logger.debug(f"Partita {game_idx}, semimossa {node.ply()}: fallimento analisi ({e}).")
                    node = nxt
                    continue

                score = info[0].get("score") if info else None
                if score and score.relative.is_mate():
                    mate_n = score.relative.mate()
                    if mate_n is not None and lo <= mate_n <= hi:
                        if counts_by_n[mate_n] >= strat_targets.get(mate_n, 9999):
                            node = nxt
                            continue

                        pv = info[0].get("pv")
                        engine_best_move = pv[0] if pv else None
                        if not engine_best_move:
                            node = nxt
                            continue

                        if require_move_match and nxt.move != engine_best_move:
                            skipped_no_match += 1
                            node = nxt
                            continue

                        legal = list(board.legal_moves)
                        try:
                            best_idx = legal.index(engine_best_move)
                        except ValueError:
                            node = nxt
                            continue

                        label = {"mate_n": mate_n, "best_move_idx": best_idx}
                        data_item = GraphBuilder.board_to_pyg_data(board, clock_seconds=0.0, label=label)
                        data_item.problem_id = f"chesscom_{game_idx}_{node.ply()}"
                        data_item.mate_n = mate_n

                        test_data.append(data_item)
                        counts_by_n[mate_n] += 1

                        if len(test_data) % checkpoint_every == 0:
                            save_checkpoint()

                node = nxt

    finally:
        try:
            engine.quit()
        except Exception:
            pass

    save_checkpoint()
    dist_str = ", ".join(f"n={n}: {counts_by_n[n]}/{strat_targets[n]}" for n in sorted(strat_targets.keys()))
    logger.info(f"Held-out completato: {len(test_data)} problemi salvati in {out_pt} (Distribuzione: {dist_str}).")
    logger.info(f"Mosse discordanti scartate rispetto al motore: {skipped_no_match}")
    return test_data


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline di preparazione dataset per Timed Graph Neural Network su scacchi."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Percorso del file di configurazione YAML (default: config.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forza la riesecuzione azzerando lo stato della pipeline.",
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        choices=["time_stats", "games_pipeline", "decompress_puzzles", "build_puzzles", "merge_and_split", "external_holdout"],
        help="Esegue solo uno step specifico della pipeline anziche' l'intero flusso.",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=None,
        help="Override del percorso dell'eseguibile Stockfish.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override del livello di logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    validate_config(cfg)

    pipe_cfg = cfg["pipeline"]
    engine_cfg = cfg["engine"]
    raw_cfg = cfg["raw_data"]
    stats_cfg = cfg["time_stats"]
    games_cfg = cfg["games_pipeline"]
    puzzle_cfg = cfg["puzzle_pipeline"]
    split_cfg = cfg["splits"]
    holdout_cfg = cfg["external_holdout"]

    log_level = args.log_level or pipe_cfg.get("log_level", "INFO")
    log_file = pipe_cfg.get("log_file")
    setup_logging(log_level, log_file)

    stockfish_path = args.stockfish_path or engine_cfg["stockfish_path"]
    dataset_dir = pipe_cfg.get("dataset_dir", "Dataset")
    puzzles_dir = os.path.join(dataset_dir, pipe_cfg.get("puzzles_subfolder", "puzzles"))
    merged_dir = os.path.join(dataset_dir, pipe_cfg.get("merged_subfolder", "merged"))

    for d in (dataset_dir, puzzles_dir, merged_dir):
        os.makedirs(d, exist_ok=True)

    state_file = pipe_cfg.get("state_file", "pipeline_state.json")
    state_path = os.path.join(dataset_dir, state_file)
    force = args.force or pipe_cfg.get("force_recompute", False)
    if force and os.path.exists(state_path):
        logger.warning("Flag FORCE attivo: azzeramento stato precedente.")
        os.remove(state_path)
    state = PipelineState(state_path)

    time_stats_path = os.path.join(dataset_dir, stats_cfg.get("output_filename", "avg_time_by_rating.json"))
    puzzle_csv_path = os.path.join(dataset_dir, puzzle_cfg.get("decompressed_csv_filename", "lichess_puzzles.csv"))
    games_output_base = os.path.join(dataset_dir, games_cfg.get("output_base_filename", "games.pt"))
    holdout_path = os.path.join(merged_dir, holdout_cfg.get("output_filename", "external_holdout.pt"))

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

    if args.step is None or args.step == "time_stats":
        run_step(state, "time_stats", is_ready_fn=lambda: file_ready(time_stats_path), do_fn=_step_time_stats)

    if file_ready(time_stats_path) and "avg_time_by_rating" not in ctx:
        ctx["avg_time_by_rating"] = load_avg_time_by_rating(time_stats_path)

    # Step 2: Pipeline partite Lichess -> grafi posizioni matto 1-5
    games_paths = {
        "train": f"{os.path.splitext(games_output_base)[0]}_train.pt",
        "val": f"{os.path.splitext(games_output_base)[0]}_val.pt",
        "test": f"{os.path.splitext(games_output_base)[0]}_test.pt",
    }

    def _step_games_pipeline():
        require_file(raw_cfg["games_zst"], "Archivio PGN mancante per l'analisi partite.")
        require_executable(stockfish_path, "Eseguibile Stockfish mancante o non avviabile.")
        pipeline = ChessAnalysisPipeline(
            zst_path=raw_cfg["games_zst"],
            stockfish_path=stockfish_path,
            output_pt=games_output_base,
            mate_range=mate_train_range,
            max_games=games_cfg.get("max_games", 180000),
            time_limit=games_cfg.get("time_limit_seconds", 0.2),
            threads=engine_cfg.get("threads", 2),
            hash_mb=engine_cfg.get("hash_mb", 256),
            split_ratios=split_ratios,
        )
        splits, paths = pipeline.run()
        ctx["games_splits"] = splits
        ctx["games_paths"] = paths

    if args.step is None or args.step == "games_pipeline":
        run_step(
            state,
            "games_pipeline",
            is_ready_fn=lambda: all(torch_pt_ready(p) for p in games_paths.values()),
            do_fn=_step_games_pipeline,
        )

    # Step 3: Decompressione puzzle Lichess
    def _step_decompress_puzzles():
        decompress_zst_csv(
            zst_path=raw_cfg["puzzles_zst"],
            out_csv=puzzle_csv_path,
            chunk_size=puzzle_cfg.get("chunk_size_bytes", 1024 * 1024),
        )

    if args.step is None or args.step == "decompress_puzzles":
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

    if args.step is None or args.step == "build_puzzles":
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

    if args.step is None or args.step == "merge_and_split":
        run_step(
            state,
            "merge_and_split",
            is_ready_fn=lambda: all(torch_pt_ready(p) for p in merged_paths.values()),
            do_fn=_step_merge,
        )

    # Step 6: Held-out esterno stratificato (n=1..10 per valutazione comparativa e profondita)
    def _step_holdout():
        build_external_holdout(
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
        )

    if args.step is None or args.step == "external_holdout":
        run_step(
            state,
            "external_holdout",
            is_ready_fn=lambda: torch_pt_ready(holdout_path),
            do_fn=_step_holdout,
        )

    logger.info("Pipeline completata con successo.")
    logger.info(f"Dataset train/val/test pronti in: {merged_dir}")
    logger.info(f"Held-out set di validazione esterna pronto in: {holdout_path}")


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
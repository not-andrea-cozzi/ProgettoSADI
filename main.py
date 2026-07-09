import io
import os
import sys
import logging
import argparse

import chess
import chess.pgn
import chess.engine
import pandas as pd
import torch
import zstandard as zstd
from tqdm import tqdm

from Model.graph_builder import GraphBuilder
from Component.ChessAnalysisPipeline import ChessAnalysisPipeline
from Component.PuzzleGraphDataset import PuzzleGraphDataset, merge_and_split
from Component.TimeStatBuilder import TimeStatsBuilder, load_avg_time_by_rating
from PipelineState import PipelineState, retry, file_ready, torch_pt_ready

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


class PipelineConfigError(Exception):
    """Errore di configurazione non recuperabile (path mancante, dipendenza assente):
    non ha senso ritentare, va segnalato subito e chiaramente."""


def require_file(path: str, hint: str = ""):
    if not os.path.exists(path):
        msg = f"File richiesto non trovato: {path}."
        if hint:
            msg += f" {hint}"
        raise PipelineConfigError(msg)


def require_executable(path: str, hint: str = ""):
    if not (os.path.exists(path) and os.access(path, os.X_OK)):
        msg = f"Eseguibile non trovato o non eseguibile: {path}."
        if hint:
            msg += f" {hint}"
        raise PipelineConfigError(msg)


@retry(max_attempts=3, base_delay=3.0, exceptions=(OSError, zstd.ZstdError))
def decompress_zst_csv(zst_path: str, out_csv: str, chunk_size: int = 1024 * 1024) -> str:
    """Decomprime un .csv.zst in un .csv su disco, con progress bar (basata sui byte
    compressi letti, non sulla dimensione finale che zstd non conosce a priori).
    Scrive su file temporaneo e fa os.replace finale: se il processo muore a meta',
    non rimane un .csv troncato che verrebbe scambiato per "gia' pronto" al resume."""
    require_file(zst_path, "Controlla il path dei dati grezzi (rawData/).")

    if file_ready(out_csv):
        logger.info(f"{out_csv} gia' presente, salto decompressione.")
        return out_csv

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    tmp_out = out_csv + ".tmp"

    total_size = os.path.getsize(zst_path)
    dctx = zstd.ZstdDecompressor()

    try:
        with open(zst_path, "rb") as f_in, open(tmp_out, "wb") as f_out:
            with tqdm(total=total_size, unit="B", unit_scale=True,
                      desc=f"Decomprimo {os.path.basename(zst_path)}") as pbar:
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


def build_puzzle_pt(csv_path: str, root: str, mate_range=(1, 5), max_puzzles=None,
                     avg_time_by_rating=None):
    """Genera i tre split puzzle (train/val/test) come liste di Data PyG.
    Lo split e' fatto per PuzzleId DENTRO PuzzleGraphDataset: ogni split="train"/
    "val"/"test" processa lo STESSO pool di righe CSV filtrate e ne prende una
    partizione disgiunta e deterministica (stesso seed), quindi le tre chiamate
    qui sotto non si sovrappongono mai.

    PuzzleGraphDataset e' un InMemoryDataset: se processed_paths[0] esiste gia' sul
    disco, PyG stesso salta process() internamente. Questo e' gia' un resume "gratuito"
    per split; qui aggiungiamo solo la validazione esplicita del file (torch_pt_ready)
    cosi' un file .pt troncato da un crash precedente viene rigenerato invece di far
    esplodere torch.load piu' avanti nel training."""
    splits = {}
    for split in ("train", "val", "test"):
        expected_path = os.path.join(root, "processed", f"puzzle_{split}.pt")
        if os.path.exists(expected_path) and not torch_pt_ready(expected_path):
            logger.warning(f"{expected_path} corrotto/troncato, lo rimuovo per forzare rigenerazione.")
            os.remove(expected_path)

        ds = PuzzleGraphDataset(csv_path, root, split=split, mate_range=mate_range,
                                 max_puzzles=max_puzzles, avg_time_by_rating=avg_time_by_rating)
        splits[split] = list(ds)
        logger.info(f"Puzzle split '{split}': {len(splits[split])} posizioni-grafo.")
    return splits


@retry(max_attempts=3, base_delay=5.0, exceptions=(chess.engine.EngineError, OSError, BrokenPipeError))
def _open_engine(stockfish_path: str) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 64})
    return engine


def build_external_holdout(
    external_csv: str,
    stockfish_path: str,
    out_pt: str,
    mate_range=(1, 10),
    time_limit: float = 0.2,
    pgn_col: str = "pgn",
    max_games=None,
    max_problems=None,
    require_move_match: bool = True,
    checkpoint_every: int = 200,
):
    """Held-out ESTERNO da dataset chess.com (60k games, colonna `pgn` con partita completa).
    Non ha FEN/Moves/MateIn pronti: scandisce ogni partita mossa per mossa, usa Stockfish
    per trovare posizioni di matto forzato in mate_range mosse (stessa logica di
    ChessAnalysisPipeline, qui sequenziale dato il volume ridotto per l'eval finale).
    MAI mischiato con games/puzzle Lichess usati in train/val.

    mate_range default ora (1, 10): la spec chiede esplicitamente "n ranging from 1 to
    10 (to test depth limits)" per l'held-out finale, a differenza del training
    (mate_range (1,5) per games/puzzle) dove n>5 e' raro e non richiesto.
    NB: mate=10 in chess.engine.Limit fa cercare Stockfish molto piu' a fondo di
    mate=5 => scansione piu' lenta per posizione. time_limit resta un tetto per
    mossa, non una garanzia di trovare matti profondi: con time_limit molto basso
    e mate_range esteso a 10, alcuni mate-in-8/9/10 potrebbero non essere trovati
    in tempo e la posizione verra' semplicemente scartata (comportamento gia'
    presente prima, solo piu' frequente ora che il range e' piu' ampio).

    require_move_match: se True, una posizione entra nell'held-out solo se la mossa
    EFFETTIVAMENTE GIOCATA coincide con la prima mossa del matto forzato trovato da
    Stockfish. Altrimenti la label userebbe la mossa storica anche quando il
    giocatore non ha eseguito il matto individuato, producendo esempi mal etichettati.

    Robustezza aggiunta:
    - salvataggio incrementale ogni `checkpoint_every` problemi trovati (torch.save
      atomico su file temporaneo + os.replace), cosi' un crash a meta' scansione
      (es. Stockfish che muore) non fa perdere ore di lavoro gia' fatto;
    - se il motore Stockfish crasha durante la scansione, viene riavviato (con retry)
      e la scansione riprende dalla partita successiva invece di abortire l'intero step."""
    require_file(external_csv, "Controlla il path del dataset esterno chess.com (rawData/).")
    require_executable(stockfish_path, "Verifica installazione/path di Stockfish.")

    df = pd.read_csv(external_csv)
    if max_games:
        df = df.head(max_games)
    if pgn_col not in df.columns:
        raise PipelineConfigError(
            f"Colonna '{pgn_col}' assente in {external_csv}. Colonne trovate: {list(df.columns)}"
        )

    lo, hi = mate_range
    test_data = []
    skipped_no_match = 0
    tmp_out = out_pt + ".tmp"

    def _checkpoint_save():
        os.makedirs(os.path.dirname(out_pt) or ".", exist_ok=True)
        torch.save(test_data, tmp_out)
        os.replace(tmp_out, out_pt)  # atomico: out_pt e' sempre valido o non esiste

    engine = _open_engine(stockfish_path)

    try:
        for game_idx, pgn_text in enumerate(tqdm(df[pgn_col].dropna(), desc="Held-out chess.com")):
            try:
                game = chess.pgn.read_game(io.StringIO(pgn_text))
            except Exception as e:
                logger.warning(f"Partita {game_idx}: PGN illeggibile ({e}), la salto.")
                continue
            if game is None:
                continue

            node = game
            while node.variations:
                nxt = node.variation(0)
                board = node.board()

                try:
                    info = engine.analyse(board, chess.engine.Limit(time=time_limit, mate=hi), multipv=1)
                except chess.engine.EngineTerminatedError:
                    logger.warning(f"Partita {game_idx}: motore Stockfish terminato inaspettatamente, riavvio.")
                    try:
                        engine.quit()
                    except Exception:
                        pass
                    engine = _open_engine(stockfish_path)
                    node = nxt
                    continue
                except Exception as e:
                    logger.debug(f"Partita {game_idx}, ply {node.ply()}: analisi fallita ({e}), salto mossa.")
                    node = nxt
                    continue

                score = info[0].get("score") if info else None
                if score and score.relative.is_mate():
                    mate_n = score.relative.mate()
                    if lo <= abs(mate_n) <= hi:
                        pv = info[0].get("pv")
                        engine_best_move = pv[0] if pv else None

                        if require_move_match and nxt.move != engine_best_move:
                            skipped_no_match += 1
                            node = nxt
                            continue

                        legal = list(board.legal_moves)
                        try:
                            best_idx = legal.index(nxt.move)
                        except ValueError:
                            node = nxt
                            continue

                        label = {"mate_n": mate_n, "best_move_idx": best_idx}
                        d = GraphBuilder.board_to_pyg_data(board, clock_seconds=0.0, label=label)
                        d.problem_id = f"chesscom_{game_idx}_{node.ply()}"
                        test_data.append(d)

                        if len(test_data) % checkpoint_every == 0:
                            _checkpoint_save()

                node = nxt

            if max_problems and len(test_data) >= max_problems:
                break
    finally:
        try:
            engine.quit()
        except Exception:
            pass

    _checkpoint_save()
    logger.info(
        f"held-out esterno: {len(test_data)} problemi -> {out_pt} "
        f"(scartati per mossa non giocata dal match Stockfish: {skipped_no_match})"
    )
    return test_data


def run_step(state: PipelineState, step_name: str, is_ready_fn, do_fn):
    """Wrapper unico di resume per ogni step del main:
    1. se lo state dice 'done' E i file di output superano il check di validita' -> skip
    2. altrimenti esegue do_fn(), e in caso di eccezione marca 'failed' con l'errore
       (senza mascherarlo: lo step fallito interrompe comunque main() con traceback,
       ma lo stato resta consultabile per il prossimo run)
    3. in caso di successo marca 'done'."""
    if state.is_done(step_name) and is_ready_fn():
        logger.info(f"[SKIP] '{step_name}' gia' completato e file validi, non rieseguo.")
        return
    if state.is_done(step_name) and not is_ready_fn():
        logger.warning(f"[REDO] '{step_name}' segnato come completato ma i file di output "
                        f"non sono validi/presenti: rieseguo.")

    logger.info(f"[RUN] '{step_name}'...")
    try:
        do_fn()
    except PipelineConfigError:
        state.mark_failed(step_name, "config error")
        raise
    except Exception as e:
        state.mark_failed(step_name, str(e))
        raise
    state.mark_done(step_name)
    logger.info(f"[DONE] '{step_name}'.")


def main():
    DATASET_DIR = "dataset"

    GAMES_ZST = "lichess_db_standard_rated_2019-06.pgn.zst"
    PUZZLE_ZST = "lichess_db_puzzle.csv.zst"
    EXTERNAL_CSV = "rawData/club_games_data.csv"

    STOCKFISH_PATH = "/usr/games/stockfish"

    MAX_GAMES_TIME_STATS = 50_000
    MAX_GAMES_PIPELINE = 180_000
    MAX_PUZZLES = 100_000
    MAX_GAMES_HOLDOUT = 500

    MATE_RANGE_TRAIN = (1, 5)
    MATE_RANGE_HOLDOUT = (1, 10)

    FORCE = False

    dataset_dir = DATASET_DIR
    puzzles_dir = os.path.join(dataset_dir, "puzzles")
    merged_dir = os.path.join(dataset_dir, "merged")
    for d in (dataset_dir, puzzles_dir, merged_dir):
        os.makedirs(d, exist_ok=True)

    state_path = os.path.join(dataset_dir, "pipeline_state.json")
    if FORCE and os.path.exists(state_path):
        os.remove(state_path)
    state = PipelineState(state_path)

    games_zst_path = os.path.join(GAMES_ZST)
    puzzle_zst_path = os.path.join(PUZZLE_ZST)
    external_csv_path = os.path.join(EXTERNAL_CSV)

    time_stats_path = os.path.join(dataset_dir, "avg_time_by_rating.json")
    puzzle_csv_path = os.path.join(dataset_dir, "lichess_puzzles.csv")
    games_output_base = os.path.join(dataset_dir, "games.pt")
    holdout_path = os.path.join(merged_dir, "external_holdout.pt")

    
    ctx = {}

    # --- Step 1: statistiche tempo medio per rating (da games reali) ---
    def _step_time_stats():
        require_file(games_zst_path, "Servono i game .pgn.zst reali per calcolare i tempi medi.")
        builder = TimeStatsBuilder(
            zst_path=games_zst_path,
            max_games=MAX_GAMES_TIME_STATS,
            bucket_size=100,
        )
        stats = builder.build_and_save(time_stats_path)
        ctx["avg_time_by_rating"] = stats

    run_step(
        state, "time_stats",
        is_ready_fn=lambda: file_ready(time_stats_path),
        do_fn=_step_time_stats,
    )
    if "avg_time_by_rating" not in ctx:
        ctx["avg_time_by_rating"] = load_avg_time_by_rating(time_stats_path)
    logger.info(f"avg_time_by_rating: {len(ctx['avg_time_by_rating'])} bucket -> {time_stats_path}")

    # --- Step 2: games Lichess -> posizioni di matto (train/val/test) ---
    games_paths = {
        "train": f"{os.path.splitext(games_output_base)[0]}_train.pt",
        "val": f"{os.path.splitext(games_output_base)[0]}_val.pt",
        "test": f"{os.path.splitext(games_output_base)[0]}_test.pt",
    }

    def _step_games_pipeline():
        require_file(games_zst_path, "Servono i game .pgn.zst per estrarre posizioni di matto.")
        require_executable(STOCKFISH_PATH, "Serve Stockfish per l'analisi delle posizioni.")
        pipeline = ChessAnalysisPipeline(
            zst_path=games_zst_path,
            stockfish_path=STOCKFISH_PATH,
            output_pt=games_output_base,
            mate_range=tuple(MATE_RANGE_HOLDOUT),
            max_games=MAX_GAMES_PIPELINE,
        )
        splits, paths = pipeline.run()
        ctx["games_splits"] = splits
        ctx["games_paths"] = paths

    run_step(
        state, "games_pipeline",
        is_ready_fn=lambda: all(torch_pt_ready(p) for p in games_paths.values()),
        do_fn=_step_games_pipeline,
    )
    if "games_splits" not in ctx:
        ctx["games_splits"] = {name: torch.load(p, weights_only=False) for name, p in games_paths.items()}
    for name, dlist in ctx["games_splits"].items():
        logger.info(f"Games split '{name}': {len(dlist)} posizioni-grafo.")

    # --- Step 3: puzzle Lichess -> decompressione + split train/val/test ---
    def _step_decompress_puzzles():
        decompress_zst_csv(zst_path=puzzle_zst_path, out_csv=puzzle_csv_path)

    run_step(
        state, "decompress_puzzles",
        is_ready_fn=lambda: file_ready(puzzle_csv_path),
        do_fn=_step_decompress_puzzles,
    )

    puzzle_processed_paths = {
        split: os.path.join(puzzles_dir, "processed", f"puzzle_{split}.pt")
        for split in ("train", "val", "test")
    }

    def _step_build_puzzles():
        splits = build_puzzle_pt(
            csv_path=puzzle_csv_path,
            root=puzzles_dir,
            mate_range=tuple(MATE_RANGE_TRAIN),
            max_puzzles=MAX_PUZZLES,
            avg_time_by_rating=ctx["avg_time_by_rating"],
        )
        ctx["puzzle_splits"] = splits

    run_step(
        state, "build_puzzles",
        is_ready_fn=lambda: all(torch_pt_ready(p) for p in puzzle_processed_paths.values()),
        do_fn=_step_build_puzzles,
    )
    if "puzzle_splits" not in ctx:
        ctx["puzzle_splits"] = build_puzzle_pt(
            csv_path=puzzle_csv_path,
            root=puzzles_dir,
            mate_range=tuple(MATE_RANGE_TRAIN),
            max_puzzles=MAX_PUZZLES,
            avg_time_by_rating=ctx["avg_time_by_rating"],
        )

    # --- Step 4: merge puzzle + games PER-SPLIT -> dataset/merged/merged_{train,val,test}.pt ---
    merged_paths = {
        split: os.path.join(merged_dir, f"merged_{split}.pt")
        for split in ("train", "val", "test")
    }

    def _step_merge():
        merge_and_split(
            puzzle_splits=ctx["puzzle_splits"],
            games_splits=ctx["games_splits"],
            out_dir=merged_dir,
        )

    run_step(
        state, "merge_and_split",
        is_ready_fn=lambda: all(torch_pt_ready(p) for p in merged_paths.values()),
        do_fn=_step_merge,
    )

    # --- Step 5: held-out ESTERNO (chess.com), MAI mischiato col training ---
    def _step_holdout():
        build_external_holdout(
            external_csv=external_csv_path,
            stockfish_path=STOCKFISH_PATH,
            out_pt=holdout_path,
            mate_range=tuple(MATE_RANGE_HOLDOUT),
            max_games=MAX_GAMES_HOLDOUT,
        )

    run_step(
        state, "external_holdout",
        is_ready_fn=lambda: torch_pt_ready(holdout_path),
        do_fn=_step_holdout,
    )

    logger.info(f"Train/val/test pronti in {merged_dir}/merged_{{train,val,test}}.pt")
    logger.info(f"Held-out ESTERNO (eval finale) in {holdout_path}")
    logger.info("NB: per Component/Trainer.py usa merged_train.pt / merged_val.pt.")
    logger.info("Per la valutazione finale/comparativa GNN vs LLM usa external_holdout.pt, non merged_test.pt.")


if __name__ == "__main__":
    try:
        main()
    except PipelineConfigError as e:
        logger.error(f"Errore di configurazione, correggi e rilancia: {e}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Pipeline interrotta da errore non gestito: {e}")
        logger.error("Rilanciando main.py riprendera' dagli step non ancora completati.")
        raise
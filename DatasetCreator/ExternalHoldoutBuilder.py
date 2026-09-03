import atexit
import io
import json
import multiprocessing as mp
import os
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import pandas as pd
import torch
from tqdm import tqdm

from DatasetCreator.GraphBuilder import GraphBuilder

def _closest_bucket_time(rating: Optional[int], avg_time_by_rating: Dict[int, float], default: float) -> float:
    """Trova il tempo medio del bucket di rating piu' vicino. Fallback su 'default'
    se rating e' None o il dizionario e' vuoto."""
    if rating is None or not avg_time_by_rating:
        return default
    closest_bucket = min(avg_time_by_rating.keys(), key=lambda b: abs(b - rating))
    return avg_time_by_rating[closest_bucket]

_engine: Optional[chess.engine.SimpleEngine] = None
_engine_pid: Optional[int] = None

_watchdog_lock = threading.Lock()
_watchdog_deadline: Optional[float] = None
_watchdog_stop = threading.Event()
_watchdog_thread: Optional[threading.Thread] = None
_WATCHDOG_POLL_SECONDS = 1.0
_WATCHDOG_MARGIN_SECONDS = 3.0


def _watchdog_arm(time_limit: float, margin: Optional[float] = None) -> None:
    global _watchdog_deadline
    eff_margin = _WATCHDOG_MARGIN_SECONDS if margin is None else margin
    with _watchdog_lock:
        _watchdog_deadline = time.monotonic() + time_limit + eff_margin


def _watchdog_disarm() -> None:
    global _watchdog_deadline
    with _watchdog_lock:
        _watchdog_deadline = None


def _watchdog_loop() -> None:
    global _engine, _engine_pid, _watchdog_deadline
    while not _watchdog_stop.is_set():
        with _watchdog_lock:
            deadline = _watchdog_deadline
        if deadline is not None and time.monotonic() > deadline:
            pid = _engine_pid
            if pid is not None:
                try:
                    import psutil
                    psutil.Process(pid).kill()
                except Exception:
                    try:
                        os.kill(pid, 9)
                    except Exception:
                        pass
            _engine = None
            _engine_pid = None
            with _watchdog_lock:
                _watchdog_deadline = None
        _watchdog_stop.wait(_WATCHDOG_POLL_SECONDS)


def _close_engine() -> None:
    global _engine, _engine_pid
    _watchdog_stop.set()
    if _engine is not None:
        try:
            _engine.quit()
        except Exception:
            pass
        finally:
            _engine = None
            _engine_pid = None


class ExternalHoldoutBuilder:
    def __init__(
        self,
        external_csv: str,
        stockfish_path: str,
        out_pt: str,
        mate_range: Tuple[int, int] = (1, 10),
        time_limit: float = 0.3,
        avg_time_by_rating: Optional[Dict[int, float]] = None,
        default_clock_seconds: float = 6.66,
        pgn_col: str = "pgn",
        max_games_to_scan: Optional[int] = None,
        target_total_problems: int = 200,
        stratification_config: Optional[Dict[str, int]] = None,
        threads: int = 2,
        hash_mb: int = 256,
        require_move_match: bool = True,
        checkpoint_every: int = 50,
        workers: Optional[int] = None,
        chunk_games: int = 200,
        config_error_cls: type = ValueError,
        min_ply: int = 0,
        ply_sample_step: int = 1,
        max_piece_count: Optional[int] = 18,
        candidate_min_legal_moves: int = 1,
        candidate_max_legal_moves: Optional[int] = None,
        skip_if_in_check: bool = False,
        pool_join_timeout: Optional[float] = 15.0,
        jsonl_out_path: Optional[str] = None,
    ):
        self._config_error_cls = config_error_cls
        self.external_csv = external_csv
        self.stockfish_path = stockfish_path
        self.out_pt = out_pt
        self.mate_range = mate_range
        self.time_limit = time_limit
        self.pgn_col = pgn_col
        self.max_games_to_scan = max_games_to_scan
        self.target_total_problems = target_total_problems
        self.stratification_config = stratification_config or {}
        self.threads = threads
        self.hash_mb = hash_mb
        self.require_move_match = require_move_match
        self.checkpoint_every = checkpoint_every
        self.min_ply = max(0, min_ply)
        self.ply_sample_step = max(1, ply_sample_step)
        self.max_piece_count = max_piece_count
        self.candidate_min_legal_moves = candidate_min_legal_moves
        self.candidate_max_legal_moves = candidate_max_legal_moves
        self.skip_if_in_check = skip_if_in_check
        self.pool_join_timeout = pool_join_timeout
        self.jsonl_out_path = jsonl_out_path or (os.path.splitext(out_pt)[0] + ".jsonl")
        cpu_count = os.cpu_count() or 2
        self.workers = workers or max(1, cpu_count - 1)
        self.chunk_games = chunk_games
        self.avg_time_by_rating = avg_time_by_rating or {}
        self.default_clock_seconds = default_clock_seconds
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        lo, hi = self.mate_range
        if lo < 1:
            raise ValueError("mate_range deve iniziare da almeno 1.")
        if hi < lo:
            raise ValueError("mate_range non valido.")
        if self.workers < 1:
            raise ValueError("workers deve essere >= 1.")
        if self.threads < 1:
            raise ValueError("threads deve essere >= 1.")
        if self.time_limit <= 0:
            raise ValueError("time_limit deve essere > 0.")
        if self.chunk_games < 1:
            raise ValueError("chunk_games deve essere >= 1.")
        if self.target_total_problems < 1:
            raise ValueError("target_total_problems deve essere >= 1.")
        if self.candidate_min_legal_moves < 0:
            raise ValueError("candidate_min_legal_moves deve essere >= 0.")
        if (
            self.candidate_max_legal_moves is not None
            and self.candidate_max_legal_moves < self.candidate_min_legal_moves
        ):
            raise ValueError("candidate_max_legal_moves deve essere >= candidate_min_legal_moves.")
        if self.max_piece_count is not None and self.max_piece_count < 2:
            raise ValueError("max_piece_count deve essere >= 2 (almeno i due re).")

    @staticmethod
    def _init_worker(stockfish_path: str, threads: int, hash_mb: int) -> None:
        global _engine, _engine_pid, _watchdog_thread
        try:
            _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            _engine.configure({"Threads": threads, "Hash": hash_mb})
            try:
                _engine_pid = _engine.transport.get_pid()
            except Exception:
                _engine_pid = None
        except Exception as e:
            _engine = None
            _engine_pid = None
            raise RuntimeError(f"Impossibile avviare Stockfish: {e}")
        atexit.register(_close_engine)

        _watchdog_stop.clear()
        _watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
        _watchdog_thread.start()

    @staticmethod
    def _is_candidate_position(
        board: "chess.Board",
        max_piece_count: Optional[int],
        candidate_min_legal_moves: int,
        candidate_max_legal_moves: Optional[int],
        skip_if_in_check: bool,
    ) -> Optional[List["chess.Move"]]:
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            return None
        if max_piece_count is not None and len(board.piece_map()) > max_piece_count:
            return None

        legal_moves = list(board.legal_moves)

        if len(legal_moves) < candidate_min_legal_moves:
            return None
        if candidate_max_legal_moves is not None and len(legal_moves) > candidate_max_legal_moves:
            return None
        if skip_if_in_check and board.is_check():
            return None

        return legal_moves

    @staticmethod
    def _analyse_game(
        args: Tuple[int, str, int, int, float, bool, int, int, Optional[int], int, Optional[int], bool, Dict[int, float], float],
    ) -> Tuple[int, List[Dict[str, Any]]]:
        (
            game_idx,
            pgn_text,
            lo,
            hi,
            time_limit,
            require_move_match,
            min_ply,
            ply_sample_step,
            max_piece_count,
            candidate_min_legal_moves,
            candidate_max_legal_moves,
            skip_if_in_check,
            avg_time_by_rating,
            default_clock_seconds,
        ) = args
        global _engine

        if _engine is None:
            return game_idx, []

        try:
            game = chess.pgn.read_game(io.StringIO(pgn_text))
        except Exception:
            return game_idx, []
        if game is None:
            return game_idx, []

        found: List[Dict[str, Any]] = []
        node = game
        white_elo_raw = game.headers.get("WhiteElo", "")
        black_elo_raw = game.headers.get("BlackElo", "")
        try:
            white_elo = int(white_elo_raw)
        except ValueError:
            white_elo = None
        try:
            black_elo = int(black_elo_raw)
        except ValueError:
            black_elo = None
        while node.variations:
            nxt = node.variation(0)

            if node.ply() < min_ply:
                node = nxt
                continue
            if (node.ply() - min_ply) % ply_sample_step != 0:
                node = nxt
                continue

            board = node.board()

            legal_moves = ExternalHoldoutBuilder._is_candidate_position(
                board,
                max_piece_count,
                candidate_min_legal_moves,
                candidate_max_legal_moves,
                skip_if_in_check,
            )
            if legal_moves is None:
                node = nxt
                continue

            try:
                _watchdog_arm(time_limit)
                info = _engine.analyse(board, chess.engine.Limit(time=time_limit, mate=hi), multipv=1)
            except (
                chess.engine.EngineTerminatedError,
                chess.engine.EngineError,
                BrokenPipeError,
                ConnectionResetError,
                OSError,
            ):
                break
            except Exception:
                node = nxt
                continue
            finally:
                _watchdog_disarm()

            score = info[0].get("score") if info else None
            if score and score.relative.is_mate():
                mate_n = score.relative.mate()
                if mate_n is not None and lo <= mate_n <= hi:
                    pv = info[0].get("pv")
                    engine_best_move = pv[0] if pv else None
                    if engine_best_move and not (require_move_match and nxt.move != engine_best_move):
                        try:
                            best_idx = legal_moves.index(engine_best_move)
                        except ValueError:
                            best_idx = None
                            if best_idx is not None:
                                mover_elo = white_elo if board.turn == chess.WHITE else black_elo
                                clock_seconds = _closest_bucket_time(
                                mover_elo, avg_time_by_rating, default_clock_seconds
                            )
                            found.append(
                                {
                                    "fen": board.fen(),
                                    "mate_n": int(mate_n),
                                    "best_move_idx": int(best_idx),
                                    "best_move_uci": engine_best_move.uci(),
                                    "problem_id": f"chesscom_{game_idx}_{node.ply()}",
                                    "clock_seconds": clock_seconds,
                                }
                            )
            node = nxt

        return game_idx, found

    def _require_file(self, path: str, hint: str = "") -> None:
        if not os.path.exists(path):
            msg = f"File richiesto non trovato: {path}."
            if hint:
                msg += f" {hint}"
            raise self._config_error_cls(msg)

    def _require_executable(self, path: str, hint: str = "") -> None:
        if not (os.path.exists(path) and os.access(path, os.X_OK)):
            msg = f"Eseguibile non trovato o permessi di esecuzione mancanti: {path}."
            if hint:
                msg += f" {hint}"
            raise self._config_error_cls(msg)

    def run(self) -> List[Any]:
        self._require_file(self.external_csv, "Verificare il file CSV esterno per l'held-out.")
        self._require_executable(self.stockfish_path, "Verificare il percorso del binario Stockfish.")

        df = pd.read_csv(self.external_csv)
        if self.max_games_to_scan:
            df = df.head(self.max_games_to_scan)
        if self.pgn_col not in df.columns:
            raise self._config_error_cls(
                f"Colonna '{self.pgn_col}' assente in {self.external_csv}. "
                f"Colonne disponibili: {list(df.columns)}"
            )

        lo, hi = self.mate_range
        t_1_5 = self.stratification_config.get("n_1_to_5_target_each", 30)
        t_6_10 = self.stratification_config.get("n_6_to_10_target_each", 10)
        strat_targets: Dict[int, int] = {n: (t_1_5 if n <= 5 else t_6_10) for n in range(lo, hi + 1)}

        counts_by_n: Dict[int, int] = defaultdict(int)
        test_data: List[Any] = []
        text_records: List[Dict[str, Any]] = []
        tmp_out = self.out_pt + ".tmp"

        def save_checkpoint():
            os.makedirs(os.path.dirname(os.path.abspath(self.out_pt)) or ".", exist_ok=True)
            torch.save(test_data, tmp_out)
            os.replace(tmp_out, self.out_pt)
            self._save_jsonl(text_records)

        def targets_satisfied() -> bool:
            if len(test_data) >= self.target_total_problems:
                return True
            return all(counts_by_n[n] >= strat_targets[n] for n in range(lo, hi + 1))

        pgn_pairs = list(df[self.pgn_col].dropna().items())
        chunks = [pgn_pairs[i : i + self.chunk_games] for i in range(0, len(pgn_pairs), self.chunk_games)]

        pool = mp.Pool(
            processes=self.workers,
            initializer=self._init_worker,
            initargs=(self.stockfish_path, self.threads, self.hash_mb),
        )

        processed_games = 0
        try:
            pbar = tqdm(total=len(pgn_pairs), desc="Estrazione Held-Out")
            for chunk in chunks:
                if targets_satisfied():
                    break
                if targets_satisfied():
                    break

                tasks = [
                    (
                        game_idx,
                        pgn_text,
                        lo,
                        hi,
                        self.time_limit,
                        self.require_move_match,
                        self.min_ply,
                        self.ply_sample_step,
                        self.max_piece_count,
                        self.candidate_min_legal_moves,
                        self.candidate_max_legal_moves,
                        self.skip_if_in_check,
                        self.avg_time_by_rating,
                        self.default_clock_seconds,
                    )
                    for game_idx, pgn_text in chunk
                ]

                for game_idx, found in pool.imap_unordered(self._analyse_game, tasks, chunksize=1):
                    processed_games += 1
                    pbar.update(1)

                    for item in found:
                        if targets_satisfied():
                            break
                        mate_n = item["mate_n"]
                        if counts_by_n[mate_n] >= strat_targets.get(mate_n, 9999):
                            continue

                        board = chess.Board(item["fen"])
                        label = {"mate_n": mate_n, "best_move_idx": item["best_move_idx"]}
                        data_item = GraphBuilder.board_to_pyg_data(board, clock_seconds=0.0, label=label)                        
                        board = chess.Board(item["fen"])
                        label = {"mate_n": mate_n, "best_move_idx": item["best_move_idx"]}
                        data_item = GraphBuilder.board_to_pyg_data(
                            board, clock_seconds=item["clock_seconds"], label=label
                        )

                        data_item.chain_edge_index = torch.empty((2, 0), dtype=torch.long)
                        data_item.problem_id = item["problem_id"]
                        data_item.mate_n = mate_n
                        data_item.fen = item["fen"]
                        data_item.best_move_uci = item["best_move_uci"]

                        test_data.append(data_item)
                        text_records.append(
                            {
                                "problem_id": item["problem_id"],
                                "fen": item["fen"],
                                "mate_n": mate_n,
                                "best_move_uci": item["best_move_uci"],
                            }
                        )
                        counts_by_n[mate_n] += 1

                        if len(test_data) % self.checkpoint_every == 0:
                            save_checkpoint()

                if targets_satisfied():
                    break
            pbar.close()
        finally:
            pool.close()

            if self.pool_join_timeout is not None:
                deadline = time.monotonic() + self.pool_join_timeout

                for proc in pool._pool:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    proc.join(timeout=remaining)

                still_alive = [proc for proc in pool._pool if proc.is_alive()]

                if still_alive:
                    stale_pids = [proc.pid for proc in still_alive]
                    print(
                        f"\n[WARNING] {len(still_alive)} worker non terminati entro "
                        f"{self.pool_join_timeout}s dopo pool.close(). Forzo pool.terminate()."
                    )
                    pool.terminate()

                    try:
                        import psutil

                        for pid in stale_pids:
                            try:
                                parent = psutil.Process(pid)
                                for child in parent.children(recursive=True):
                                    child.terminate()
                            except psutil.NoSuchProcess:
                                continue
                    except ImportError:
                        print(
                            "[WARNING] Modulo 'psutil' non disponibile: impossibile "
                            "terminare automaticamente processi orfani residui."
                        )

            pool.join()

        save_checkpoint()
        dist_str = ", ".join(f"n={n}: {counts_by_n[n]}/{strat_targets[n]}" for n in sorted(strat_targets.keys()))
        print(f"Held-out completato: {len(test_data)} problemi salvati in {self.out_pt} (Distribuzione: {dist_str}).")
        print(f"Held-out testuale salvato in {self.jsonl_out_path}.")
        print(f"Partite analizzate: {processed_games}")
        return test_data

    def _save_jsonl(self, text_records: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.jsonl_out_path)) or ".", exist_ok=True)
        tmp_path = self.jsonl_out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in text_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self.jsonl_out_path)
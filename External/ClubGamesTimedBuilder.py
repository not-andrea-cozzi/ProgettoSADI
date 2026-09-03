"""
ClubGamesTimedBuilder.py

Costruisce un dataset esterno di validazione da
DatasetCreator/RawData/club_games_data.csv.zip (colonna "pgn", una partita
chess.com per riga, stesso file gia' usato da ExternalHoldoutBuilder per
l'held-out mate 1-10) MA con una differenza fondamentale: qui si usa il
CLOCK REALE per-mossa (commenti {[%clk H:MM:SS]} chess.com) per calcolare
move_duration, esattamente come fa ChessAnalysisPipeline.py per il dataset
di TRAIN (games_pipeline) -- ExternalHoldoutBuilder invece ignora il tempo
(clock_seconds=0.0 costante).

Motivazione: dato che in training timed batte untimed ma sul test interno
succede l'opposto, questo dataset serve a verificare la stessa metrica
(move accuracy timed vs untimed, stratificata per mate_n) su partite
chess.com MAI viste in training, con segnale temporale reale e non
fittizio -- a differenza sia di HFFenDatasetBuilder (clock costante) sia
di ExternalHoldoutBuilder (clock_seconds=0.0).

Skip iniziale: le prime `skip_games` righe del CSV vengono saltate. Questo
e' pensato per evitare overlap con l'held-out gia' generato da
ExternalHoldoutBuilder, che di default scansiona le partite a partire
dall'inizio del file (vedi Yaml/main.yaml -> external_holdout.max_games_to_scan).
Con skip_games=10000 si valida su una fetta del CSV diversa da quella gia'
usata per l'held-out esistente.

A differenza di HFFenDatasetBuilder (che riceveva solo una FEN + una mossa
per riga, senza contesto di partita), qui si riusa lo stesso schema di
scansione mossa-per-mossa di ChessAnalysisPipeline: filtri economici,
Stockfish con Limit(mate=hi), ply_sample_step per non controllare ogni
singola mezza-mossa, watchdog per Stockfish appeso, checkpoint incrementale.

Gestione interruzioni (Ctrl-C): su terminale POSIX, SIGINT viene inoltrato
a tutto il process group, quindi anche ai worker della Pool. Se un worker
viene interrotto a meta' operazione puo' rompere la pipe verso il
result_handler interno della Pool, lasciando cache/result_handler in stato
incoerente e facendo esplodere i finalizer di atexit con
"AssertionError: Cannot have cache with result_handler not alive". Per
questo i worker ignorano SIGINT (solo il processo principale reagisce) e
run() esegue uno shutdown esplicito via pool.terminate()/join() prima di
ripropagare l'eccezione.
"""
from __future__ import annotations

import atexit
import io
import json
import multiprocessing as mp
import os
import re
import signal
import threading
import time
import zipfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import pandas as pd
import torch
from tqdm import tqdm

from DatasetCreator.GraphBuilder import GraphBuilder

_engine: Optional[chess.engine.SimpleEngine] = None
_engine_pid: Optional[int] = None

import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

_watchdog_lock = threading.Lock()
_watchdog_deadline: Optional[float] = None
_watchdog_stop = threading.Event()
_watchdog_thread: Optional[threading.Thread] = None
_WATCHDOG_POLL_SECONDS = 1.0
_WATCHDOG_MARGIN_SECONDS = 3.0

_CLK_RE = re.compile(r"\[\s*%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*\]")


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


def _worker_sigterm_handler(signum, frame) -> None:
    """Eseguito nel worker quando la Pool chiama pool.terminate() (SIGTERM).
    Senza questo handler l'engine Stockfish del worker resterebbe orfano,
    perche' atexit non gira su una terminazione per segnale."""
    _close_engine()
    os._exit(0)


class ClubGamesTimedBuilder:
    """Costruisce un .pt di validazione esterna da club_games_data.csv.zip
    usando il clock reale per-mossa (move_duration), non un valore fittizio."""

    _PIECE_VALUES: Dict[int, int] = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    def __init__(
        self,
        csv_path: str,
        stockfish_path: str,
        out_pt: str,
        mate_range: Tuple[int, int] = (1, 10),
        analysis_time: float = 0.2,
        pgn_col: str = "pgn",
        skip_games: int = 10000,
        max_games_to_scan: Optional[int] = None,
        default_move_seconds: float = 15.0,
        require_clock: bool = False,
        min_ply: int = 0,
        ply_sample_step: int = 3,
        max_positions_per_game: Optional[int] = 20,
        threads: int = 1,
        hash_mb: int = 128,
        checkpoint_every: int = 50,
        workers: Optional[int] = None,
        chunk_games: int = 200,
        config_error_cls: type = ValueError,
        max_piece_count: Optional[int] = 18,
        min_material_for_mate_attempt: int = 4,
        candidate_min_legal_moves: int = 1,
        candidate_max_legal_moves: Optional[int] = None,
        skip_if_in_check: bool = False,
        only_decisive_games: bool = True,
        skip_time_forfeit: bool = True,
        min_game_plies: int = 20,
        pool_join_timeout: Optional[float] = 15.0,
        jsonl_out_path: Optional[str] = None,
    ):
        self._config_error_cls = config_error_cls
        self.csv_path = csv_path
        self.stockfish_path = stockfish_path
        self.out_pt = out_pt
        self.mate_range = mate_range
        self.analysis_time = analysis_time
        self.pgn_col = pgn_col
        self.skip_games = max(0, skip_games)
        self.max_games_to_scan = max_games_to_scan
        self.default_move_seconds = default_move_seconds
        self.require_clock = require_clock
        self.min_ply = max(0, min_ply)
        self.ply_sample_step = max(1, ply_sample_step)
        self.max_positions_per_game = max_positions_per_game
        self.threads = threads
        self.hash_mb = hash_mb
        self.checkpoint_every = checkpoint_every
        self.chunk_games = chunk_games
        self.max_piece_count = max_piece_count
        self.min_material_for_mate_attempt = min_material_for_mate_attempt
        self.candidate_min_legal_moves = candidate_min_legal_moves
        self.candidate_max_legal_moves = candidate_max_legal_moves
        self.skip_if_in_check = skip_if_in_check
        self.only_decisive_games = only_decisive_games
        self.skip_time_forfeit = skip_time_forfeit
        self.min_game_plies = min_game_plies
        self.pool_join_timeout = pool_join_timeout
        self.jsonl_out_path = jsonl_out_path or (os.path.splitext(out_pt)[0] + ".jsonl")
        cpu_count = os.cpu_count() or 2
        self.workers = workers or max(1, cpu_count - 1)

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        lo, hi = self.mate_range
        if lo < 1:
            raise self._config_error_cls("mate_range deve iniziare da almeno 1.")
        if hi < lo:
            raise self._config_error_cls("mate_range non valido.")
        if self.workers < 1:
            raise self._config_error_cls("workers deve essere >= 1.")
        if self.threads < 1:
            raise self._config_error_cls("threads deve essere >= 1.")
        if self.analysis_time <= 0:
            raise self._config_error_cls("analysis_time deve essere > 0.")
        if self.chunk_games < 1:
            raise self._config_error_cls("chunk_games deve essere >= 1.")
        if self.candidate_min_legal_moves < 0:
            raise self._config_error_cls("candidate_min_legal_moves deve essere >= 0.")
        if (
            self.candidate_max_legal_moves is not None
            and self.candidate_max_legal_moves < self.candidate_min_legal_moves
        ):
            raise self._config_error_cls("candidate_max_legal_moves deve essere >= candidate_min_legal_moves.")
        if self.max_piece_count is not None and self.max_piece_count < 2:
            raise self._config_error_cls("max_piece_count deve essere >= 2 (almeno i due re).")
        if self.min_material_for_mate_attempt < 0:
            raise self._config_error_cls("min_material_for_mate_attempt deve essere >= 0.")
        if self.skip_games < 0:
            raise self._config_error_cls("skip_games deve essere >= 0.")

    # ================================================================
    # WORKER INIT
    # ================================================================

    @staticmethod
    def _init_worker(stockfish_path: str, threads: int, hash_mb: int) -> None:
        global _engine, _engine_pid, _watchdog_thread

        # Solo il processo principale deve reagire a Ctrl-C. Se anche i
        # worker ricevono SIGINT (comportamento di default: il terminale lo
        # inoltra a tutto il process group) si rischia di spezzare la pipe
        # verso i thread interni della Pool a meta' operazione, lasciando
        # cache/result_handler in stato incoerente e facendo esplodere il
        # finalizer di atexit con "Cannot have cache with result_handler not
        # alive". I worker ignorano quindi SIGINT; la terminazione pulita e'
        # gestita da run() via pool.terminate(), che usa SIGTERM.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, _worker_sigterm_handler)

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

    # ================================================================
    # CLOCK / TIME CONTROL (identico a ChessAnalysisPipeline)
    # ================================================================

    @staticmethod
    def _parse_clock(comment: str) -> Optional[float]:
        if not comment:
            return None
        match = _CLK_RE.search(comment)
        if not match:
            return None
        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    @staticmethod
    def _parse_time_control(time_control: str) -> Tuple[float, float]:
        if not time_control or time_control == "-":
            return 0.0, 0.0
        match = re.match(r"^(\d+)\+(\d+)$", time_control)
        if match:
            return float(match.group(1)), float(match.group(2))
        match = re.match(r"^(\d+)$", time_control)
        if match:
            return float(match.group(1)), 0.0
        return 0.0, 0.0

    @staticmethod
    def _compute_move_duration(
        previous_clock: Optional[float], current_clock: Optional[float], increment: float
    ) -> Optional[float]:
        if previous_clock is None or current_clock is None:
            return None
        spent = previous_clock - current_clock + increment
        return max(0.0, spent)

    # ================================================================
    # FILTRI ECONOMICI (identici a ChessAnalysisPipeline / ExternalHoldoutBuilder)
    # ================================================================

    def _game_is_eligible(self, game: "chess.pgn.Game") -> bool:
        if game is None:
            return False
        try:
            ply_count = game.end().ply()
        except Exception:
            return False
        if ply_count < self.min_game_plies:
            return False
        if self.only_decisive_games:
            result = game.headers.get("Result", "")
            if result not in ("1-0", "0-1"):
                return False
        if self.skip_time_forfeit:
            termination = game.headers.get("Termination", "")
            if "Time forfeit" in termination:
                return False
        return True

    def _get_candidate_legal_moves(self, board: "chess.Board") -> Optional[List["chess.Move"]]:
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            return None
        if self.max_piece_count is not None and len(board.piece_map()) > self.max_piece_count:
            return None

        legal_moves = list(board.legal_moves)
        if len(legal_moves) < self.candidate_min_legal_moves:
            return None
        if self.candidate_max_legal_moves is not None and len(legal_moves) > self.candidate_max_legal_moves:
            return None
        if self.skip_if_in_check and board.is_check():
            return None
        return legal_moves

    def _has_mating_material(self, board: "chess.Board") -> bool:
        mover = board.turn
        material = sum(
            self._PIECE_VALUES.get(piece.piece_type, 0)
            for piece in board.piece_map().values()
            if piece.color == mover
        )
        return material >= self.min_material_for_mate_attempt

    # ================================================================
    # ANALISI DI UNA PARTITA (eseguita nel worker)
    # ================================================================

    def _analyse_position(self, board: "chess.Board"):
        global _engine
        if _engine is None:
            return None
        try:
            limit = chess.engine.Limit(time=self.analysis_time, mate=self.mate_range[1])
            _watchdog_arm(self.analysis_time)
            return _engine.analyse(board, limit, multipv=1)
        except (
            chess.engine.EngineTerminatedError,
            chess.engine.EngineError,
            BrokenPipeError,
            ConnectionResetError,
            OSError,
        ):
            return None
        except Exception:
            return None
        finally:
            _watchdog_disarm()

    def _worker(self, args: Tuple[int, str]) -> Tuple[int, List[Any]]:
        global _engine
        game_idx, pgn_text = args

        if _engine is None:
            return game_idx, []

        try:
            game = chess.pgn.read_game(io.StringIO(pgn_text))
        except Exception:
            return game_idx, []
        if not self._game_is_eligible(game):
            return game_idx, []

        time_control = game.headers.get("TimeControl", "")
        base_time, increment = self._parse_time_control(time_control)

        previous_clock = {
            chess.WHITE: base_time if base_time > 0 else None,
            chess.BLACK: base_time if base_time > 0 else None,
        }

        data_list: List[Any] = []
        node = game
        mate_lo, mate_hi = self.mate_range
        positions_analysed = 0

        while node.variations:
            next_node = node.variation(0)
            board = node.board()
            comment = next_node.comment or ""
            mover_color = board.turn

            current_clock = self._parse_clock(comment)
            move_duration = self._compute_move_duration(previous_clock[mover_color], current_clock, increment)
            if current_clock is not None:
                previous_clock[mover_color] = current_clock

            if node.ply() < self.min_ply:
                node = next_node
                continue
            if (node.ply() - self.min_ply) % self.ply_sample_step != 0:
                node = next_node
                continue
            if self.require_clock and current_clock is None:
                node = next_node
                continue
            if self.max_positions_per_game is not None and positions_analysed >= self.max_positions_per_game:
                break

            legal_moves = self._get_candidate_legal_moves(board)
            if legal_moves is None:
                node = next_node
                continue
            if not self._has_mating_material(board):
                node = next_node
                continue

            info = self._analyse_position(board)
            positions_analysed += 1
            if not info:
                node = next_node
                continue

            best_info = info[0]
            score = best_info.get("score")
            if score is None:
                node = next_node
                continue
            relative_score = score.relative
            if not relative_score.is_mate():
                node = next_node
                continue
            mate_n = relative_score.mate()
            if mate_n is None or not (mate_n > 0 and mate_lo <= mate_n <= mate_hi):
                node = next_node
                continue

            pv = best_info.get("pv")
            if not pv:
                node = next_node
                continue
            best_move = pv[0]
            try:
                best_move_idx = legal_moves.index(best_move)
            except ValueError:
                node = next_node
                continue

            clock_seconds = move_duration if move_duration is not None else self.default_move_seconds

            label = {"mate_n": int(mate_n), "best_move_idx": int(best_move_idx)}
            try:
                data = GraphBuilder.board_to_pyg_data(
                    board, clock_seconds=clock_seconds, label=label, legal_moves=legal_moves
                )
            except Exception:
                node = next_node
                continue

            data.chain_edge_index = torch.empty((2, 0), dtype=torch.long)
            data.game_id = int(game_idx)
            data.ply = int(node.ply())
            data.clock_seconds = float(clock_seconds)
            data.clock_is_real = bool(move_duration is not None)
            data.mate_n = int(mate_n)
            data.best_move_idx = int(best_move_idx)
            data.best_move_uci = best_move.uci()
            data.fen = board.fen()
            data.problem_id = f"club_{game_idx}_{node.ply()}"

            data_list.append(data)
            node = next_node

        return game_idx, data_list

    # ================================================================
    # CSV READING (una partita per riga, con skip)
    # ================================================================

    def _read_pgn_rows(self):
        """Legge il csv (anche dentro uno zip) e ritorna (row_idx, pgn_text)
        a partire da skip_games, fino a max_games_to_scan se specificato."""
        if self.csv_path.endswith(".zip"):
            with zipfile.ZipFile(self.csv_path) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    raise self._config_error_cls(f"Nessun CSV trovato dentro {self.csv_path}.")
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
        else:
            df = pd.read_csv(self.csv_path)

        if self.pgn_col not in df.columns:
            raise self._config_error_cls(
                f"Colonna '{self.pgn_col}' assente in {self.csv_path}. Colonne disponibili: {list(df.columns)}"
            )

        series = df[self.pgn_col].dropna()
        series = series.iloc[self.skip_games :]
        if self.max_games_to_scan is not None:
            series = series.iloc[: self.max_games_to_scan]
        return list(series.items())

    # ================================================================
    # RUN
    # ================================================================

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
        self._require_file(self.csv_path, "Verificare il percorso di club_games_data.csv.zip.")
        self._require_executable(self.stockfish_path, "Verificare il percorso del binario Stockfish.")

        pgn_pairs = self._read_pgn_rows()
        if not pgn_pairs:
            raise RuntimeError(
                f"Nessuna riga da processare dopo skip_games={self.skip_games} "
                f"(e max_games_to_scan={self.max_games_to_scan})."
            )
        logger_total = len(pgn_pairs)

        found_data: List[Any] = []
        text_records: List[Dict[str, Any]] = []
        tmp_out = self.out_pt + ".tmp"
        counts_by_n: Dict[int, int] = defaultdict(int)

        def save_checkpoint():
            os.makedirs(os.path.dirname(os.path.abspath(self.out_pt)) or ".", exist_ok=True)
            torch.save(found_data, tmp_out)
            os.replace(tmp_out, self.out_pt)
            self._save_jsonl(text_records)

        chunks = [pgn_pairs[i : i + self.chunk_games] for i in range(0, len(pgn_pairs), self.chunk_games)]

        pool = mp.Pool(
            processes=self.workers,
            initializer=self._init_worker,
            initargs=(self.stockfish_path, self.threads, self.hash_mb),
        )

        processed_games = 0
        accepted_games = 0
        pbar = tqdm(total=logger_total, desc=f"Scansione club_games (skip={self.skip_games})")

        def cleanup_after_failure() -> None:
            # Ignora un secondo Ctrl-C durante lo shutdown stesso, cosi' non
            # puo' interrompere pool.terminate()/join() a meta' e ricreare
            # lo stesso stato incoerente che si vuole evitare.
            old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                pbar.close()
                pool.terminate()
                pool.join()
                if found_data:
                    save_checkpoint()
            finally:
                signal.signal(signal.SIGINT, old_sigint)

        try:
            for chunk in chunks:
                tasks = list(chunk)
                for game_idx, data_list in pool.imap_unordered(self._worker, tasks, chunksize=1):
                    processed_games += 1
                    pbar.update(1)

                    if not data_list:
                        continue
                    accepted_games += 1

                    for data_item in data_list:
                        found_data.append(data_item)
                        text_records.append(
                            {
                                "problem_id": data_item.problem_id,
                                "fen": data_item.fen,
                                "mate_n": data_item.mate_n,
                                "best_move_uci": data_item.best_move_uci,
                                "clock_seconds": data_item.clock_seconds,
                                "clock_is_real": data_item.clock_is_real,
                            }
                        )
                        counts_by_n[data_item.mate_n] += 1

                    if len(found_data) % self.checkpoint_every < len(data_list):
                        save_checkpoint()
        except KeyboardInterrupt:
            print("\n[WARNING] Interruzione richiesta dall'utente: arresto pulito dei worker in corso...")
            cleanup_after_failure()
            raise
        except Exception:
            print("\n[WARNING] Errore durante la scansione: arresto pulito dei worker in corso...")
            cleanup_after_failure()
            raise

        pbar.close()
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
        dist_str = ", ".join(f"n={n}: {counts_by_n[n]}" for n in sorted(counts_by_n.keys()))
        real_clock_count = sum(1 for d in found_data if getattr(d, "clock_is_real", False))
        print(
            f"Scansione completata: {processed_games} partite processate "
            f"(skip iniziale={self.skip_games}), {accepted_games} con almeno una posizione mate, "
            f"{len(found_data)} posizioni totali salvate in {self.out_pt}."
        )
        print(f"Distribuzione per n: {dist_str}.")
        print(
            f"Posizioni con clock REALE (da %clk): {real_clock_count}/{len(found_data)} "
            f"({100 * real_clock_count / max(len(found_data), 1):.1f}%). Le restanti usano "
            f"default_move_seconds={self.default_move_seconds} (partita senza clock o mossa non annotata)."
        )
        print(f"Testuale salvato in {self.jsonl_out_path}.")
        return found_data

    def _save_jsonl(self, text_records: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.jsonl_out_path)) or ".", exist_ok=True)
        tmp_path = self.jsonl_out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in text_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self.jsonl_out_path)
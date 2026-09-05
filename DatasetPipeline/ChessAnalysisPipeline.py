import atexit
import io
import os
import re
import random
import multiprocessing as mp
from typing import Tuple, Optional, Dict, List, Generator

import chess
import chess.pgn
import chess.engine
import zstandard as zstd
import torch
from tqdm import tqdm
import torch.multiprocessing as torch_mp

from DatasetPipeline.GraphBuilder import GraphBuilder

# Evita problemi di sharing dei tensori tra processi PyTorch.
torch_mp.set_sharing_strategy("file_system")

# Ogni worker possiede la propria istanza Stockfish.
_engine: Optional[chess.engine.SimpleEngine] = None

# Ogni worker possiede la propria istanza Syzygy (se configurata).
# None se syzygy_path non e' stato fornito, oppure se l'apertura
# dei file tablebase fallisce: in entrambi i casi lo screening WDL
# viene semplicemente saltato (fallback trasparente su Stockfish).
_tablebase: Optional["chess.syzygy.Tablebase"] = None

# Esempi supportati:
# [ %clk 0:05:32.4 ]
# [ %clk 1:23:45 ]
_CLK_RE = re.compile(r"\[\s*%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*\]")


def _close_engine() -> None:
    """
    Chiude in modo pulito l'istanza Stockfish (e Syzygy, se aperta)
    del worker corrente.

    FONDAMENTALE: senza questa chiusura esplicita, il sottoprocesso
    Stockfish resta in attesa su una read bloccante dalla pipe stdin
    (non ha mai ricevuto il comando UCI 'quit'). Quando il Pool prova
    a fare pool.join() dopo pool.close(), il processo worker Python
    non termina mai perche' il suo figlio Stockfish e' ancora vivo e
    appeso: l'intera pipeline si blocca indefinitamente in run(),
    anche a lavoro di analisi gia' completato, PRIMA del torch.save
    finale (quindi con perdita totale del lavoro svolto).

    Registrata sia come atexit (rete di sicurezza per terminazioni
    impreviste del worker) sia chiamata esplicitamente a fine _worker
    quando il pool sta per chiudere i processi.
    """
    global _engine, _tablebase

    if _engine is not None:
        try:
            _engine.quit()
        except Exception:
            # Se il processo Stockfish e' gia' morto o non risponde,
            # non c'e' nulla di sensato da fare: proseguiamo comunque
            # con la chiusura del worker.
            pass
        finally:
            _engine = None

    if _tablebase is not None:
        try:
            _tablebase.close()
        except Exception:
            pass
        finally:
            _tablebase = None


class ChessAnalysisPipeline:
    """
    Pipeline:
        PGN.zst
          |
          v
        streaming PGN
          |
          v
        filtri economici partita
          |
          v
        filtri economici posizione
          |
          v
        selezione posizioni candidate
          |
          v
        Stockfish
          |
          v
        selezione mate 1-5
          |
          v
        GraphBuilder
          |
          v
        train / val / test

    Training:
        mate 1-5

    La valutazione esterna mate 1-10 viene gestita
    separatamente dal MainDatasetCreator.
    """

    def __init__(
        self,
        zst_path: str,
        stockfish_path: str,
        output_pt: str,
        mate_range: Tuple[int, int] = (1, 5),
        search_depth: int = 6,
        workers: Optional[int] = None,
        threads: int = 1,
        hash_mb: int = 128,
        multipv: int = 1,
        max_games: Optional[int] = None,
        skip_games: int = 0,
        seed: int = 42,
        default_move_seconds: float = 15.0,
        require_clock: bool = False,
        min_ply: int = 16,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        analysis_time: Optional[float] = 0.15,
        min_game_plies: int = 20,
        skip_no_time_control: bool = False,
        max_positions_per_game: Optional[int] = 20,
        candidate_min_legal_moves: int = 1,
        candidate_max_legal_moves: Optional[int] = None,
        skip_if_in_check: bool = False,
        max_piece_count: Optional[int] = 18,
        only_decisive_games: bool = True,
        skip_time_forfeit: bool = True,
        ply_sample_step: int = 3,
        pool_join_timeout: Optional[float] = 15.0,
        syzygy_path: Optional[str] = None,
        min_material_for_mate_attempt: int = 4,
    ):
        self.zst_path = zst_path
        self.stockfish_path = stockfish_path
        self.output_pt = output_pt
        self.mate_range = mate_range
        self.search_depth = search_depth
        self.threads = threads
        self.hash_mb = hash_mb
        self.multipv = multipv
        self.max_games = max_games
        self.skip_games = skip_games
        self.seed = seed
        self.default_move_seconds = default_move_seconds
        self.require_clock = require_clock
        self.min_ply = min_ply
        self.min_game_plies = min_game_plies
        self.skip_no_time_control = skip_no_time_control
        self.max_positions_per_game = max_positions_per_game
        self.split_ratios = split_ratios
        self.analysis_time = analysis_time

        # Filtri pre-Stockfish.
        self.candidate_min_legal_moves = candidate_min_legal_moves
        self.candidate_max_legal_moves = candidate_max_legal_moves
        self.skip_if_in_check = skip_if_in_check
        # NUOVO: scarta posizioni troppo affollate (mate 1-5 raro con molto materiale).
        self.max_piece_count = max_piece_count
        # NUOVO: scarta partite patte (mai portano a mate) e vittorie per tempo
        # (spesso la posizione finale non è un vero mate tattico).
        self.only_decisive_games = only_decisive_games
        self.skip_time_forfeit = skip_time_forfeit
        # NUOVO: campiona 1 posizione ogni N ply invece di analizzarle tutte.
        # Un mate forzato in 1-5 resta rilevabile anche non controllando ogni
        # singola mezza-mossa: questo taglia direttamente il numero di
        # chiamate Stockfish, che e' il vero collo di bottiglia.
        self.ply_sample_step = max(1, ply_sample_step)

        # NUOVO: timeout massimo (secondi) per pool.join(). Se allo scadere
        # i worker non sono ancora terminati (es. Stockfish rimasto appeso
        # nonostante _close_engine), forziamo pool.terminate() invece di
        # restare bloccati per sempre. None disabilita il timeout
        # (comportamento equivalente al blocking join originale).
        self.pool_join_timeout = pool_join_timeout

        # NUOVO: cartella file Syzygy (.rtbw/.rtbz) per lo screening WDL
        # pre-Stockfish. None (default) disattiva completamente questa via:
        # nessuna dipendenza aggiuntiva se non hai i file tablebase.
        self.syzygy_path = syzygy_path

        # NUOVO: filtro economico basato sul materiale del lato che deve
        # muovere. Un matto forzato in poche mosse richiede tipicamente
        # abbastanza forza d'attacco (una donna, una coppia di torri, torre
        # + pezzo minore, ecc.). Il valore e' la somma dei valori standard
        # (Q=9,R=5,B=3,N=3,P=1) dei pezzi del lato di turno, re escluso.
        # Sotto soglia, un mate 1-5 e' statisticamente trascurabile: si
        # scarta la posizione prima di chiamare Stockfish. Il default (4)
        # e' volutamente permissivo (es. R da solo passa, R+minore passa,
        # ma K da solo o K+P da soli no) per non perdere veri mate.
        self.min_material_for_mate_attempt = min_material_for_mate_attempt

        # Per questo workload è generalmente preferibile avere
        # molti processi con un solo thread Stockfish ciascuno.
        cpu_count = os.cpu_count() or 2
        self.workers = workers or max(1, cpu_count - 1)

        self._validate_parameters()

    # ================================================================
    # VALIDATION
    # ================================================================

    def _validate_parameters(self) -> None:
        lo, hi = self.mate_range

        if lo < 1:
            raise ValueError("mate_range deve iniziare da almeno 1.")

        if hi < lo:
            raise ValueError("mate_range non valido.")

        if self.search_depth < 1:
            raise ValueError("search_depth deve essere >= 1.")

        if self.workers < 1:
            raise ValueError("workers deve essere >= 1.")

        if self.threads < 1:
            raise ValueError("threads deve essere >= 1.")

        if self.multipv < 1:
            raise ValueError("multipv deve essere >= 1.")

        if self.min_ply < 0:
            raise ValueError("min_ply deve essere >= 0.")

        if self.min_game_plies < 0:
            raise ValueError("min_game_plies deve essere >= 0.")

        if self.candidate_min_legal_moves < 0:
            raise ValueError(
                "candidate_min_legal_moves deve essere >= 0."
            )

        if (
            self.candidate_max_legal_moves is not None
            and self.candidate_max_legal_moves < self.candidate_min_legal_moves
        ):
            raise ValueError(
                "candidate_max_legal_moves deve essere >= "
                "candidate_min_legal_moves."
            )

        if self.analysis_time is not None and self.analysis_time <= 0:
            raise ValueError("analysis_time deve essere > 0.")

        if self.max_piece_count is not None and self.max_piece_count < 2:
            raise ValueError("max_piece_count deve essere >= 2 (almeno i due re).")

        if len(self.split_ratios) != 3:
            raise ValueError(
                "split_ratios deve contenere train, val e test."
            )

        if abs(sum(self.split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios deve sommare a 1.0.")
            
        if self.skip_games < 0:
            raise ValueError("skip_games deve essere >= 0.")

        if self.min_material_for_mate_attempt < 0:
            raise ValueError(
                "min_material_for_mate_attempt deve essere >= 0."
            )

    # ================================================================
    # STOCKFISH INITIALIZATION
    # ================================================================

    @staticmethod
    def _init_worker(
        stockfish_path: str,
        threads: int,
        hash_mb: int,
        syzygy_path: Optional[str] = None,
    ) -> None:
        """
        Crea una singola istanza Stockfish per worker, e opzionalmente
        una istanza Syzygy se syzygy_path e' stato configurato.
        """
        global _engine, _tablebase

        try:
            _engine = chess.engine.SimpleEngine.popen_uci(
                stockfish_path
            )

            _engine.configure(
                {
                    "Threads": threads,
                    "Hash": hash_mb,
                }
            )

        except Exception as e:
            _engine = None
            raise RuntimeError(
                f"Impossibile avviare Stockfish: {e}"
            )

        # Apertura Syzygy: solo best-effort. Se la cartella non esiste,
        # e' vuota, o mancano i moduli, NON blocchiamo la pipeline:
        # semplicemente lo screening WDL resta disattivato per questo
        # worker e tutto procede via Stockfish come prima.
        if syzygy_path:
            try:
                _tablebase = chess.syzygy.open_tablebase(syzygy_path)
            except Exception:
                _tablebase = None

        # Rete di sicurezza: garantisce che Stockfish (e Syzygy) ricevano
        # una chiusura pulita anche se il worker termina per vie diverse
        # dal normale ritorno di _worker (crash, eccezione non gestita).
        atexit.register(_close_engine)

    # ================================================================
    # CLOCK
    # ================================================================

    @staticmethod
    def _parse_clock(
        comment: str,
    ) -> Optional[float]:
        """
        Estrae il clock Lichess.

        Esempio:
            [ %clk 0:05:32.4 ]

        -> 332.4 secondi
        """
        if not comment:
            return None

        match = _CLK_RE.search(comment)

        if not match:
            return None

        hours, minutes, seconds = match.groups()

        return (
            int(hours) * 3600
            + int(minutes) * 60
            + float(seconds)
        )

    # ================================================================
    # TIME CONTROL
    # ================================================================

    @staticmethod
    def _parse_time_control(
        time_control: str,
    ) -> Tuple[float, float]:
        """
        Esempi:
            300+0 -> 300, 0
            300+5 -> 300, 5
            600   -> 600, 0
            -     -> 0, 0
        """
        if not time_control or time_control == "-":
            return 0.0, 0.0

        match = re.match(
            r"^(\d+)\+(\d+)$",
            time_control,
        )

        if match:
            return (
                float(match.group(1)),
                float(match.group(2)),
            )

        match = re.match(
            r"^(\d+)$",
            time_control,
        )

        if match:
            return (
                float(match.group(1)),
                0.0,
            )

        return 0.0, 0.0

    # ================================================================
    # MOVE DURATION
    # ================================================================

    @staticmethod
    def _compute_move_duration(
        previous_clock: Optional[float],
        current_clock: Optional[float],
        increment: float,
    ) -> Optional[float]:
        """
        Calcola il tempo speso sulla mossa.

        spent =
            previous_clock
            - current_clock
            + increment
        """
        if (
            previous_clock is None
            or current_clock is None
        ):
            return None

        spent = (
            previous_clock
            - current_clock
            + increment
        )

        return max(0.0, spent)

    # ================================================================
    # ECONOMIC GAME FILTER
    # ================================================================

    def _game_is_eligible(
        self,
        game: chess.pgn.Game,
    ) -> bool:
        """
        Filtro molto economico eseguito prima di Stockfish.

        NON determina se esiste un mate.
        Serve solamente a eliminare partite chiaramente inutili.
        """
        if game is None:
            return False

        try:
            ply_count = game.end().ply()
        except Exception:
            return False

        if ply_count < self.min_game_plies:
            return False

        if self.skip_no_time_control:
            time_control = game.headers.get(
                "TimeControl",
                "",
            )

            if not time_control or time_control == "-":
                return False

        # NUOVO: scarta partite patte -- un mate 1-5 puo' esistere solo
        # su una linea che porta a un vero scacco matto (1-0 / 0-1).
        if self.only_decisive_games:
            result = game.headers.get("Result", "")
            if result not in ("1-0", "0-1"):
                return False

        # NUOVO: le vittorie per tempo scaduto raramente corrispondono
        # a una posizione con un vero mate forzato entro pochi ply.
        if self.skip_time_forfeit:
            termination = game.headers.get("Termination", "")
            if "Time forfeit" in termination:
                return False

        return True

    # ================================================================
    # ECONOMIC POSITION FILTER
    # ================================================================

    def _get_candidate_legal_moves(
        self,
        board: chess.Board,
    ) -> Optional[List[chess.Move]]:
        """
        Filtro economico eseguito PRIMA di Stockfish.

        È volutamente conservativo:
        non cerca di stabilire se esiste un mate.

        Elimina soltanto posizioni terminali o chiaramente
        non adatte all'analisi.

        Restituisce direttamente le mosse legali così da
        evitare di calcolarle nuovamente dopo Stockfish.
        """

        # La posizione è già terminata.
        if board.is_checkmate():
            return None

        if board.is_stalemate():
            return None

        if board.is_insufficient_material():
            return None

        # NUOVO: filtro materiale economico (nessuna chiamata a Stockfish).
        # Mate forzato in 1-5 ply e' statisticamente raro su scacchiere
        # molto affollate; scartarle qui evita analisi costose che quasi
        # certamente non produrranno un mate nel range richiesto.
        if self.max_piece_count is not None:
            if len(board.piece_map()) > self.max_piece_count:
                return None

        # Calcoliamo le mosse legali una sola volta.
        legal_moves = list(board.legal_moves)

        if len(legal_moves) < self.candidate_min_legal_moves:
            return None

        if (
            self.candidate_max_legal_moves is not None
            and len(legal_moves) > self.candidate_max_legal_moves
        ):
            return None

        # IMPORTANTE:
        # di default NON scartiamo le posizioni in check.
        # Una posizione in check può tranquillamente contenere
        # un mate 1, mate 2, ecc.
        if self.skip_if_in_check and board.is_check():
            return None

        return legal_moves

    # ================================================================
    # ECONOMIC MATE-POTENTIAL FILTER
    # ================================================================

    _PIECE_VALUES: Dict[int, int] = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    def _has_mating_material(
        self,
        board: chess.Board,
    ) -> bool:
        """
        Filtro economico (nessuna chiamata a Stockfish/Syzygy).

        Somma il valore standard dei pezzi del lato che deve muovere
        (re escluso). Un matto forzato in poche mosse richiede quasi
        sempre una soglia minima di forza d'attacco: re nudo o re+pedone
        da soli non producono quasi mai un mate 1-5 forzato contro un
        avversario che gioca le mosse legali migliori.

        Deliberatamente permissivo (soglia bassa di default) per non
        scartare mate reali: l'obiettivo e' eliminare solo i casi in
        cui un matto forzato e' materialmente impossibile o comunque
        statisticamente trascurabile, non fare una valutazione tattica.
        """
        mover = board.turn

        material = sum(
            self._PIECE_VALUES.get(piece.piece_type, 0)
            for piece in board.piece_map().values()
            if piece.color == mover
        )

        return material >= self.min_material_for_mate_attempt

    # ================================================================
    # SYZYGY SCREENING
    # ================================================================

    def _syzygy_says_no_mate(
        self,
        board: chess.Board,
    ) -> bool:
        """
        Screening istantaneo pre-Stockfish via tablebase Syzygy (se
        configurata e disponibile per la composizione di materiale
        corrente).

        Ritorna True SOLO quando siamo certi che un mate per il lato
        di turno e' impossibile (posizione patta o persa per chi deve
        muovere secondo WDL): in quel caso si scarta la posizione senza
        mai chiamare Stockfish.

        Ritorna False in ogni altro caso, incluso quando Syzygy non e'
        disponibile, non copre questa composizione di pezzi, o il probe
        fallisce: il fallback e' sempre "lascia decidere Stockfish",
        mai un falso scarto.
        """
        global _tablebase

        if _tablebase is None:
            return False

        # Syzygy non include posizioni con diritti di arrocco ancora
        # disponibili: il probe non e' valido in quel caso.
        if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
            return False

        try:
            wdl = _tablebase.probe_wdl(board)
        except (KeyError, chess.syzygy.MissingTableError):
            # Composizione non coperta dai file disponibili.
            return False
        except Exception:
            # Qualunque altro problema di probing: non blocchiamo mai
            # la pipeline per questo, si procede su Stockfish.
            return False

        # wdl e' relativo al lato che deve muovere: <= 0 significa
        # patta o persa, quindi nessun mate per lui e' possibile.
        return wdl is not None and wdl <= 0

    # ================================================================
    # STOCKFISH ANALYSIS
    # ================================================================

    def _analyse_position(
        self,
        board: chess.Board,
    ):
        """
        Analizza una posizione con Stockfish.

        Se analysis_time è specificato usa un limite temporale
        (default, piu' stabile su CPU eterogenee/cloud).
        Altrimenti usa la profondità configurata.

        La ricerca viene limitata al massimo mate del dataset.
        Per games_pipeline: mate 1-5.
        """
        global _engine

        if _engine is None:
            return None

        try:
            if self.analysis_time is not None:
                limit = chess.engine.Limit(
                    time=self.analysis_time,
                    mate=self.mate_range[1],
                )
            else:
                limit = chess.engine.Limit(
                    depth=self.search_depth,
                    mate=self.mate_range[1],
                )

            return _engine.analyse(
                board,
                limit,
                multipv=self.multipv,
            )

        except (
            chess.engine.EngineTerminatedError,
            chess.engine.EngineError,
            BrokenPipeError,
            OSError,
        ):
            return None

        except Exception:
            return None

    # ================================================================
    # WORKER
    # ================================================================

    def _worker(
        self,
        args: Tuple[int, str],
    ) -> Tuple[int, List[torch.Tensor]]:
        global _engine

        game_id, pgn_text = args

        if _engine is None:
            return game_id, []

        # ------------------------------------------------------------
        # PARSE PGN
        # ------------------------------------------------------------

        try:
            game = chess.pgn.read_game(
                io.StringIO(pgn_text)
            )
        except Exception:
            return game_id, []

        if game is None:
            return game_id, []

        # ------------------------------------------------------------
        # ECONOMIC GAME FILTER
        # ------------------------------------------------------------

        if not self._game_is_eligible(game):
            return game_id, []

        # ------------------------------------------------------------
        # TIME CONTROL
        # ------------------------------------------------------------

        time_control = game.headers.get(
            "TimeControl",
            "",
        )

        base_time, increment = (
            self._parse_time_control(
                time_control
            )
        )

        # ------------------------------------------------------------
        # INITIAL CLOCK
        # ------------------------------------------------------------

        previous_clock = {
            chess.WHITE: (
                base_time
                if base_time > 0
                else None
            ),
            chess.BLACK: (
                base_time
                if base_time > 0
                else None
            ),
        }

        # ------------------------------------------------------------
        # OUTPUT
        # ------------------------------------------------------------

        data_list: List[torch.Tensor] = []

        node = game

        mate_lo, mate_hi = self.mate_range

        positions_analysed = 0

        positions_filtered = 0

        # ------------------------------------------------------------
        # MAIN LOOP
        # ------------------------------------------------------------

        while node.variations:
            next_node = node.variation(0)

            board = node.board()

            comment = next_node.comment or ""

            mover_color = board.turn

            # --------------------------------------------------------
            # CLOCK
            # --------------------------------------------------------

            current_clock = self._parse_clock(
                comment
            )

            move_duration = (
                self._compute_move_duration(
                    previous_clock[mover_color],
                    current_clock,
                    increment,
                )
            )

            if current_clock is not None:
                previous_clock[mover_color] = (
                    current_clock
                )

            # --------------------------------------------------------
            # MIN PLY
            # --------------------------------------------------------

            if node.ply() < self.min_ply:
                node = next_node
                continue

            # --------------------------------------------------------
            # PLY SAMPLING
            # --------------------------------------------------------
            # Analizza solo 1 posizione ogni ply_sample_step, invece di
            # ogni singola mezza-mossa. Riduce direttamente il numero di
            # chiamate Stockfish (il vero collo di bottiglia), mantenendo
            # comunque buona copertura tattica per mate 1-5.
            if (node.ply() - self.min_ply) % self.ply_sample_step != 0:
                node = next_node
                continue

            # --------------------------------------------------------
            # CLOCK REQUIRED
            # --------------------------------------------------------

            if (
                self.require_clock
                and current_clock is None
            ):
                node = next_node
                continue

            # --------------------------------------------------------
            # MAX POSITIONS PER GAME
            # --------------------------------------------------------

            if (
                self.max_positions_per_game is not None
                and positions_analysed
                >= self.max_positions_per_game
            ):
                break

            # --------------------------------------------------------
            # PRE-STOCKFISH POSITION FILTER
            # --------------------------------------------------------

            legal_moves = self._get_candidate_legal_moves(
                board
            )

            if legal_moves is None:
                positions_filtered += 1
                node = next_node
                continue

            # --------------------------------------------------------
            # MATE-POTENTIAL FILTER (economico, no engine)
            # --------------------------------------------------------
            # Scarta posizioni dove il lato di turno non ha abbastanza
            # materiale per un matto forzato plausibile in mate_hi mosse.
            if not self._has_mating_material(board):
                positions_filtered += 1
                node = next_node
                continue

            # --------------------------------------------------------
            # SYZYGY SCREENING (istantaneo, prima di Stockfish)
            # --------------------------------------------------------
            # Se la tablebase e' disponibile e certifica che la
            # posizione e' patta o persa per chi muove, un mate per
            # lui e' impossibile: si evita la chiamata a Stockfish.
            if self._syzygy_says_no_mate(board):
                positions_filtered += 1
                node = next_node
                continue

            # --------------------------------------------------------
            # STOCKFISH
            # --------------------------------------------------------

            info = self._analyse_position(
                board
            )

            positions_analysed += 1

            if not info:
                node = next_node
                continue

            # --------------------------------------------------------
            # BEST LINE
            # --------------------------------------------------------

            best_info = info[0]

            score = best_info.get("score")

            if score is None:
                node = next_node
                continue

            relative_score = score.relative

            # --------------------------------------------------------
            # MATE?
            # --------------------------------------------------------

            if not relative_score.is_mate():
                node = next_node
                continue

            mate_n = relative_score.mate()

            if mate_n is None:
                node = next_node
                continue

            # --------------------------------------------------------
            # RANGE
            # --------------------------------------------------------

            if not (
                mate_n > 0
                and mate_lo <= mate_n <= mate_hi
            ):
                node = next_node
                continue

            # --------------------------------------------------------
            # PV
            # --------------------------------------------------------

            pv = best_info.get("pv")

            if not pv:
                node = next_node
                continue

            best_move = pv[0]

            # --------------------------------------------------------
            # BEST MOVE INDEX
            # --------------------------------------------------------

            try:
                best_move_idx = legal_moves.index(
                    best_move
                )
            except ValueError:
                node = next_node
                continue

            # --------------------------------------------------------
            # CLOCK FEATURE
            # --------------------------------------------------------

            clock_seconds = (
                move_duration
                if move_duration is not None
                else self.default_move_seconds
            )

            # --------------------------------------------------------
            # LABEL
            # --------------------------------------------------------

            label = {
                "mate_n": int(mate_n),
                "best_move_idx": int(best_move_idx),
            }

            # --------------------------------------------------------
            # GRAPH
            # --------------------------------------------------------

            try:
                data = (
                    GraphBuilder.board_to_pyg_data(
                        board,
                        clock_seconds=clock_seconds,
                        label=label,
                        legal_moves=legal_moves,
                    )
                )
            except Exception:
                node = next_node
                continue

            # --------------------------------------------------------
            # METADATA
            # --------------------------------------------------------

            data.game_id = int(game_id)

            data.ply = int(
                node.ply()
            )

            data.clock_seconds = float(
                clock_seconds
            )

            data.mate_n = int(
                mate_n
            )

            data.best_move_idx = int(
                best_move_idx
            )

            data_list.append(data)

            node = next_node

        return game_id, data_list

    # ================================================================
    # PGN STREAM
    # ================================================================

    def _stream_pgn_texts(
        self,
    ) -> Generator[
        Tuple[int, str],
        None,
        None,
    ]:
        """
        Legge il file PGN.zst in streaming senza caricarlo
        interamente nella RAM, saltando le prime skip_games partite.
        """

        decompressor = zstd.ZstdDecompressor()

        with open(
            self.zst_path,
            "rb",
        ) as compressed_file:

            with decompressor.stream_reader(
                compressed_file
            ) as reader:

                text_stream = io.TextIOWrapper(
                    reader,
                    encoding="utf-8",
                )

                game_id = 0
                yielded_games = 0

                current_game: List[str] = []

                for line in text_stream:

                    # Nuova partita.
                    if (
                        line.startswith("[Event ")
                        and current_game
                    ):
                        game_id += 1

                        if game_id > self.skip_games:
                            yield (
                                game_id,
                                "".join(current_game),
                            )
                            yielded_games += 1

                            if (
                                self.max_games is not None
                                and yielded_games >= self.max_games
                            ):
                                return

                        current_game = [line]

                    else:
                        current_game.append(line)

                # Ultima partita.
                if current_game:
                    game_id += 1
                    
                    if game_id > self.skip_games:
                        if (
                            self.max_games is None
                            or yielded_games < self.max_games
                        ):
                            yield (
                                game_id,
                                "".join(current_game),
                            )

    # ================================================================
    # SPLIT
    # ================================================================

    def _assign_game_split(
        self,
        game_id: int,
    ) -> str:
        """
        Split deterministico a livello di partita.

        Evita data leakage tra posizioni della stessa partita.
        """

        rng = random.Random(
            self.seed + game_id
        )

        value = rng.random()

        train_ratio, val_ratio, _ = (
            self.split_ratios
        )

        if value < train_ratio:
            return "train"

        if value < train_ratio + val_ratio:
            return "val"

        return "test"

    # ================================================================
    # RUN
    # ================================================================

    def run(
        self,
    ) -> Tuple[
        Dict[str, List[torch.Tensor]],
        Dict[str, str],
    ]:
        """
        Esegue l'intera pipeline.
        """

        # ------------------------------------------------------------
        # OUTPUT DIRECTORY
        # ------------------------------------------------------------

        output_directory = os.path.dirname(
            self.output_pt
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        # ------------------------------------------------------------
        # DATA
        # ------------------------------------------------------------

        split_data: Dict[
            str,
            List[torch.Tensor],
        ] = {
            "train": [],
            "val": [],
            "test": [],
        }

        # ------------------------------------------------------------
        # POOL
        # ------------------------------------------------------------

        pool = mp.Pool(
            processes=self.workers,
            initializer=self._init_worker,
            initargs=(
                self.stockfish_path,
                self.threads,
                self.hash_mb,
                self.syzygy_path,
            ),
        )

        processed_games = 0
        accepted_games = 0
        generated_positions = 0

        try:
            game_stream = (
                self._stream_pgn_texts()
            )

            results = pool.imap_unordered(
                self._worker,
                game_stream,
                chunksize=1,
            )

            for game_id, data_list in tqdm(
                results,
                desc="Analisi Partite Stockfish",
                total=self.max_games,
                dynamic_ncols=True,
            ):
                processed_games += 1

                if not data_list:
                    continue

                accepted_games += 1

                split_name = (
                    self._assign_game_split(
                        game_id
                    )
                )

                split_data[
                    split_name
                ].extend(
                    data_list
                )

                generated_positions += (
                    len(data_list)
                )

        finally:
            # ----------------------------------------------------------
            # CHIUSURA POOL (fix del deadlock)
            # ----------------------------------------------------------
            # CRITICO: pool.close() smette di accettare nuovi task ma NON
            # chiude i processi worker ne' i loro sottoprocessi Stockfish.
            # Ogni worker termina solo quando il processo Python esce
            # naturalmente. Se Stockfish non ha mai ricevuto 'quit', resta
            # appeso in lettura sulla pipe e il processo worker non muore
            # mai: pool.join() si blocca indefinitamente, PRIMA del
            # torch.save finale, con perdita totale del lavoro svolto
            # (bug osservato in produzione: 100% dei task completati,
            # processo comunque bloccato per ore in pool.join()).
            #
            # _close_engine() e' registrata con atexit in _init_worker:
            # quando ogni processo worker esce (perche' il Pool lo
            # termina), l'handler atexit prova a chiudere Stockfish
            # pulitamente PRIMA che il processo Python muoia del tutto.
            # Non e' pero' garanzia assoluta (un worker puo' restare
            # comunque appeso in casi patologici, es. Stockfish non
            # risponde nemmeno a 'quit'), quindi affianchiamo un timeout
            # deterministico: se dopo pool_join_timeout secondi qualche
            # worker e' ancora vivo, forziamo pool.terminate() (SIGTERM
            # ai processi) invece di restare bloccati indefinitamente.
            pool.close()

            if self.pool_join_timeout is not None:
                import time

                deadline = time.monotonic() + self.pool_join_timeout

                # IMPORTANTE: il timeout deve essere un budget TOTALE
                # condiviso tra tutti i worker, non un timeout pieno per
                # ciascuno. La versione precedente chiamava
                # proc.join(timeout=self.pool_join_timeout) in sequenza
                # per ogni processo: con N worker, il caso peggiore
                # diventava N * pool_join_timeout secondi invece di
                # pool_join_timeout secondi totali (osservato: con 8
                # worker e timeout=30s, fino a 4 minuti di attesa reale
                # prima che scattasse pool.terminate()).
                for proc in pool._pool:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    proc.join(timeout=remaining)

                still_alive = [
                    proc for proc in pool._pool if proc.is_alive()
                ]

                if still_alive:
                    stale_pids = [proc.pid for proc in still_alive]

                    print(
                        f"\n[WARNING] {len(still_alive)} worker non "
                        f"terminati entro {self.pool_join_timeout}s "
                        f"dopo pool.close() (Stockfish non ha chiuso "
                        f"correttamente). Forzo pool.terminate() per "
                        f"evitare un blocco indefinito prima del "
                        f"salvataggio."
                    )
                    pool.terminate()

                    # pool.terminate() manda SIGTERM ai processi worker
                    # Python, ma non garantisce che il loro sottoprocesso
                    # Stockfish (figlio del worker, non del Pool) venga
                    # ucciso a sua volta: puo' restare come processo
                    # orfano che continua a occupare CPU/RAM inutilmente.
                    # Best-effort: proviamo a terminarli via psutil se
                    # disponibile, altrimenti segnaliamo il da farsi.
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
                            "[WARNING] Modulo 'psutil' non disponibile: "
                            "impossibile terminare automaticamente eventuali "
                            "processi Stockfish orfani residui. Verificare "
                            "manualmente con 'ps aux | grep stockfish'."
                        )

            pool.join()

        # ------------------------------------------------------------
        # SAVE
        # ------------------------------------------------------------

        base_path, extension = (
            os.path.splitext(
                self.output_pt
            )
        )

        if not extension:
            extension = ".pt"

        paths: Dict[str, str] = {}

        for (
            split_name,
            data_list,
        ) in split_data.items():

            output_path = (
                f"{base_path}_"
                f"{split_name}"
                f"{extension}"
            )

            torch.save(
                data_list,
                output_path
            )

            paths[
                split_name
            ] = output_path

            print(
                f"\nSalvato {split_name}: "
                f"{len(data_list):,} posizioni"
            )

        # ------------------------------------------------------------
        # SUMMARY
        # ------------------------------------------------------------

        total_positions = sum(
            len(items)
            for items in split_data.values()
        )

        print("\n" + "=" * 60)
        print("DATASET GENERATO")
        print("=" * 60)

        print(
            f"Partite processate: "
            f"{processed_games:,}"
        )

        print(
            f"Partite con posizioni mate: "
            f"{accepted_games:,}"
        )

        print(
            f"Train: "
            f"{len(split_data['train']):,}"
        )

        print(
            f"Validation: "
            f"{len(split_data['val']):,}"
        )

        print(
            f"Test: "
            f"{len(split_data['test']):,}"
        )

        print(f"Totale posizioni:{total_positions:,}")

        print("=" * 60)

        return split_data, paths
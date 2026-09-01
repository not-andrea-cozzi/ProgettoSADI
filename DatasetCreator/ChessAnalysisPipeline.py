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

from DatasetCreator.GraphBuilder import GraphBuilder

# Evita problemi di sharing dei tensori tra processi PyTorch.
torch_mp.set_sharing_strategy("file_system")

# Ogni worker possiede la propria istanza Stockfish.
_engine: Optional[chess.engine.SimpleEngine] = None

# Esempi supportati:
# [ %clk 0:05:32.4 ]
# [ %clk 1:23:45 ]
_CLK_RE = re.compile(r"\[\s*%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*\]")


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
        search_depth: int = 10,
        workers: Optional[int] = None,
        threads: int = 1,
        hash_mb: int = 128,
        multipv: int = 1,
        max_games: Optional[int] = None,
        seed: int = 42,
        default_move_seconds: float = 15.0,
        require_clock: bool = False,
        min_ply: int = 16,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        analysis_time: Optional[float] = None,
        min_game_plies: int = 20,
        skip_no_time_control: bool = False,
        max_positions_per_game: Optional[int] = None,
        candidate_min_legal_moves: int = 1,
        candidate_max_legal_moves: Optional[int] = None,
        skip_if_in_check: bool = False,
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

        if len(self.split_ratios) != 3:
            raise ValueError(
                "split_ratios deve contenere train, val e test."
            )

        if abs(sum(self.split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios deve sommare a 1.0.")

    # ================================================================
    # STOCKFISH INITIALIZATION
    # ================================================================

    @staticmethod
    def _init_worker(
        stockfish_path: str,
        threads: int,
        hash_mb: int,
    ) -> None:
        """
        Crea una singola istanza Stockfish per worker.
        """
        global _engine

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
    # STOCKFISH ANALYSIS
    # ================================================================

    def _analyse_position(
        self,
        board: chess.Board,
    ):
        """
        Analizza una posizione con Stockfish.

        Se analysis_time è specificato usa un limite temporale.
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
        interamente nella RAM.
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

                current_game: List[str] = []

                for line in text_stream:

                    # Nuova partita.
                    if (
                        line.startswith("[Event ")
                        and current_game
                    ):
                        game_id += 1

                        yield (
                            game_id,
                            "".join(current_game),
                        )

                        if (
                            self.max_games is not None
                            and game_id >= self.max_games
                        ):
                            return

                        current_game = [line]

                    else:
                        current_game.append(line)

                # Ultima partita.
                if current_game:
                    if (
                        self.max_games is None
                        or game_id < self.max_games
                    ):
                        game_id += 1

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
            pool.close()
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

        print(
            f"Totale posizioni: "
            f"{total_positions:,}"
        )

        print("=" * 60)

        return split_data, paths
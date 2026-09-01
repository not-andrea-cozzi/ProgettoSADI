import io
import os
import re
import random
import multiprocessing as mp
from DatasetCreator.GraphBuilder import GraphBuilder
import chess
import chess.pgn
import chess.engine
import zstandard as zstd
import torch
from tqdm import tqdm
import torch.multiprocessing as torch_mp
torch_mp.set_sharing_strategy('file_system')

_engine = None
_CLK_RE = re.compile(r'\[%clk (\d+):(\d+):(\d+)\]')

class ChessAnalysisPipeline:
    def __init__(self, zst_path, stockfish_path, output_pt,
                 mate_range=(1, 5), time_limit=0.2, multipv=3,
                 workers=None, max_games=None, seed=42, 
                 default_move_seconds=15.0, require_eval_comment=True):
        self.zst_path = zst_path
        self.stockfish_path = stockfish_path
        self.output_pt = output_pt
        self.mate_range = mate_range
        self.time_limit = time_limit
        self.multipv = multipv
        self.workers = workers or 5
        self.max_games = max_games
        self.seed = seed
        self.default_move_seconds = default_move_seconds
        self.require_eval_comment = require_eval_comment

    @staticmethod
    def _init_worker(stockfish_path):
        global _engine
        _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        _engine.configure({"Threads": 1, "Hash": 64})

    @staticmethod
    def _parse_clock(comment: str) -> float | None:
        m = _CLK_RE.search(comment or "")
        if not m:
            return None
        h, mi, s = map(int, m.groups())
        return float(h * 3600 + mi * 60 + s)

    @staticmethod
    def _parse_increment(time_control: str) -> float:
        if not time_control or time_control == "-":
            return 0.0
        m = re.match(r'(\d+)\+(\d+)', time_control)
        return float(m.group(2)) if m else 0.0

    def _worker(self, args):
        game_id, pgn_text = args
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return game_id, []

        increment = self._parse_increment(game.headers.get("TimeControl", ""))
        data_list = []
        node = game
        lo, hi = self.mate_range

        prev_clock = {chess.WHITE: None, chess.BLACK: None}

        while node.variations:
            nxt = node.variation(0)
            comment = nxt.comment or ""
            mover_color = node.board().turn

            clk = self._parse_clock(comment)
            move_duration = None
            if clk is not None:
                prev = prev_clock[mover_color]
                if prev is not None:
                    spent = prev - clk + increment
                    if spent > 0:
                        move_duration = spent
                prev_clock[mover_color] = clk

            if self.require_eval_comment and "#" not in comment:
                node = nxt
                continue

            board = node.board()
            try:
                info = _engine.analyse(board, chess.engine.Limit(time=self.time_limit, mate=hi), multipv=self.multipv)
            except Exception:
                node = nxt
                continue

            if info and info[0].get("score") and info[0]["score"].relative.is_mate():
                mate_n = info[0]["score"].relative.mate()
                if mate_n > 0 and lo <= mate_n <= hi:
                    legal = list(board.legal_moves)
                    
                    try:
                        best_move = info[0]["pv"][0]
                        best_idx = legal.index(best_move)
                    except (ValueError, IndexError, KeyError):
                        node = nxt
                        continue

                    clock = move_duration if move_duration is not None else self.default_move_seconds
                    label = {"mate_n": mate_n, "best_move_idx": best_idx}
                    d = GraphBuilder.board_to_pyg_data(board, clock_seconds=clock, label=label)
                    d.game_id = game_id
                    data_list.append(d)

            node = nxt

        return game_id, data_list

    def _stream_pgn_texts(self):
        dctx = zstd.ZstdDecompressor()
        with open(self.zst_path, "rb") as f, dctx.stream_reader(f) as r:
            text = io.TextIOWrapper(r, encoding="utf-8")
            gid = 0
            while True:
                if self.max_games and gid >= self.max_games:
                    break
                g = chess.pgn.read_game(text)
                if g is None:
                    break
                gid += 1
                yield gid, str(g)

    def _assign_game_split(self, game_id: int) -> str:
        rng = random.Random(self.seed + game_id)
        r = rng.random()
        if r < 0.8:
            return "train"
        elif r < 0.9:
            return "val"
        return "test"

    def run(self):
        out_dir = os.path.dirname(self.output_pt)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        split_data = {"train": [], "val": [], "test": []}
        with mp.Pool(self.workers, initializer=self._init_worker, initargs=(self.stockfish_path,)) as pool:
            gen = self._stream_pgn_texts()
            for game_id, data_list in tqdm(pool.imap(self._worker, gen, chunksize=20),
                                            desc="Scansione", total=self.max_games):
                if not data_list:
                    continue
                split_name = self._assign_game_split(game_id)
                split_data[split_name].extend(data_list)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        base, ext = os.path.splitext(self.output_pt)
        paths = {}
        for name, dlist in split_data.items():
            path = f"{base}_{name}{ext}"
            torch.save(dlist, path)
            paths[name] = path

        return split_data, paths
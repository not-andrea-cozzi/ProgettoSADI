import io
import os
import multiprocessing as mp
from Model.graph_builder import GraphBuilder
import chess
import chess.pgn
import chess.engine
import zstandard as zstd
import torch
from tqdm import tqdm

_engine = None


class ChessAnalysisPipeline:
    """Estrae posizioni di matto (mate in 1-5) da partite Lichess (.pgn.zst)
    e produce Data PyG salvati in .pt, unificabile con PuzzleGraphDataset."""

    def __init__(self, zst_path, stockfish_path, output_pt,
                 mate_range=(1, 5), time_limit=0.2, multipv=3,
                 workers=None, max_games=None):
        self.zst_path = zst_path
        self.stockfish_path = stockfish_path
        self.output_pt = output_pt
        self.mate_range = mate_range
        self.time_limit = time_limit
        self.multipv = multipv
        self.workers = 5
        self.max_games = max_games

    @staticmethod
    def _init_worker(stockfish_path):
        global _engine
        _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        _engine.configure({"Threads": 1, "Hash": 64})

    def _worker(self, args):
        game_id, pgn_text = args
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return []

        data_list = []
        node = game
        lo, hi = self.mate_range

        while node.variations:
            nxt = node.variation(0)
            comment = nxt.comment or ""
            if "#" not in comment:
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
                if lo <= abs(mate_n) <= hi:
                    legal = list(board.legal_moves)
                    try:
                        best_idx = legal.index(nxt.move)
                    except ValueError:
                        node = nxt
                        continue

                    clock = self._extract_clock(comment) or 15.0
                    label = {"mate_n": mate_n, "best_move_idx": best_idx}
                    d = GraphBuilder.board_to_pyg_data(board, clock_seconds=clock, label=label)
                    d.game_id = game_id
                    data_list.append(d)

            node = nxt

        return data_list

    @staticmethod
    def _extract_clock(comment: str) -> float | None:
        import re
        match = re.search(r'\[%clk (.*?)\]', comment)
        if not match:
            return None
        parts = match.group(1).split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return float(h * 3600 + m * 60 + s)
        if len(parts) == 2:
            m, s = map(int, parts)
            return float(m * 60 + s)
        return None

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

    def run(self):
        # Verifica ANCHE prima di iniziare la scansione (che puo' richiedere decine di minuti),
        # cosi' l'errore per path mancante esce subito e non dopo aver buttato via il lavoro.
        out_dir = os.path.dirname(self.output_pt)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        all_data = []
        with mp.Pool(self.workers, initializer=self._init_worker, initargs=(self.stockfish_path,)) as pool:
            gen = self._stream_pgn_texts()
            for data_list in tqdm(pool.imap(self._worker, gen, chunksize=20),
                                   desc="Scansione", total=self.max_games):
                all_data.extend(data_list)

        # Doppio check: se la cartella e' stata rimossa nel frattempo, non perdere comunque il lavoro.
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save(all_data, self.output_pt)
        return all_data
import io
import os
import random
import multiprocessing as mp
from Model.graph_builder import GraphBuilder
import chess
import chess.pgn
import chess.engine
import zstandard as zstd
import torch
from tqdm import tqdm
import torch.multiprocessing as torch_mp
torch_mp.set_sharing_strategy('file_system')


_engine = None


class ChessAnalysisPipeline:
    """Estrae posizioni di matto (mate in 1-5) da partite Lichess (.pgn.zst)
    e produce Data PyG salvati in .pt, unificabile con PuzzleGraphDataset.

    FIX split-leakage: lo split train/val/test avviene sui game_id (una partita
    intera), non sulle singole posizioni-mossa estratte. Tutte le posizioni di
    matto trovate nella stessa partita finiscono nello stesso split. run() produce
    3 file separati (output_pt_train / _val / _test) pronti per merge_and_split,
    che a quel punto NON deve piu' ri-splittare nulla."""

    def __init__(self, zst_path, stockfish_path, output_pt,
                 mate_range=(1, 5), time_limit=0.2, multipv=3,
                 workers=None, max_games=None, seed=42):
        self.zst_path = zst_path
        self.stockfish_path = stockfish_path
        self.output_pt = output_pt
        self.mate_range = mate_range
        self.time_limit = time_limit
        self.multipv = multipv
        self.workers = 5
        self.max_games = max_games
        self.seed = seed

    @staticmethod
    def _init_worker(stockfish_path):
        global _engine
        _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        _engine.configure({"Threads": 1, "Hash": 64})

    def _worker(self, args):
        game_id, pgn_text = args
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return game_id, []

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
                if mate_n > 0 and lo <= mate_n <= hi:
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

        return game_id, data_list

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

    def _assign_game_split(self, game_id: int) -> str:
        """Split deterministico per game_id (80/10/10), indipendente dall'ordine
        di arrivo dai worker paralleli: stesso game_id -> sempre stesso split."""
        rng = random.Random(self.seed + game_id)
        r = rng.random()
        if r < 0.8:
            return "train"
        elif r < 0.9:
            return "val"
        return "test"

    def run(self):
        # Verifica ANCHE prima di iniziare la scansione (che puo' richiedere decine di minuti),
        # cosi' l'errore per path mancante esce subito e non dopo aver buttato via il lavoro.
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

        # Doppio check: se la cartella e' stata rimossa nel frattempo, non perdere comunque il lavoro.
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        base, ext = os.path.splitext(self.output_pt)
        paths = {}
        for name, dlist in split_data.items():
            path = f"{base}_{name}{ext}"
            torch.save(dlist, path)
            paths[name] = path

        return split_data, paths
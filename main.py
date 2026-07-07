import io, csv, multiprocessing as mp
import chess, chess.pgn, chess.engine
from tqdm import tqdm

STOCKFISH = "/usr/games/stockfish"
_engine = None

def init_worker():
    global _engine
    _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    _engine.configure({"Threads": 1, "Hash": 64})

def worker(args):
    game_id, pgn_text = args
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return []
    rows, node = [], game
    while node.variations:
        nxt = node.variation(0)
        comment = nxt.comment or ""
        if "#" not in comment:
            node = nxt
            continue
        board = node.board()
        try:
            info = _engine.analyse(board, chess.engine.Limit(time=0.2, mate=5), multipv=3)
        except Exception:
            node = nxt; continue
        if info and info[0].get("score") and info[0]["score"].relative.is_mate():
            mate_n = info[0]["score"].relative.mate()
            if 1 <= abs(mate_n) <= 5:
                best = info[0]["pv"][0].uci() if info[0].get("pv") else None
                alts = [i["pv"][0].uci() for i in info[1:] if i.get("pv")]
                rows.append([game_id, game.headers.get("White"), game.headers.get("Black"),
                             game.headers.get("Result"), board.fullmove_number, board.turn,
                             nxt.san(), nxt.move.uci(), comment, best, mate_n, ",".join(alts)])
        node = nxt
    return rows

def stream_pgn_texts(path):
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f, dctx.stream_reader(f) as r:
        text = io.TextIOWrapper(r, encoding="utf-8")
        gid = 0
        while True:
            g = chess.pgn.read_game(text)
            if g is None: break
            gid += 1
            yield gid, str(g)

if __name__ == "__main__":
    headers = ["Game_ID","White","Black","Result","Turno","Colore",
               "Mossa_SAN","Mossa_UCI","Lichess_Eval","SF_Best","SF_Mate","SF_Alts"]
    
    with open("dataset/analisi_partite.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(headers)
        with mp.Pool(mp.cpu_count(), initializer=init_worker) as pool:
            gen = stream_pgn_texts("lichess_db_standard_rated_2013-01.pgn.zst")
            for rows in tqdm(pool.imap(worker, gen, chunksize=20), desc="Scansione"):
                if rows: w.writerows(rows)
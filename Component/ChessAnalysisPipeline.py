import io
import csv
import json
import multiprocessing as mp
import chess
import chess.pgn
import chess.engine
import zstandard as zstd
import networkx as nx
from networkx.readwrite import json_graph
from tqdm import tqdm

_engine = None

def board_to_graph_json(board: chess.Board) -> str:
    """
    Ottimizza la conversione della scacchiera in grafo e la serializza in JSON.
    Usa piece_map() per evitare di iterare inutilmente su case vuote.
    """
    G = nx.MultiDiGraph()
    piece_map = board.piece_map()  # Dizionario {square: piece} super veloce

    # 1. Creazione Nodi (64 case)
    for sq in chess.SQUARES:
        piece = piece_map.get(sq)
        if piece:
            G.add_node(sq, has_piece=1, piece_type=piece.piece_type, color=int(piece.color))
        else:
            G.add_node(sq, has_piece=0, piece_type=0, color=-1)

    # 2. Archi: Mosse Legali
    for move in board.legal_moves:
        G.add_edge(move.from_square, move.to_square, edge_type='legal_move')

    # 3. Archi: Attacchi & Inchiodature (calcolati solo sui pezzi reali)
    for sq, piece in piece_map.items():
        # Attacchi
        for target_sq in board.attacks(sq):
            G.add_edge(sq, target_sq, edge_type='attack')

        # Inchiodature (Pins)
        if board.is_pinned(piece.color, sq):
            pin_ray_squares = chess.SquareSet(board.pin(piece.color, sq))
            for ray_sq in pin_ray_squares:
                attacker = piece_map.get(ray_sq)
                if attacker and attacker.color != piece.color:
                    if attacker.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                        G.add_edge(ray_sq, sq, edge_type='pin')

    # Serializza immediatamente in JSON
    return json.dumps(json_graph.node_link_data(G))


class ChessAnalysisPipeline:
    def __init__(self, zst_path, stockfish_path, output_csv,
                 mate_range=(1, 5), time_limit=0.2, multipv=3, 
                 workers=None, max_games=None):
        
        self.zst_path = zst_path
        self.stockfish_path = stockfish_path
        self.output_csv = output_csv
        self.mate_range = mate_range
        self.time_limit = time_limit
        self.multipv = multipv
        self.workers = workers or mp.cpu_count()
        self.max_games = max_games 
        
        # Aggiunta la colonna "Graph_JSON" alla fine delle intestazioni
        self.headers = [
            "Game_ID", "White", "Black", "Result", "Turno", "Colore",
            "Mossa_SAN", "Mossa_UCI", "Lichess_Eval", "SF_Best", 
            "SF_Mate", "SF_Alts", "Graph_JSON"
        ]

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
            
        rows = []
        node = game
        lo, hi = self.mate_range
        
        while node.variations:
            nxt = node.variation(0)
            comment = nxt.comment or ""
            
            # Filtro rapido: analizza solo se la mossa ha un commento di mate di Lichess
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
                    best = info[0]["pv"][0].uci() if info[0].get("pv") else None
                    alts = [i["pv"][0].uci() for i in info[1:] if i.get("pv")]
                    
                    # Generazione del grafo per la posizione vincente
                    graph_json_str = board_to_graph_json(board)
                    
                    rows.append([
                        game_id, 
                        game.headers.get("White"), 
                        game.headers.get("Black"),
                        game.headers.get("Result"), 
                        board.fullmove_number, 
                        board.turn,
                        nxt.san(), 
                        nxt.move.uci(), 
                        comment, 
                        best, 
                        mate_n, 
                        ",".join(alts),
                        graph_json_str  # Inserimento del grafo nel CSV
                    ])
                    
            node = nxt
            
        return rows

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
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.headers)
            
            with mp.Pool(self.workers, initializer=self._init_worker, initargs=(self.stockfish_path,)) as pool:
                gen = self._stream_pgn_texts()
                
                for rows in tqdm(pool.imap(self._worker, gen, chunksize=20), 
                                 desc="Scansione", 
                                 total=self.max_games):
                    if rows:
                        w.writerows(rows)
                        f.flush()  
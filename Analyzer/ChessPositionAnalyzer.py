import chess
import chess.engine
from Model.ChessMove import ChessMove

class ChessPositionAnalyzer:
    def __init__(self):
        pass

    def enrich_move_with_stockfish(self, node, chess_move_obj: ChessMove, engine):
        """Analizza la posizione prima della mossa cercando un matto in 5 mosse con 3 varianti"""
        board_before_move = node.parent.board()
        
        analysis = engine.analyse(
            board_before_move, 
            chess.engine.Limit(time=0.2, mate=5), 
            multipv=3
        )
        
        if analysis:
            # 1. Variante Principale (Best Move)
            if "pv" in analysis[0] and len(analysis[0]["pv"]) > 0:
                chess_move_obj.stockfish_best_move = analysis[0]["pv"][0].uci()
            
            # 2. Estrazione del Mate N (se presente)
            score = analysis[0].get("score")
            if score and score.relative.is_mate():
                chess_move_obj.stockfish_mate_n = score.relative.mate()
            
            # 3. Mosse Alternative (Varianti 2 e 3 del MultiPV)
            chess_move_obj.stockfish_alt_moves = []
            for alt_info in analysis[1:]:
                if "pv" in alt_info and len(alt_info["pv"]) > 0:
                    chess_move_obj.stockfish_alt_moves.append(alt_info["pv"][0].uci())
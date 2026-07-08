from enum import Enum
import re
import chess
import chess.pgn
import chess.engine


class ColorePedina(Enum):
    BIANCA = 0
    NERA = 1

class TipoPedina(Enum):
    PEDONE = 0
    ALFIERE = 1
    CAVALLO = 2
    REGINA = 3
    RE = 4
    TORRE = 5

class ChessMove:
    """Rappresenta una singola mossa di un game con tutti i suoi metadati"""
    def __init__(self, node):
        self.move = node.move
        self.uci = node.move.uci()  
        self.san = node.san()
        
        self.turn: int = (node.ply() + 1) // 2
        self.colore = ColorePedina.BIANCA if node.parent.turn() == chess.WHITE else ColorePedina.NERA

        scacchiera_prima = node.parent.board()
        pezzo_python_chess = scacchiera_prima.piece_type_at(node.move.from_square)
        mappatura_pezzi = {
            chess.PAWN: TipoPedina.PEDONE, chess.BISHOP: TipoPedina.ALFIERE,
            chess.KNIGHT: TipoPedina.CAVALLO, chess.QUEEN: TipoPedina.REGINA,
            chess.KING: TipoPedina.RE, chess.ROOK: TipoPedina.TORRE
        }
        self.tipo_pedina = mappatura_pezzi.get(pezzo_python_chess)

        self.lichess_eval: float | str | None = self._extract_eval(node.comment)
        self.lichess_clock: float | None = self._extract_clock(node.comment)

        self.stockfish_best_move: str | None = None
        self.stockfish_mate_n: int | None = None
        self.stockfish_alt_moves: list[str] = []

    def _extract_eval(self, comment: str) -> float | str | None:
        if not comment: return None
        match = re.search(r'\[%eval (.*?)\]', comment)
        if match:
            val = match.group(1)
            return val if val.startswith('#') else float(val)
        return None

    def _extract_clock(self, comment: str) -> float | None:
        if not comment: return None
        match = re.search(r'\[%clk (.*?)\]', comment)
        if match:
            parts = match.group(1).split(':')
            if len(parts) == 3:  
                h, m, s = map(int, parts)
                return float(h * 3600 + m * 60 + s)
            elif len(parts) == 2: 
                m, s = map(int, parts)
                return float(m * 60 + s)
        return None
import math
import torch
import chess
from torch_geometric.data import Data

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_TYPE_IDX = {pt: i + 1 for i, pt in enumerate(PIECE_TYPES)}


class GraphBuilder:
    """Costruisce oggetti torch_geometric.data.Data a partire da una chess.Board.

    Nodi: 64 caselle, feature = [has_piece, piece_type(0-6), color(-1/0/1), clock_norm,
                               turn, is_check, castle_wk, castle_wq, castle_bk, castle_bq]
    Archi: legal_move, attack, pin

    NB sul campo "color" (colonna x[:,2]): e' il valore RAW, non un indice di embedding.
    Convenzione: -1.0 = nessun pezzo sulla casella, 0.0 = pezzo nero (chess.BLACK == False == 0),
    1.0 = pezzo bianco (chess.WHITE == True == 1). PolicyGNN.InputEncoder rimappa questo raw
    in un indice 0/1/2 SOLO internamente (color_idx = color_raw + 1) per poter indicizzare
    l'embedding; il dato salvato in Data.x resta sempre -1/0/1. Se in futuro si legge x[:,2]
    altrove (es. dashboard/debug), va interpretato con QUESTA convenzione raw, non come indice.
    """

    CLOCK_CAP_SECONDS = 600.0  # oltre questa soglia clock_norm satura a 1.0 (log-scale)

    @staticmethod
    def _clock_norm(clock_seconds: float) -> float:
        """Normalizzazione log-scale: preserva la differenza tra clock brevi (bullet/blitz)
        senza schiacciare tutto cio' che supera pochi minuti come farebbe una scala lineare."""
        cap = GraphBuilder.CLOCK_CAP_SECONDS
        denom = math.log1p(cap)
        if denom <= 0:
            # difensivo: evita ZeroDivisionError se CLOCK_CAP_SECONDS venisse mai
            # impostato a 0 o negativo da chi estende la classe.
            return 0.0
        return min(math.log1p(max(clock_seconds, 0.0)) / denom, 1.0)

    @staticmethod
    def board_to_pyg_data(board: chess.Board, clock_seconds: float = 0.0, label: dict | None = None,
                           legal_moves: list | None = None) -> Data:
        """Converte board in torch_geometric.data.Data.

        clock_seconds: tempo da codificare nel nodo. IMPORTANTE — deve essere il tempo
                       SPESO sulla mossa (durata), non il tempo residuo sull'orologio,
                       come richiesto dalla spec ("move durations from real games").
                       Il calcolo residuo->durata va fatto dal chiamante PRIMA di invocare
                       questo metodo (vedi Component/ChessAnalysisPipeline._time_spent_for_move
                       e Component/TimeStatBuilder, che gia' fanno questa conversione).

        legal_moves: se già calcolate dal caller, passarle qui per evitare di rigenerarle
                     (board.legal_moves) una seconda volta."""

        piece_map = board.piece_map()
        clock_norm = GraphBuilder._clock_norm(clock_seconds)

        turn = 1.0 if board.turn == chess.WHITE else 0.0
        is_check = 1.0 if board.is_check() else 0.0
        castle_wk = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        castle_wq = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        castle_bk = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        castle_bq = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        global_feat = [turn, is_check, castle_wk, castle_wq, castle_bk, castle_bq]

        # --- costruzione feature nodi (vettorizzata, niente loop di 64 torch.tensor(...)) ---
        x = torch.zeros((64, 10), dtype=torch.float)
        x[:, 2] = -1.0  # default: nessun pezzo -> color = -1 (vedi convenzione in docstring classe)
        x[:, 3] = clock_norm
        x[:, 4:10] = torch.tensor(global_feat, dtype=torch.float)

        if piece_map:
            squares = torch.tensor(list(piece_map.keys()), dtype=torch.long)
            ptypes = torch.tensor([PIECE_TYPE_IDX[p.piece_type] for p in piece_map.values()], dtype=torch.float)
            colors = torch.tensor([float(int(p.color)) for p in piece_map.values()], dtype=torch.float)
            x[squares, 0] = 1.0
            x[squares, 1] = ptypes
            x[squares, 2] = colors

        # --- costruzione archi ---
        edge_src, edge_dst, edge_type = [], [], []  # 0=legal_move, 1=attack, 2=pin

        if legal_moves is None:
            legal_moves = list(board.legal_moves)
        for move in legal_moves:
            edge_src.append(move.from_square)
            edge_dst.append(move.to_square)
            edge_type.append(0)

        for sq, piece in piece_map.items():
            for target_sq in board.attacks(sq):
                edge_src.append(sq)
                edge_dst.append(target_sq)
                edge_type.append(1)

            # un'unica chiamata a pin() al posto di is_pinned()+pin(): se il pezzo non e'
            # pinnato, pin() ritorna uno SquareSet "pieno" (lunghezza 64) da ignorare.
            pin_ray = board.pin(piece.color, sq)
            if len(pin_ray) < 64:
                for ray_sq in pin_ray:
                    attacker = piece_map.get(ray_sq)
                    if attacker and attacker.color != piece.color and attacker.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                        edge_src.append(ray_sq)
                        edge_dst.append(sq)
                        edge_type.append(2)

        if not edge_src:
            edge_src, edge_dst, edge_type = [0], [0], [0]

        data = Data(
            x=x,
            edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
            edge_attr=torch.tensor(edge_type, dtype=torch.long),
        )

        if label:
            data.mate_n = torch.tensor([label.get("mate_n") or 0], dtype=torch.long)
            data.y = torch.tensor([label.get("best_move_idx", -1)], dtype=torch.long)

        return data
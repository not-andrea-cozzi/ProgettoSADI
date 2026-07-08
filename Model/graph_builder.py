import torch
import chess
from torch_geometric.data import Data

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def board_to_pyg_data(board: chess.Board, clock_seconds: float = 0.0, label: dict | None = None) -> Data:
    """Converte board in torch_geometric.data.Data.
    Nodi: 64 caselle, feature = [has_piece, piece_type(0-6), color(-1/0/1), clock_norm,
                               turn, is_check, castle_wk, castle_wq, castle_bk, castle_bq]
    Archi: legal_move, attack, pin """

    piece_map = board.piece_map()
    x = torch.zeros((64, 10), dtype=torch.float)
    clock_norm = min(clock_seconds / 60.0, 1.0)  #60 da cambiare poi 

    turn = 1.0 if board.turn == chess.WHITE else 0.0
    is_check = 1.0 if board.is_check() else 0.0
    castle_wk = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    castle_wq = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    castle_bk = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    castle_bq = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    global_feat = [turn, is_check, castle_wk, castle_wq, castle_bk, castle_bq]

    for sq in chess.SQUARES:
        piece = piece_map.get(sq)
        if piece:
            ptype = PIECE_TYPES.index(piece.piece_type) + 1
            x[sq] = torch.tensor([1.0, ptype, float(int(piece.color)), clock_norm] + global_feat)
        else:
            x[sq] = torch.tensor([0.0, 0.0, -1.0, clock_norm] + global_feat)

    edge_index = []
    edge_attr = []  #0=legal_move, 1=attack, 2=pin

    for move in board.legal_moves:
        edge_index.append([move.from_square, move.to_square])
        edge_attr.append(0)

    for sq, piece in piece_map.items():
        for target_sq in board.attacks(sq):
            edge_index.append([sq, target_sq])
            edge_attr.append(1)
        if board.is_pinned(piece.color, sq):
            for ray_sq in chess.SquareSet(board.pin(piece.color, sq)):
                attacker = piece_map.get(ray_sq)
                if attacker and attacker.color != piece.color and attacker.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                    edge_index.append([ray_sq, sq])
                    edge_attr.append(2)

    if not edge_index:
        edge_index = [[0, 0]]
        edge_attr = [0]

    data = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.long),
    )

    if label:
        data.mate_n = torch.tensor([label.get("mate_n") or 0], dtype=torch.long)
        data.y = torch.tensor([label.get("best_move_idx", -1)], dtype=torch.long)

    return data
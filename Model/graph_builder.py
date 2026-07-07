import torch
import chess
from torch_geometric.data import Data

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def board_to_pyg_data(board: chess.Board, clock_seconds: float = 0.0, label: dict | None = None) -> Data:
    """Converte board in torch_geometric.data.Data.
    Nodi: 64 case, feature = [has_piece, piece_type(0-6), color(-1/0/1), clock_norm]
    Archi: legal_move / attack / pin (con edge_attr type-id)
    label: dict opzionale {'mate_n': int|None, 'best_move_idx': int} per training supervisionato
    """
    piece_map = board.piece_map()
    x = torch.zeros((64, 4), dtype=torch.float)
    clock_norm = min(clock_seconds / 60.0, 1.0)  # normalizza su 60s

    for sq in chess.SQUARES:
        piece = piece_map.get(sq)
        if piece:
            ptype = PIECE_TYPES.index(piece.piece_type) + 1
            x[sq] = torch.tensor([1.0, ptype, float(int(piece.color)), clock_norm])
        else:
            x[sq] = torch.tensor([0.0, 0.0, -1.0, clock_norm])

    edge_index = []
    edge_attr = []  # 0=legal_move, 1=attack, 2=pin

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
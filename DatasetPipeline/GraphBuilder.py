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
    Archi: legal_move (tipo 0), attack (tipo 1), pin (tipo 2)

    IMPORTANTE per l'addestramento:
    - Gli archi di tipo 0 (mosse legali) vengono aggiunti **prima** di tutti gli altri,
      e il loro ordine è identico a quello della lista `legal_moves` utilizzata.
    - L'attributo `data.y` deve contenere l'indice **locale** (all'interno di quella lista)
      della mossa corretta.
    - Durante il training, il modello filtrerà `edge_attr == 0` e manterrà esattamente
      quell'ordine, quindi `data.y` resterà allineato.
    """

    CLOCK_CAP_SECONDS = 600.0  # oltre questa soglia clock_norm satura a 1.0 (log-scale)

    @staticmethod
    def _clock_norm(clock_seconds: float) -> float:
        """Normalizzazione log-scale: preserva la differenza tra clock brevi (bullet/blitz)
        senza schiacciare tutto ciò che supera pochi minuti come farebbe una scala lineare."""
        cap = GraphBuilder.CLOCK_CAP_SECONDS
        denom = math.log1p(cap)
        if denom <= 0:
            return 0.0
        return min(math.log1p(max(clock_seconds, 0.0)) / denom, 1.0)

    @staticmethod
    def board_to_pyg_data(
        board: chess.Board,
        clock_seconds: float = 0.0,
        label: dict | None = None,
        legal_moves: list | None = None
    ) -> Data:
        """Converte board in torch_geometric.data.Data.

        Args:
            board: posizione scacchistica.
            clock_seconds: tempo SPESO sulla mossa (durata), non residuo.
            label: dizionario opzionale con chiavi:
                   - "mate_n": profondità di matto (intero)
                   - "best_move_idx": indice della mossa migliore nella lista `legal_moves`
            legal_moves: lista di mosse legali (opzionale, se non fornita viene calcolata).
                         **L'ordine di questa lista determina l'indice usato in `best_move_idx`.**
        """
        piece_map = board.piece_map()
        clock_norm = GraphBuilder._clock_norm(clock_seconds)

        turn = 1.0 if board.turn == chess.WHITE else 0.0
        is_check = 1.0 if board.is_check() else 0.0
        castle_wk = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        castle_wq = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        castle_bk = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        castle_bq = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        global_feat = [turn, is_check, castle_wk, castle_wq, castle_bk, castle_bq]

        # --- costruzione feature nodi (vettorizzata) ---
        x = torch.zeros((64, 10), dtype=torch.float)
        x[:, 2] = -1.0  # default: nessun pezzo -> color = -1
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
        edge_src, edge_dst, edge_type = [], [], []
        # 0 = legal_move, 1 = attack, 2 = pin

        # 1. Mosse legali: aggiunte per prime, ordine = legal_moves (o board.legal_moves)
        if legal_moves is None:
            legal_moves = list(board.legal_moves)
        # Se legal_moves è vuoto, aggiungiamo un arco fittizio per evitare tensori vuoti
        if not legal_moves:
            legal_moves = []  # non aggiungiamo nulla, ma gestiamo dopo
        for move in legal_moves:
            edge_src.append(move.from_square)
            edge_dst.append(move.to_square)
            edge_type.append(0)

        # 2. Attacchi
        for sq, piece in piece_map.items():
            for target_sq in board.attacks(sq):
                edge_src.append(sq)
                edge_dst.append(target_sq)
                edge_type.append(1)

            # 3. Pin
            pin_ray = board.pin(piece.color, sq)
            if len(pin_ray) < 64:
                for ray_sq in pin_ray:
                    attacker = piece_map.get(ray_sq)
                    if attacker and attacker.color != piece.color and attacker.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                        edge_src.append(ray_sq)
                        edge_dst.append(sq)
                        edge_type.append(2)

        # Se non ci sono archi (caso raro, ad esempio re da solo), aggiungiamo un arco fittizio
        if not edge_src:
            edge_src, edge_dst, edge_type = [0], [0], [0]

        data = Data(
            x=x,
            edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
            edge_attr=torch.tensor(edge_type, dtype=torch.long),
        )

        if label:
            # mate_n: profondità del matto (deve essere ≤ MAX_MATE_N)
            mate_n = label.get("mate_n", 0)
            data.mate_n = torch.tensor([mate_n], dtype=torch.long)

            # best_move_idx: indice locale nella lista legal_moves
            best_idx = label.get("best_move_idx", -1)
            # Controllo di sicurezza: se l'indice è -1 o fuori range, lo sostituiamo con 0
            # (ma questo non dovrebbe accadere se il dataset è ben costruito)
            if best_idx < 0 or (legal_moves and best_idx >= len(legal_moves)):
                best_idx = 0
            data.y = torch.tensor([best_idx], dtype=torch.long)

        return data
"""
GraphBuilder.py (schema compresso)

Costruisce sample per l'addestramento del modello mate-solving, con SOLO i
campi utili alla predizione della best move (+ mate_n come task ausiliario),
usando i tipi di dato piu' piccoli che li rappresentano senza perdita di
informazione utile. Nessuna stringa nel sample: fen/best_move_uci/problem_id
vanno in un .jsonl separato via write_debug_jsonl, indicizzato sullo stesso
ordine di scrittura del dataset .pt.

SCHEMA DEL SAMPLE (torch_geometric.data.Data):

    board_packed    uint8[64]   1 byte/casella: bit0-2 piece_type (0-6),
                                 bit3 color (1=bianco, ignorato se
                                 piece_type==0), bit4-7 riservati/0.
    global_flags    uint8[1]    bitfield: bit0=turn (1=bianco muove),
                                 bit1=is_check, bit2=castle_wk, bit3=castle_wq,
                                 bit4=castle_bk, bit5=castle_bq.
    clock_norm      float16[1]  tempo speso sulla mossa, log-normalizzato
                                 in [0,1] (vedi _clock_norm).
    edge_index      uint8[2,E]  indici di casella 0-63 (uint8 basta, < 255).
                                 NOTA: salvato su disco come uint8; va
                                 espanso a long SOLO a runtime nel collate,
                                 perche' PyG/PyTorch richiedono edge_index
                                 int64 per le operazioni di indicizzazione
                                 interne (scatter, message passing). Il
                                 risparmio di questo schema e' quindi su
                                 DISCO; in RAM durante il forward
                                 l'edge_index espanso torna a pesare come
                                 la versione originale per la durata del
                                 batch corrente.
    edge_attr       uint8[E]    0=legal_move, 1=attack, 2=pin, 3=pad
                                 (arco fittizio di sentinella, vedi NOTA
                                 EDGE_PAD sotto: MAI un arco reale).
    mate_n          uint8[1]    profondita' di matto (1-10 nel range usato).
    best_move_idx   uint8[1]    indice LOCALE nella lista legal_moves
                                 passata a build (max 218 mosse legali
                                 possibili in teoria, sempre < 255: uint8
                                 sufficiente).
    value_target    float16[1]  1/mate_n per un matto forzato per il mover
                                 (NaN se mate_n non disponibile/0). Campo
                                 ridondante rispetto a mate_n (derivabile
                                 con 1/mate_n) ma MANTENUTO su disco perche'
                                 richiesto come campo d'ingresso dalla
                                 libreria esterna di training gia' pronta
                                 (contratto d'interfaccia fisso).
    rating          float16[1]  rating del mover (NaN se assente). Usato
                                 come feature di input: a parita' di
                                 posizione, giocatori di rating diverso
                                 scelgono mosse diverse, quindi condizionare
                                 la predizione al rating e' informativo
                                 (normalizzato a runtime, vedi RatingStats).

NOTA EDGE_PAD (fix bug arco fittizio):
Se una posizione non produce NESSUN arco (nessuna mossa legale, nessun
attacco, nessun pin — caso patologico, praticamente mai osservato su una
board valida non gia' in stallo/scacco matto, che dovrebbero essere
filtrate a monte), il grafo NON puo' restare senza archi perche' alcuni
layer GNN a valle richiedono edge_index non vuoto. La versione precedente
riempiva questo caso con un arco [0,0] taggato edge_attr=EDGE_LEGAL_MOVE
(0), indistinguibile da un vero arco "mossa legale" una volta salvato su
disco: un consumatore a valle non ha modo di sapere che quell'arco e'
fittizio, e potrebbe far corrispondere erroneamente best_move_idx=0 a
questo arco fantasma. Ora si usa un self-loop [0,0] con edge_attr=EDGE_PAD
(3), un codice riservato e mai prodotto altrove, cosi' il fallback resta
sempre distinguibile da un arco reale in fase di ispezione/debug o da
qualunque logica a valle che filtri per tipo di arco.

RATIO DI COMPRESSIONE SU DISCO (indicativo, board con ~20 pezzi):
    ~40 archi  -> originale ~3544B/sample, compresso ~193B/sample (~18x)
    ~20 archi  -> originale ~3064B/sample, compresso ~133B/sample (~23x)
"""
import math
import json
import os
from typing import Optional, List

import chess
import torch
from torch_geometric.data import Data

_PIECE_TYPE_TO_CODE = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}
_CODE_TO_PIECE_TYPE = {v: k for k, v in _PIECE_TYPE_TO_CODE.items()}
_COLOR_BIT = 1 << 3
_PIECE_TYPE_MASK = 0b0111

# edge_attr codes
EDGE_LEGAL_MOVE = 0
EDGE_ATTACK = 1
EDGE_PIN = 2
EDGE_PAD = 3  # sentinella: arco fittizio di riempimento, MAI un arco reale

CLOCK_CAP_SECONDS = 600.0


class GraphBuilder:
    """Costruisce sample compressi per il training e offre un decoder per
    debug/ispezione manuale (non usato nel training stesso)."""

    # ================================================================
    # NORMALIZZAZIONE CLOCK
    # ================================================================

    @staticmethod
    def _clock_norm(clock_seconds: float) -> float:
        """Normalizzazione log-scale: preserva la differenza tra clock
        brevi (bullet/blitz) senza schiacciare tutto cio' che supera pochi
        minuti come farebbe una scala lineare."""
        denom = math.log1p(CLOCK_CAP_SECONDS)
        if denom <= 0:
            return 0.0
        return min(math.log1p(max(clock_seconds, 0.0)) / denom, 1.0)

    # ================================================================
    # BOARD -> uint8[64] PACKED
    # ================================================================

    @staticmethod
    def _pack_board(board: "chess.Board") -> torch.Tensor:
        packed = torch.zeros(64, dtype=torch.uint8)
        for square, piece in board.piece_map().items():
            byte = _PIECE_TYPE_TO_CODE[piece.piece_type] & _PIECE_TYPE_MASK
            if piece.color == chess.WHITE:
                byte |= _COLOR_BIT
            packed[square] = byte
        return packed

    @staticmethod
    def unpack_board(packed: torch.Tensor) -> dict:
        """Decodifica board_packed in {square: (piece_type, color)}.
        Solo per debug/ispezione: il modello usa CompactDecoder per
        l'espansione a runtime dentro il forward."""
        out = {}
        for square in range(64):
            byte = int(packed[square].item())
            code = byte & _PIECE_TYPE_MASK
            if code == 0:
                continue
            color = bool(byte & _COLOR_BIT)
            out[square] = (_CODE_TO_PIECE_TYPE[code], chess.WHITE if color else chess.BLACK)
        return out

    # ================================================================
    # GLOBAL FLAGS -> uint8 BITFIELD
    # ================================================================

    @staticmethod
    def _pack_global_flags(board: "chess.Board") -> torch.Tensor:
        flags = 0
        if board.turn == chess.WHITE:
            flags |= 1 << 0
        if board.is_check():
            flags |= 1 << 1
        if board.has_kingside_castling_rights(chess.WHITE):
            flags |= 1 << 2
        if board.has_queenside_castling_rights(chess.WHITE):
            flags |= 1 << 3
        if board.has_kingside_castling_rights(chess.BLACK):
            flags |= 1 << 4
        if board.has_queenside_castling_rights(chess.BLACK):
            flags |= 1 << 5
        return torch.tensor([flags], dtype=torch.uint8)

    # ================================================================
    # BUILD
    # ================================================================

    @staticmethod
    def board_to_pyg_data(
        board: "chess.Board",
        clock_seconds: float = 0.0,
        label: Optional[dict] = None,
        legal_moves: Optional[List["chess.Move"]] = None,
        rating: Optional[float] = None,
    ) -> Data:
        """Converte board in un sample compresso.

        Args:
            board: posizione scacchistica.
            clock_seconds: tempo SPESO sulla mossa (durata), non residuo.
            label: dict opzionale con "mate_n" e "best_move_idx" (stesso
                   contratto della versione precedente, per compatibilita'
                   con i chiamanti esistenti in GamesBuilder/PuzzleGraphDataset).
            legal_moves: lista di mosse legali; il suo ORDINE determina
                   l'indice usato in best_move_idx e l'ordine degli archi
                   edge_attr==0.
            rating: rating del mover, se disponibile (None -> NaN salvato).
        """
        if legal_moves is None:
            legal_moves = list(board.legal_moves)

        board_packed = GraphBuilder._pack_board(board)
        global_flags = GraphBuilder._pack_global_flags(board)
        clock_norm = GraphBuilder._clock_norm(clock_seconds)

        edge_src, edge_dst, edge_type = [], [], []

        for move in legal_moves:
            edge_src.append(move.from_square)
            edge_dst.append(move.to_square)
            edge_type.append(EDGE_LEGAL_MOVE)

        piece_map = board.piece_map()
        for sq, piece in piece_map.items():
            for target_sq in board.attacks(sq):
                edge_src.append(sq)
                edge_dst.append(target_sq)
                edge_type.append(EDGE_ATTACK)

            pin_ray = board.pin(piece.color, sq)
            if len(pin_ray) < 64:
                for ray_sq in pin_ray:
                    attacker = piece_map.get(ray_sq)
                    if (
                        attacker
                        and attacker.color != piece.color
                        and attacker.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN)
                    ):
                        edge_src.append(ray_sq)
                        edge_dst.append(sq)
                        edge_type.append(EDGE_PIN)

        if not edge_src:
            # Nessun arco reale prodotto (caso patologico): self-loop
            # sentinella EDGE_PAD, MAI confondibile con un arco vero
            # (vedi NOTA EDGE_PAD in testa al file).
            edge_src, edge_dst, edge_type = [0], [0], [EDGE_PAD]

        edge_index_u8 = torch.tensor([edge_src, edge_dst], dtype=torch.uint8)
        edge_attr = torch.tensor(edge_type, dtype=torch.uint8)

        mate_n = 0
        best_move_idx = 0
        if label:
            mate_n = int(label.get("mate_n", 0))
            best_move_idx = int(label.get("best_move_idx", -1))
            if best_move_idx < 0 or (legal_moves and best_move_idx >= len(legal_moves)):
                best_move_idx = 0
            if best_move_idx > 255:
                raise ValueError(
                    f"best_move_idx={best_move_idx} eccede uint8 (max 255): "
                    f"posizione con troppe mosse legali per lo schema compresso."
                )
            if mate_n > 255:
                raise ValueError(f"mate_n={mate_n} eccede uint8 (max 255).")

        value_target = (1.0 / float(mate_n)) if mate_n and mate_n > 0 else float("nan")
        rating_val = float(rating) if rating is not None else float("nan")

        # PyG/PyTorch richiedono edge_index int64 per le op interne di
        # indicizzazione: data.edge_index resta long per compatibilita' con
        # InMemoryDataset.collate() e strumenti che lo ispezionano subito
        # dopo la costruzione (es. PuzzleGraphDataset.process()). La copia
        # compressa "canonica" da salvare su disco e' edge_index_u8; il
        # collate di training la rilegge e la riespande a runtime (vedi
        # Component/CompactCollate.py), NON usa mai questo edge_index long
        # per il salvataggio.
        data = Data(edge_index=edge_index_u8.long())
        data.edge_index_u8 = edge_index_u8
        data.edge_attr = edge_attr
        data.board_packed = board_packed
        data.global_flags = global_flags
        data.clock_norm = torch.tensor([clock_norm], dtype=torch.float16)
        data.mate_n = torch.tensor([mate_n], dtype=torch.uint8)
        data.best_move_idx = torch.tensor([best_move_idx], dtype=torch.uint8)
        data.value_target = torch.tensor([value_target], dtype=torch.float16)
        data.rating = torch.tensor([rating_val], dtype=torch.float16)

        return data

    # ================================================================
    # DEBUG JSONL (fen / best_move_uci / problem_id fuori dal tensor set)
    # ================================================================

    @staticmethod
    def write_debug_jsonl(records: List[dict], out_path: str) -> None:
        """records: lista di dict con almeno {"fen", "best_move_uci",
        "problem_id"} (+ extra opzionali: mate_n, source, clock_source...).
        Scritto in ordine: la riga N corrisponde al sample N del dataset
        .pt salvato nello stesso run, cosi' si puo' sempre risalire da un
        sample al suo FEN/mossa originali senza portare stringhe nel
        tensor dataset usato in training."""
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, out_path)
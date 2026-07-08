"""

Cambiare poi `GraphEncoder` con l'equivalente TimeGNN e il resto (training loop,
policy head) resta valido.

Input atteso:
    x:          [N, 10] -> [has_piece, piece_type(0-6), color(-1/0/1), clock_norm,
                             turn, is_check, castle_wk, castle_wq, castle_bk, castle_bq]
    edge_index: [2, E]
    edge_attr:  [E]     -> 0=legal_move, 1=attack, 2=pin
    batch:      [N]     assegnazione nodo->grafo (fornita da PyG DataLoader)

Output:
    move_scores: punteggio non normalizzato per ciascun arcco legal_move
    mate_logits: predizione ausiliaria del "mate in n" (0..N_MAX) a livello di grafo
"""
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import softmax as scatter_softmax

N_PIECE_TYPES = 7   # 0 = nessun pezzo, 1-6 = pedone..re
N_COLORS = 3         # mappiamo -1/0/1 -> 0/1/2
N_EDGE_TYPES = 3      # legal_move, attack, pin
MAX_MATE_N = 10       # come da held-out set del progetto (n fino a 10)


class InputEncoder(nn.Module):
    #Trasforma le 4 feature per nodo in un embedding denso.

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.piece_emb = nn.Embedding(N_PIECE_TYPES, hidden_dim // 2)
        self.color_emb = nn.Embedding(N_COLORS, hidden_dim // 4)
        # scalari: has_piece, clock, turn, is_check, castle_wk, castle_wq, castle_bk, castle_bq
        self.scalar_proj = nn.Linear(8, hidden_dim // 4)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_piece = x[:, 0:1]
        piece_type = x[:, 1].long().clamp(0, N_PIECE_TYPES - 1)
        color_raw = x[:, 2].long()  # -1, 0, 1
        color_idx = (color_raw + 1).clamp(0, N_COLORS - 1)  # -> 0,1,2
        clock = x[:, 3:4]
        global_scalars = x[:, 4:10]  # turn, is_check, castle_wk, castle_wq, castle_bk, castle_bq, ARROCCHI

        piece_vec = self.piece_emb(piece_type)
        color_vec = self.color_emb(color_idx)
        scalar_vec = self.scalar_proj(torch.cat([has_piece, clock, global_scalars], dim=-1))

        combined = torch.cat([piece_vec, color_vec, scalar_vec], dim=-1)
        return self.out_proj(combined)


class GraphEncoder(nn.Module):
    #Stack di layer di message passing con edge features (tipo di arco).

    def __init__(self, hidden_dim: int, num_layers: int = 4, heads: int = 4):
        super().__init__()
        self.input_encoder = InputEncoder(hidden_dim)
        self.edge_emb = nn.Embedding(N_EDGE_TYPES, hidden_dim)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerConv(
                    hidden_dim, hidden_dim // heads,
                    heads=heads, edge_dim=hidden_dim, dropout=0.1
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        h = self.input_encoder(x)
        e = self.edge_emb(edge_attr)

        for conv, norm in zip(self.layers, self.norms):
            h_new = conv(h, edge_index, e)
            h = norm(h + h_new)  # residual + norm
        return h


class PolicyGNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, heads: int = 4):
        super().__init__()
        self.encoder = GraphEncoder(hidden_dim, num_layers, heads)

        self.move_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.mate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, MAX_MATE_N + 1),
        )

    def forward(self, data):
        h = self.encoder(data.x, data.edge_index, data.edge_attr)

        legal_mask = data.edge_attr == 0
        src, dst = data.edge_index[:, legal_mask]
        move_feat = torch.cat([h[src], h[dst]], dim=-1)
        move_scores = self.move_scorer(move_feat).squeeze(-1)  # [E_legal]

        edge_batch = data.batch[src]  # a quale grafo appartiene ciascun arco legal_move

        graph_emb = global_mean_pool(h, data.batch)
        mate_logits = self.mate_head(graph_emb)

        return move_scores, edge_batch, mate_logits


def legal_move_log_probs(move_scores: torch.Tensor, edge_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Softmax raggruppata per grafo (ogni grafo ha un numero diverso di mosse legali,
    quindi non possiamo usare un softmax piatto)"""
    probs = scatter_softmax(move_scores, edge_batch, num_nodes=num_graphs)
    return torch.log(probs.clamp_min(1e-12))


def policy_targets_to_global_index(edge_batch: torch.Tensor, y_local: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Converte l'indice locale (best_move_idx dentro le sue mosse legali)
    nell'indice globale dentro il tensore move_scores del batch."""
    counts = torch.bincount(edge_batch, minlength=num_graphs)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    return offsets + y_local
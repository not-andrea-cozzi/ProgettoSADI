import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import softmax as scatter_softmax

N_PIECE_TYPES = 7   # 0 = vuoto, 1-6 = pedone..re
N_COLORS = 3        # -1, 0, 1 -> 0, 1, 2
N_EDGE_TYPES = 3    # 0=legal_move, 1=attack, 2=pin
MAX_MATE_N = 10

class InputEncoder(nn.Module):
    """Proietta il vettore x [N, 10] in un embedding denso."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.piece_emb = nn.Embedding(N_PIECE_TYPES, hidden_dim // 2)
        self.color_emb = nn.Embedding(N_COLORS, hidden_dim // 4)
        self.scalar_proj = nn.Linear(8, hidden_dim // 4)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        piece_type = x[:, 1].long().clamp(0, N_PIECE_TYPES - 1)
        color_idx = (x[:, 2].long() + 1).clamp(0, N_COLORS - 1)
        scalars = torch.cat([x[:, 0:1], x[:, 3:4], x[:, 4:10]], dim=-1)

        piece_vec = self.piece_emb(piece_type)
        color_vec = self.color_emb(color_idx)
        scalar_vec = self.scalar_proj(scalars)

        return self.out_proj(torch.cat([piece_vec, color_vec, scalar_vec], dim=-1))

class GraphEncoder(nn.Module):
    """Message passing su grafo scacchistico."""
    def __init__(self, hidden_dim: int, num_layers: int = 4, heads: int = 4):
        super().__init__()
        self.input_encoder = InputEncoder(hidden_dim)
        self.edge_emb = nn.Embedding(N_EDGE_TYPES, hidden_dim)

        self.layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=hidden_dim, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.input_encoder(x)
        e = self.edge_emb(edge_attr)
        for conv, norm in zip(self.layers, self.norms):
            h = norm(h + conv(h, edge_index, e))
        return h

def build_head(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, in_dim // 2 if in_dim != out_dim else in_dim),
        nn.ReLU(),
        nn.Linear(in_dim // 2 if in_dim != out_dim else in_dim, out_dim)
    )

class PolicyGNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, heads: int = 4):
        super().__init__()
        self.encoder = GraphEncoder(hidden_dim, num_layers, heads)
        self.move_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.mate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, MAX_MATE_N + 1)
        )

    def forward(self, data):
        h = self.encoder(data.x, data.edge_index, data.edge_attr)
        # IMPORTANTE: si assume che gli archi con edge_attr==0 (mosse legali) siano
        # ordinati nello stesso ordine di board.legal_moves usato durante la creazione
        # del dataset. Altrimenti l'indice target (data.y) non corrisponderà.
        legal_mask = data.edge_attr == 0
        src, dst = data.edge_index[:, legal_mask]

        move_feat = torch.cat([h[src], h[dst]], dim=-1)
        move_scores = self.move_scorer(move_feat).squeeze(-1)
        edge_batch = data.batch[src]

        graph_emb = global_mean_pool(h, data.batch)
        mate_logits = self.mate_head(graph_emb)
        return move_scores, edge_batch, mate_logits

def legal_move_log_probs(move_scores: torch.Tensor, edge_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    probs = scatter_softmax(move_scores, edge_batch, num_nodes=num_graphs)
    return torch.log(probs.clamp_min(1e-12))

def policy_targets_to_global_index(edge_batch: torch.Tensor, y_local: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """
    Converte l'indice locale (all'interno delle mosse legali di un grafo) in un indice globale
    nel tensore concatenato di tutte le mosse legali.
    PRE: edge_batch deve essere ordinato in modo che tutti gli archi di un grafo siano contigui.
    """
    # Calcola i conteggi per grafo
    counts = torch.bincount(edge_batch, minlength=num_graphs)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    return offsets + y_local
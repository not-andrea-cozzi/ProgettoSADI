import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from Model.PolicyGNN import GraphEncoder, MAX_MATE_N
from timegnn.models.gat_time_decay import TimeAwareGATConv

class TimeChainRefiner(nn.Module):
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, lambda_decay: float = 0.01):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim deve essere divisibile per num_heads")
        out_per_head = hidden_dim // num_heads

        self.gat = TimeAwareGATConv(
            hidden_dim, out_per_head, heads=num_heads, concat=True,
            edge_dim=1, lambda_decay=lambda_decay,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, pos_emb: torch.Tensor, chain_edge_index: torch.Tensor,
                chain_edge_attr: torch.Tensor) -> torch.Tensor:
        if chain_edge_index.numel() == 0:
            # nessun puzzle multi-mossa in questo batch: nessun contesto da propagare
            return pos_emb
        refined = self.gat(pos_emb, chain_edge_index, edge_attr=chain_edge_attr)
        return self.norm(pos_emb + refined)  # residual: vedi nota sul nodo senza archi entranti


class TimedPolicyGNN(nn.Module):
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, heads: int = 4,
                 lambda_decay: float = 0.01, use_time: bool = True):
        super().__init__()
        self.use_time = use_time
        self.encoder = GraphEncoder(hidden_dim, num_layers, heads)
        self.time_refiner = TimeChainRefiner(hidden_dim, num_heads=heads, lambda_decay=lambda_decay)

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

    def forward(self, inner_batch, chain_edge_index, chain_edge_attr):
        x = inner_batch.x
        if not self.use_time:
            x = x.clone()
            x[:, 3] = 0.0  # azzera clock_norm: nessun segnale temporale a livello nodo

        h = self.encoder(x, inner_batch.edge_index, inner_batch.edge_attr)

        legal_mask = inner_batch.edge_attr == 0
        src, dst = inner_batch.edge_index[:, legal_mask]
        move_feat = torch.cat([h[src], h[dst]], dim=-1)
        move_scores = self.move_scorer(move_feat).squeeze(-1)  # [E_legal], INVARIATO da PolicyGNN

        edge_batch = inner_batch.batch[src]

        pos_emb = global_mean_pool(h, inner_batch.batch)  # [num_posizioni_nel_batch, hidden_dim]
        if self.use_time:
            pos_emb = self.time_refiner(pos_emb, chain_edge_index, chain_edge_attr)
        # use_time=False: mate_head vede il pos_emb "grezzo", nessun contesto tra ply

        mate_logits = self.mate_head(pos_emb)

        return move_scores, edge_batch, mate_logits
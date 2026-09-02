import torch

def _argmax_per_graph(scores, edge_batch, num_graphs):
    """Restituisce l'indice globale dello score massimo per ogni grafo."""
    preds = []
    for g in range(num_graphs):
        mask = edge_batch == g
        if mask.any():
            local_scores = scores[mask]
            best_local = local_scores.argmax()
            preds.append(mask.nonzero(as_tuple=False)[best_local].item())
        else:
            preds.append(-1)
    return torch.tensor(preds, dtype=torch.long)

import argparse
import os
import torch
from torch_geometric.loader import DataLoader

from Model.PolicyGNN import PolicyGNN, legal_move_log_probs, policy_targets_to_global_index

"""
Training loop per PolicyGNN su merged_train.pt / merged_val.pt
(prodotti da Component/PuzzleGraphDataset.merge_and_split).

Uso:
    python -m Component.Trainer --data_dir dataset/merged -- 50 epoche

l'unica cosa da cambiare qui è l'import e l'istanziazione del modello
"""

def load_split(data_dir: str, name: str):
    path = os.path.join(data_dir, f"merged_{name}.pt")
    data_list = torch.load(path, weights_only=False)
    return data_list


def run_epoch(model, loader, optimizer, device, mate_loss_weight: float = 0.3, train: bool = True):
    model.train(train)
    total_loss, total_correct, total_examples = 0.0, 0, 0

    for batch in loader:
        batch = batch.to(device)
        num_graphs = batch.num_graphs

        with torch.set_grad_enabled(train):
            move_scores, edge_batch, mate_logits = model(batch)
            log_probs = legal_move_log_probs(move_scores, edge_batch, num_graphs)

            target_idx = policy_targets_to_global_index(edge_batch, batch.y, num_graphs)
            policy_loss = -log_probs[target_idx].mean()

            mate_target = batch.mate_n.clamp(0, mate_logits.size(-1) - 1)
            mate_loss = torch.nn.functional.cross_entropy(mate_logits, mate_target)

            loss = policy_loss + mate_loss_weight * mate_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #accuracy: per ogni grafo, la mossa con punteggio massimo tra le sue legal_move edges
        pred_is_best = _argmax_per_graph(move_scores, edge_batch, num_graphs) == target_idx
        total_correct += pred_is_best.sum().item()
        total_examples += num_graphs
        total_loss += loss.item() * num_graphs

    return total_loss / total_examples, total_correct / total_examples


def _argmax_per_graph(scores: torch.Tensor, edge_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Ritorna, per ciascun grafo, l'indice globale (dentro `scores`) dell'arco
    legal_move con score massimo."""
    best = scores.new_full((num_graphs,), float("-inf"))
    best_idx = torch.zeros(num_graphs, dtype=torch.long, device=scores.device)
    for i in range(scores.size(0)):
        g = edge_batch[i]
        if scores[i] > best[g]:
            best[g] = scores[i]
            best_idx[g] = i
    return best_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/merged")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_data = load_split(args.data_dir, "train")
    val_data = load_split(args.data_dir, "val")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model = PolicyGNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, device, train=False)

        print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pt"))

    print(f"Fine training. Miglior val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
import os
import sys
import logging
import torch
from torch.utils.data import DataLoader
from timegnn.train.early_stopping import EarlyStopping

from Component.PuzzleGraphDataset import PuzzleGraphDataset
from Component.PuzzleSequenceDataset import PuzzleSequenceDataset, timed_collate_fn
from Component.TimeStatBuilder import load_avg_time_by_rating
from Training.TimeChainGnn import TimedPolicyGNN
from Model.PolicyGNN import legal_move_log_probs, policy_targets_to_global_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("timed_trainer")

PUZZLE_CSV = "dataset/lichess_puzzles.csv"
PUZZLE_ROOT = "dataset/puzzles"
TIME_STATS_JSON = "dataset/avg_time_by_rating.json"
MATE_RANGE = (1, 5)
MAX_PUZZLES = None

EPOCHS = 50
BATCH_SIZE = 64          # numero di PUZZLE per batch (non di posizioni: ogni puzzle ha piu' ply)
LR = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 4
LAMBDA_DECAY = 0.01
CHECKPOINT_DIR = "checkpoints"


def load_puzzle_split(csv_path, root, split, mate_range, max_puzzles, avg_time_by_rating):
    ds = PuzzleGraphDataset(csv_path, root, split=split, mate_range=mate_range,
                             max_puzzles=max_puzzles, avg_time_by_rating=avg_time_by_rating)
    return list(ds)


def run_epoch(model, loader, optimizer, device, mate_loss_weight: float = 0.3, train: bool = True):
    model.train(train)
    total_loss, total_move_correct, total_mate_correct, total_examples = 0.0, 0, 0, 0

    for inner_batch, chain_edge_index, chain_edge_attr in loader:
        inner_batch = inner_batch.to(device)
        chain_edge_index = chain_edge_index.to(device)
        chain_edge_attr = chain_edge_attr.to(device)
        num_graphs = inner_batch.num_graphs

        with torch.set_grad_enabled(train):
            move_scores, edge_batch, mate_logits = model(inner_batch, chain_edge_index, chain_edge_attr)
            log_probs = legal_move_log_probs(move_scores, edge_batch, num_graphs)

            target_idx = policy_targets_to_global_index(edge_batch, inner_batch.y, num_graphs)
            policy_loss = -log_probs[target_idx].mean()

            mate_target = inner_batch.mate_n.clamp(0, mate_logits.size(-1) - 1)
            mate_loss = torch.nn.functional.cross_entropy(mate_logits, mate_target)

            loss = policy_loss + mate_loss_weight * mate_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        move_pred_is_best = _argmax_per_graph(move_scores, edge_batch, num_graphs) == target_idx
        mate_pred_correct = mate_logits.argmax(dim=-1) == mate_target

        total_move_correct += move_pred_is_best.sum().item()
        total_mate_correct += mate_pred_correct.sum().item()
        total_examples += num_graphs
        total_loss += loss.item() * num_graphs

    return (total_loss / total_examples,
            total_move_correct / total_examples,
            total_mate_correct / total_examples)


def _argmax_per_graph(scores: torch.Tensor, edge_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Identico a Component/Trainer.py: per ciascun grafo, indice globale
    (dentro `scores`) dell'arco legal_move con score massimo."""
    best = scores.new_full((num_graphs,), float("-inf"))
    best_idx = torch.zeros(num_graphs, dtype=torch.long, device=scores.device)
    for i in range(scores.size(0)):
        g = edge_batch[i]
        if scores[i] > best[g]:
            best[g] = scores[i]
            best_idx[g] = i
    return best_idx


def _checkpoint_path(tag: str, kind: str) -> str:
    # kind: "latest" (per riprendere dopo un crash) o "best" (miglior val_move_acc)
    return os.path.join(CHECKPOINT_DIR, f"{tag}_{kind}.pt")


def save_checkpoint(path, model, optimizer, epoch, best_val_move_acc, best_val_mate_acc, early_stopper):
    tmp_path = path + ".tmp"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_move_acc": best_val_move_acc,
        "best_val_mate_acc": best_val_mate_acc,
        "es_counter": early_stopper.counter,
        "es_best_loss": early_stopper.best_loss,
    }, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint_if_exists(path, model, optimizer, device, early_stopper):
    if not os.path.exists(path):
        return 0, 0.0, 0.0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    early_stopper.counter = ckpt.get("es_counter", 0)
    early_stopper.best_loss = ckpt.get("es_best_loss", float("inf"))
    return ckpt["epoch"], ckpt["best_val_move_acc"], ckpt["best_val_mate_acc"]


def train_one_config(use_time: bool, train_loader, val_loader, device):
    tag = "timed" if use_time else "untimed"
    model = TimedPolicyGNN(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                            lambda_decay=LAMBDA_DECAY, use_time=use_time).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    early_stopper = EarlyStopping(patience=7, delta=0.0)

    latest_path = _checkpoint_path(tag, "latest")
    best_path = _checkpoint_path(tag, "best")

    start_epoch, best_val_move_acc, best_val_mate_acc = load_checkpoint_if_exists(
        latest_path, model, optimizer, device, early_stopper)

    if start_epoch >= EPOCHS:
        logger.info(f"[{tag}] gia' completato ({start_epoch}/{EPOCHS} epoche), salto.")
        return best_val_move_acc, best_val_mate_acc
    if start_epoch > 0:
        logger.info(f"[{tag}] riprendo da epoca {start_epoch + 1}/{EPOCHS} (checkpoint trovato in {latest_path})")

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        train_loss, train_move_acc, train_mate_acc = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_move_acc, val_mate_acc = run_epoch(model, val_loader, optimizer, device, train=False)

        logger.info(f"[{tag}][epoch {epoch:03d}] train_loss={train_loss:.4f} "
                    f"train_move_acc={train_move_acc:.4f} train_mate_acc={train_mate_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_move_acc={val_move_acc:.4f} val_mate_acc={val_mate_acc:.4f}")

        if val_move_acc > best_val_move_acc:
            best_val_move_acc = val_move_acc
            best_val_mate_acc = val_mate_acc
            save_checkpoint(best_path, model, optimizer, epoch, best_val_move_acc, best_val_mate_acc, early_stopper)

        # salvato SEMPRE (anche se non e' il migliore): e' questo che permette
        # di riprendere dall'ultima epoca completata invece che da zero.
        save_checkpoint(latest_path, model, optimizer, epoch, best_val_move_acc, best_val_mate_acc, early_stopper)

        if early_stopper(val_loss):
            logger.info(f"[{tag}] early stop a epoca {epoch} "
                        f"(val_loss non migliora da {early_stopper.patience} epoche, "
                        f"best_val_loss={early_stopper.best_loss:.4f})")
            break

    return best_val_move_acc, best_val_mate_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    avg_time_by_rating = (load_avg_time_by_rating(TIME_STATS_JSON)
                           if os.path.exists(TIME_STATS_JSON) else None)

    train_positions = load_puzzle_split(PUZZLE_CSV, PUZZLE_ROOT, "train", MATE_RANGE, MAX_PUZZLES, avg_time_by_rating)
    val_positions = load_puzzle_split(PUZZLE_CSV, PUZZLE_ROOT, "val", MATE_RANGE, MAX_PUZZLES, avg_time_by_rating)

    train_loader = DataLoader(PuzzleSequenceDataset(train_positions), batch_size=BATCH_SIZE,
                               shuffle=True, collate_fn=timed_collate_fn)
    val_loader = DataLoader(PuzzleSequenceDataset(val_positions), batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=timed_collate_fn)

    
    move_acc_timed, mate_acc_timed = train_one_config(True, train_loader, val_loader, device)
    move_acc_untimed, mate_acc_untimed = train_one_config(False, train_loader, val_loader, device)

    logger.info("--- confronto finale (best val_acc per epoca) ---")
    logger.info(f"con tempo:    move_acc={move_acc_timed:.4f}  mate_acc={mate_acc_timed:.4f}")
    logger.info(f"senza tempo:  move_acc={move_acc_untimed:.4f}  mate_acc={mate_acc_untimed:.4f}")


if __name__ == "__main__":
    main()
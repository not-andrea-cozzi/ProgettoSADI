import argparse
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
logger = logging.getLogger("timed_trainer_5070")


def load_split(data_dir: str, name: str):
    path = os.path.join(data_dir, f"merged_{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} non trovato. Verifica --data_dir o esegui prima main.py "
            f"per generare i dataset merged."
        )
    return torch.load(path, weights_only=False)


def run_epoch(model, loader, optimizer, device, mate_loss_weight: float, train: bool):
    model.train(train)
    total_loss, total_move_correct, total_mate_correct, total_examples = 0.0, 0, 0, 0

    for inner_batch, chain_edge_index, chain_edge_attr in loader:
        inner_batch = inner_batch.to(device, non_blocking=True)
        chain_edge_index = chain_edge_index.to(device, non_blocking=True)
        chain_edge_attr = chain_edge_attr.to(device, non_blocking=True)
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
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        move_pred_is_best = _argmax_per_graph(move_scores, edge_batch, num_graphs) == target_idx
        mate_pred_correct = mate_logits.argmax(dim=-1) == mate_target

        total_move_correct += move_pred_is_best.sum().item()
        total_mate_correct += mate_pred_correct.sum().item()
        total_examples += num_graphs
        total_loss += loss.item() * num_graphs

    return (
        total_loss / total_examples,
        total_move_correct / total_examples,
        total_mate_correct / total_examples,
    )


def _argmax_per_graph(scores: torch.Tensor, edge_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Identico a Training/TimedTraining.py: per ciascun grafo, indice globale
    (dentro `scores`) dell'arco legal_move con score massimo."""
    best = scores.new_full((num_graphs,), float("-inf"))
    best_idx = torch.zeros(num_graphs, dtype=torch.long, device=scores.device)
    for i in range(scores.size(0)):
        g = edge_batch[i]
        if scores[i] > best[g]:
            best[g] = scores[i]
            best_idx[g] = i
    return best_idx


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


def train_one_config(use_time, train_loader, val_loader, device, args, checkpoint_dir):
    tag = "timed" if use_time else "untimed"
    model = TimedPolicyGNN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lambda_decay=args.lambda_decay,
        use_time=use_time,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stopper = EarlyStopping(patience=args.patience, delta=0.0)

    latest_path = os.path.join(checkpoint_dir, f"{tag}_latest.pt")
    best_path = os.path.join(checkpoint_dir, f"{tag}_best.pt")

    start_epoch, best_val_move_acc, best_val_mate_acc = load_checkpoint_if_exists(
        latest_path, model, optimizer, device, early_stopper)

    if start_epoch >= args.epochs:
        logger.info(f"[{tag}] gia' completato ({start_epoch}/{args.epochs}), salto.")
        return best_val_move_acc, best_val_mate_acc
    if start_epoch > 0:
        logger.info(f"[{tag}] riprendo da epoca {start_epoch + 1}/{args.epochs}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss, train_move_acc, train_mate_acc = run_epoch(
            model, train_loader, optimizer, device, args.mate_loss_weight, train=True)
        val_loss, val_move_acc, val_mate_acc = run_epoch(
            model, val_loader, optimizer, device, args.mate_loss_weight, train=False)

        logger.info(
            f"[{tag}][epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"train_move_acc={train_move_acc:.4f} train_mate_acc={train_mate_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_move_acc={val_move_acc:.4f} val_mate_acc={val_mate_acc:.4f}"
        )

        if val_move_acc > best_val_move_acc:
            best_val_move_acc = val_move_acc
            best_val_mate_acc = val_mate_acc
            save_checkpoint(best_path, model, optimizer, epoch, best_val_move_acc, best_val_mate_acc, early_stopper)

        save_checkpoint(latest_path, model, optimizer, epoch, best_val_move_acc, best_val_mate_acc, early_stopper)

        if early_stopper(val_loss):
            logger.info(f"[{tag}] early stop a epoca {epoch} (best_val_loss={early_stopper.best_loss:.4f})")
            break

    return best_val_move_acc, best_val_mate_acc


def main():
    parser = argparse.ArgumentParser(description="Training TimedPolicyGNN, ottimizzato per NVIDIA 5070 (fp32)")
    parser.add_argument("--data_dir", type=str, default="dataset/merged")
    parser.add_argument("--time_stats_json", type=str, default="dataset/avg_time_by_rating.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32,
                         help="Numero di PUZZLE per batch, non di posizioni. "
                              "Valore conservativo per 12GB VRAM in fp32: alza se non satura la GPU.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--lambda_decay", type=float, default=0.01)
    parser.add_argument("--mate_loss_weight", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--only", type=str, choices=["timed", "untimed", "both"], default="both",
                         help="Allena solo la variante timed, solo untimed, o entrambe (ablation completa)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA non disponibile in questo ambiente: il training girera' su CPU "
            "e sara' molto piu' lento. Verifica installazione driver/CUDA per la 5070."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    avg_time_by_rating = (
        load_avg_time_by_rating(args.time_stats_json)
        if os.path.exists(args.time_stats_json) else None
    )
    if avg_time_by_rating is None:
        logger.warning(
            f"{args.time_stats_json} non trovato: i clock simulati dei puzzle "
            f"useranno il fallback lineare in PuzzleGraphDataset, non tempi reali Lichess."
        )

    logger.info("Carico merged_train.pt / merged_val.pt (gia' generati da main.py)...")
    train_positions = load_split(args.data_dir, "train")
    val_positions = load_split(args.data_dir, "val")
    logger.info(f"train: {len(train_positions)} posizioni-grafo | val: {len(val_positions)} posizioni-grafo")

    train_dataset = PuzzleSequenceDataset(train_positions)
    val_dataset = PuzzleSequenceDataset(val_positions)
    logger.info(
        f"Sequenze puzzle raggruppate: train={len(train_dataset)} val={len(val_dataset)} "
        f"(le posizioni da games.pt, senza puzzle_id, NON entrano nel canale a catena "
        f"per costruzione di PuzzleSequenceDataset.group_puzzle_sequences)."
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=timed_collate_fn, num_workers=args.num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=timed_collate_fn, num_workers=args.num_workers, pin_memory=pin_memory,
    )

    results = {}
    if args.only in ("timed", "both"):
        move_acc, mate_acc = train_one_config(True, train_loader, val_loader, device, args, args.checkpoint_dir)
        results["timed"] = (move_acc, mate_acc)
    if args.only in ("untimed", "both"):
        move_acc, mate_acc = train_one_config(False, train_loader, val_loader, device, args, args.checkpoint_dir)
        results["untimed"] = (move_acc, mate_acc)

    if len(results) == 2:
        logger.info("--- confronto finale (best val_acc per epoca) ---")
        logger.info(f"con tempo:   move_acc={results['timed'][0]:.4f}  mate_acc={results['timed'][1]:.4f}")
        logger.info(f"senza tempo: move_acc={results['untimed'][0]:.4f}  mate_acc={results['untimed'][1]:.4f}")


if __name__ == "__main__":
    main()
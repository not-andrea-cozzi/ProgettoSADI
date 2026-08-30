"""
Valutazione dei modelli timed vs untimed sul test set (merged_test.pt).
Genera:
  - test_metrics.csv          : metriche aggregate per tag
  - test_per_rating.csv       : metriche per fascia rating
  - test_report.txt           : riepilogo testuale
  - plots/aggregate_bars.png  : barre timed vs untimed su metriche aggregate
  - plots/per_rating_move.png : move_acc per fascia rating (curva)
  - plots/per_rating_mate.png : mate_acc per fascia rating (curva)
  - plots/hist_rating.png     : distribuzione delle posizioni per fascia rating
  - plots/confusion_mate_timed.png / _untimed.png : confusion matrix mate-in-N
"""

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # backend non-interattivo, safe su Windows / senza display
import matplotlib.pyplot as plt

from Component.PuzzleSequenceDataset import PuzzleSequenceDataset, timed_collate_fn
from Training.TimeChainGnn import TimedPolicyGNN
from Model.PolicyGNN import legal_move_log_probs, policy_targets_to_global_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_evaluator")


# -----------------------------------------------------------------------------
# helper: argmax per grafo (indice GLOBALE nel tensore edge-level)
# -----------------------------------------------------------------------------
def _argmax_per_graph(scores: torch.Tensor, edge_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    best_score = scores.new_full((num_graphs,), float("-inf"))
    best_score.scatter_reduce_(0, edge_batch, scores, reduce="amax", include_self=True)
    is_best = scores == best_score[edge_batch]
    idx_range = torch.arange(scores.size(0), device=scores.device)
    sentinel = scores.size(0) + 1
    masked = torch.where(is_best, idx_range, torch.full_like(idx_range, sentinel))
    argmax_global = torch.full((num_graphs,), sentinel, dtype=torch.long, device=scores.device)
    argmax_global.scatter_reduce_(0, edge_batch, masked, reduce="amin", include_self=True)
    return argmax_global


# -----------------------------------------------------------------------------
# loader del test set
# -----------------------------------------------------------------------------
def load_test_split(data_dir: str):
    path = os.path.join(data_dir, "merged_test.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} non trovato.")
    return torch.load(path, weights_only=False)


# -----------------------------------------------------------------------------
# costruzione modello + caricamento checkpoint best
# -----------------------------------------------------------------------------
def build_and_load_model(ckpt_path: str, use_time: bool, args, device: torch.device) -> TimedPolicyGNN:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} non trovato. "
                                f"Hai gia' lanciato train_model.py?")
    model = TimedPolicyGNN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lambda_decay=args.lambda_decay,
        use_time=use_time,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    # se il modello era stato salvato dopo torch.compile, i nomi hanno prefisso "_orig_mod."
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Caricato {ckpt_path} (epoca={ckpt.get('epoch','?')}, "
                f"val_move_acc={ckpt.get('best_val_move_acc',0):.4f}, "
                f"val_mate_acc={ckpt.get('best_val_mate_acc',0):.4f})")
    return model


# -----------------------------------------------------------------------------
# valutazione: raccoglie predizioni + rating per plot per-fascia e confusion matrix
# -----------------------------------------------------------------------------
def evaluate(model, loader, device, mate_loss_weight: float, num_mate_classes: int):
    model.eval()
    total_loss = 0.0
    total_examples = 0

    # accumulatori per confusion matrix e per-rating
    move_correct_flags: List[np.ndarray] = []
    mate_correct_flags: List[np.ndarray] = []
    mate_true_all: List[np.ndarray] = []
    mate_pred_all: List[np.ndarray] = []
    rating_all: List[np.ndarray] = []
    has_rating = True  # verra' spento se il campo manca

    with torch.no_grad():
        for inner_batch, chain_edge_index, chain_edge_attr in loader:
            inner_batch = inner_batch.to(device, non_blocking=True)
            chain_edge_index = chain_edge_index.to(device, non_blocking=True)
            chain_edge_attr = chain_edge_attr.to(device, non_blocking=True)
            num_graphs = inner_batch.num_graphs

            move_scores, edge_batch, mate_logits = model(inner_batch, chain_edge_index, chain_edge_attr)
            log_probs = legal_move_log_probs(move_scores, edge_batch, num_graphs)

            target_idx = policy_targets_to_global_index(edge_batch, inner_batch.y, num_graphs)
            policy_loss = -log_probs[target_idx].mean()

            mate_target = inner_batch.mate_n.clamp(0, mate_logits.size(-1) - 1)
            mate_loss = torch.nn.functional.cross_entropy(mate_logits, mate_target)
            loss = policy_loss + mate_loss_weight * mate_loss

            move_pred_global = _argmax_per_graph(move_scores, edge_batch, num_graphs)
            move_correct = (move_pred_global == target_idx).cpu().numpy()

            mate_pred = mate_logits.argmax(dim=-1)
            mate_correct = (mate_pred == mate_target).cpu().numpy()

            move_correct_flags.append(move_correct)
            mate_correct_flags.append(mate_correct)
            mate_true_all.append(mate_target.cpu().numpy())
            mate_pred_all.append(mate_pred.cpu().numpy())

            if has_rating:
                if hasattr(inner_batch, "rating") and inner_batch.rating is not None:
                    rating_all.append(inner_batch.rating.detach().cpu().numpy().reshape(-1))
                else:
                    has_rating = False
                    logger.warning("Campo 'rating' non trovato in inner_batch: i plot per fascia rating verranno saltati.")

            total_loss += loss.item() * num_graphs
            total_examples += num_graphs

    move_correct_np = np.concatenate(move_correct_flags) if move_correct_flags else np.array([])
    mate_correct_np = np.concatenate(mate_correct_flags) if mate_correct_flags else np.array([])
    mate_true_np = np.concatenate(mate_true_all) if mate_true_all else np.array([])
    mate_pred_np = np.concatenate(mate_pred_all) if mate_pred_all else np.array([])
    rating_np = np.concatenate(rating_all) if (has_rating and rating_all) else None

    return {
        "loss": total_loss / max(total_examples, 1),
        "move_acc": float(move_correct_np.mean()) if len(move_correct_np) else 0.0,
        "mate_acc": float(mate_correct_np.mean()) if len(mate_correct_np) else 0.0,
        "n": total_examples,
        "move_correct": move_correct_np,
        "mate_correct": mate_correct_np,
        "mate_true": mate_true_np,
        "mate_pred": mate_pred_np,
        "rating": rating_np,
    }


# -----------------------------------------------------------------------------
# aggregazione per fascia rating
# -----------------------------------------------------------------------------
def per_rating_metrics(results: dict, bin_edges: np.ndarray) -> Dict[str, np.ndarray]:
    """Restituisce, per ciascun bin di rating, (n, move_acc, mate_acc)."""
    rating = results["rating"]
    move_c = results["move_correct"]
    mate_c = results["mate_correct"]

    if rating is None or len(rating) == 0:
        return {}

    idx = np.digitize(rating, bin_edges) - 1  # indice bin
    idx = np.clip(idx, 0, len(bin_edges) - 2)

    n_bins = len(bin_edges) - 1
    n = np.zeros(n_bins, dtype=np.int64)
    move_acc = np.full(n_bins, np.nan)
    mate_acc = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = idx == b
        n[b] = mask.sum()
        if n[b] > 0:
            move_acc[b] = move_c[mask].mean()
            mate_acc[b] = mate_c[mask].mean()
    return {"n": n, "move_acc": move_acc, "mate_acc": mate_acc}


# -----------------------------------------------------------------------------
# plot
# -----------------------------------------------------------------------------
def plot_aggregate_bars(res_t: dict, res_u: dict, out_path: str):
    metrics = ["move_acc", "mate_acc"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals_t = [res_t[m] for m in metrics]
    vals_u = [res_u[m] for m in metrics]
    ax.bar(x - width / 2, vals_t, width, label="timed")
    ax.bar(x + width / 2, vals_u, width, label="untimed")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Test set — timed vs untimed (N={res_t['n']})")
    for i, (vt, vu) in enumerate(zip(vals_t, vals_u)):
        ax.text(i - width / 2, vt + 0.01, f"{vt:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, vu + 0.01, f"{vu:.3f}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info(f"Salvato {out_path}")


def plot_per_rating(bin_centers, pr_t, pr_u, metric_key: str, ylabel: str, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(bin_centers, pr_t[metric_key], marker="o", label="timed")
    ax.plot(bin_centers, pr_u[metric_key], marker="s", label="untimed")
    ax.set_xlabel("rating puzzle (centro fascia)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info(f"Salvato {out_path}")


def plot_rating_histogram(rating: np.ndarray, bin_edges: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rating, bins=bin_edges, edgecolor="black")
    ax.set_xlabel("rating puzzle")
    ax.set_ylabel("numero posizioni")
    ax.set_title("Distribuzione test set per fascia rating")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info(f"Salvato {out_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int,
                          title: str, out_path: str):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    # normalizza per riga (recall per classe)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("mate-in-N predetto")
    ax.set_ylabel("mate-in-N reale")
    ax.set_title(title)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black",
                        fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="frazione (row-normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info(f"Salvato {out_path}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Valutazione TimedPolicyGNN sul test set + plot")
    parser.add_argument("--data_dir", type=str, default="dataset/merged")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--plots_dir", type=str, default="plots")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Dove salvare i CSV e il report.")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--lambda_decay", type=float, default=0.01)
    parser.add_argument("--mate_loss_weight", type=float, default=0.3)
    parser.add_argument("--num_mate_classes", type=int, default=8,
                        help="Numero classi mate-in-N usate a training (deve combaciare col modello).")
    parser.add_argument("--rating_bin_width", type=int, default=200,
                        help="Ampiezza di ciascuna fascia di rating.")
    parser.add_argument("--rating_min", type=int, default=600)
    parser.add_argument("--rating_max", type=int, default=2800)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        logger.warning("CUDA non disponibile: valutazione su CPU.")

    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("Carico merged_test.pt...")
    test_positions = load_test_split(args.data_dir)
    logger.info(f"test: {len(test_positions)} posizioni-grafo")

    test_dataset = PuzzleSequenceDataset(test_positions)
    logger.info(f"Sequenze puzzle raggruppate: test={len(test_dataset)}")

    pin_memory = device.type == "cuda"
    persistent = args.num_workers > 0
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=timed_collate_fn, num_workers=args.num_workers,
        pin_memory=pin_memory, persistent_workers=persistent,
    )

    # carica i due best
    timed_ckpt = os.path.join(args.checkpoint_dir, "timed_best.pt")
    untimed_ckpt = os.path.join(args.checkpoint_dir, "untimed_best.pt")
    model_timed = build_and_load_model(timed_ckpt, use_time=True, args=args, device=device)
    model_untimed = build_and_load_model(untimed_ckpt, use_time=False, args=args, device=device)

    logger.info("Valutazione timed...")
    res_t = evaluate(model_timed, test_loader, device, args.mate_loss_weight, args.num_mate_classes)
    logger.info(f"[timed]   loss={res_t['loss']:.4f}  move_acc={res_t['move_acc']:.4f}  "
                f"mate_acc={res_t['mate_acc']:.4f}  N={res_t['n']}")

    logger.info("Valutazione untimed...")
    res_u = evaluate(model_untimed, test_loader, device, args.mate_loss_weight, args.num_mate_classes)
    logger.info(f"[untimed] loss={res_u['loss']:.4f}  move_acc={res_u['move_acc']:.4f}  "
                f"mate_acc={res_u['mate_acc']:.4f}  N={res_u['n']}")

    # ---- CSV aggregato ----
    csv_path = os.path.join(args.out_dir, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "n", "loss", "move_acc", "mate_acc"])
        w.writerow(["timed", res_t["n"], f"{res_t['loss']:.6f}",
                    f"{res_t['move_acc']:.6f}", f"{res_t['mate_acc']:.6f}"])
        w.writerow(["untimed", res_u["n"], f"{res_u['loss']:.6f}",
                    f"{res_u['move_acc']:.6f}", f"{res_u['mate_acc']:.6f}"])
    logger.info(f"Salvato {csv_path}")

    # ---- report testuale ----
    report_path = os.path.join(args.out_dir, "test_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Test set — TimedPolicyGNN vs baseline ===\n\n")
        f.write(f"N posizioni valutate: {res_t['n']}\n\n")
        f.write(f"{'metric':<12}{'timed':>12}{'untimed':>12}{'delta':>12}\n")
        for m in ["loss", "move_acc", "mate_acc"]:
            delta = res_t[m] - res_u[m]
            f.write(f"{m:<12}{res_t[m]:>12.4f}{res_u[m]:>12.4f}{delta:>+12.4f}\n")
    logger.info(f"Salvato {report_path}")

    # ---- plot aggregato ----
    plot_aggregate_bars(res_t, res_u, os.path.join(args.plots_dir, "aggregate_bars.png"))

    # ---- plot per fascia rating (se rating disponibile) ----
    if res_t["rating"] is not None and len(res_t["rating"]) > 0:
        bin_edges = np.arange(args.rating_min, args.rating_max + 1, args.rating_bin_width)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        pr_t = per_rating_metrics(res_t, bin_edges)
        pr_u = per_rating_metrics(res_u, bin_edges)

        plot_per_rating(bin_centers, pr_t, pr_u, "move_acc",
                        "move accuracy",
                        "Move accuracy per fascia rating",
                        os.path.join(args.plots_dir, "per_rating_move.png"))
        plot_per_rating(bin_centers, pr_t, pr_u, "mate_acc",
                        "mate accuracy",
                        "Mate accuracy per fascia rating",
                        os.path.join(args.plots_dir, "per_rating_mate.png"))
        plot_rating_histogram(res_t["rating"], bin_edges,
                              os.path.join(args.plots_dir, "hist_rating.png"))

        # CSV per-rating
        per_rating_csv = os.path.join(args.out_dir, "test_per_rating.csv")
        with open(per_rating_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bin_lo", "bin_hi", "n_timed",
                        "move_acc_timed", "mate_acc_timed",
                        "move_acc_untimed", "mate_acc_untimed"])
            for i in range(len(bin_centers)):
                w.writerow([
                    int(bin_edges[i]), int(bin_edges[i + 1]),
                    int(pr_t["n"][i]),
                    f"{pr_t['move_acc'][i]:.6f}" if not np.isnan(pr_t['move_acc'][i]) else "",
                    f"{pr_t['mate_acc'][i]:.6f}" if not np.isnan(pr_t['mate_acc'][i]) else "",
                    f"{pr_u['move_acc'][i]:.6f}" if not np.isnan(pr_u['move_acc'][i]) else "",
                    f"{pr_u['mate_acc'][i]:.6f}" if not np.isnan(pr_u['mate_acc'][i]) else "",
                ])
        logger.info(f"Salvato {per_rating_csv}")
    else:
        logger.info("Skip plot per-rating: rating non disponibile nel batch.")

    # ---- confusion matrix mate-in-N ----
    plot_confusion_matrix(res_t["mate_true"], res_t["mate_pred"], args.num_mate_classes,
                          "Confusion matrix mate-in-N (timed)",
                          os.path.join(args.plots_dir, "confusion_mate_timed.png"))
    plot_confusion_matrix(res_u["mate_true"], res_u["mate_pred"], args.num_mate_classes,
                          "Confusion matrix mate-in-N (untimed)",
                          os.path.join(args.plots_dir, "confusion_mate_untimed.png"))

    logger.info("Fatto. Guarda la cartella '%s' per i grafici.", args.plots_dir)


if __name__ == "__main__":
    main()
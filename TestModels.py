import csv
import logging
import os
import sys
from types import SimpleNamespace
from typing import List
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader

from Component.PuzzleSequenceDataset import PuzzleSequenceDataset, timed_collate_fn
from Training_TestModel.TimeChainGnn import TimedPolicyGNN
from Training_TestModel.PolicyGNN import legal_move_log_probs, policy_targets_to_global_index
from Training_TestModel.EvaluatorPlotter import EvaluatorPlotter


def load_config(config_path: str = "test.yaml") -> SimpleNamespace:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File di configurazione {config_path} non trovato.")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)


CONFIG = load_config("Yaml/test.yaml")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_evaluator")


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


def load_test_split(data_dir: str):
    path = os.path.join(data_dir, "merged_test.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} non trovato.")
    return torch.load(path, weights_only=False)


def build_and_load_model(ckpt_path: str, use_time: bool, cfg: SimpleNamespace, device: torch.device) -> TimedPolicyGNN:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} non trovato. Hai già eseguito il training?")
    model = TimedPolicyGNN(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        lambda_decay=cfg.lambda_decay,
        use_time=use_time,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Caricato {ckpt_path} (epoca={ckpt.get('epoch','?')}, "
                f"val_move_acc={ckpt.get('best_val_move_acc', 0):.4f}, "
                f"val_mate_acc={ckpt.get('best_val_mate_acc', 0):.4f})")
    return model


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, mate_loss_weight: float) -> dict:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    move_correct_flags: List[np.ndarray] = []
    mate_correct_flags: List[np.ndarray] = []
    mate_true_all: List[np.ndarray] = []
    mate_pred_all: List[np.ndarray] = []
    rating_all: List[np.ndarray] = []
    has_rating = True

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
                    logger.warning("Campo 'rating' non trovato in inner_batch: plot per rating saltati.")

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


def main():
    cfg = CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"GPU attiva: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        logger.warning("CUDA non disponibile: esecuzione su CPU.")

    # Inizializzazione della classe Plotter Esterna
    plotter = EvaluatorPlotter(plots_dir=cfg.plots_dir, out_dir=cfg.out_dir)

    logger.info("Caricamento merged_test.pt...")
    test_positions = load_test_split(cfg.data_dir)
    logger.info(f"Test set caricato: {len(test_positions)} posizioni.")

    test_dataset = PuzzleSequenceDataset(test_positions)
    pin_memory = device.type == "cuda"
    persistent = cfg.num_workers > 0
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=timed_collate_fn, num_workers=cfg.num_workers,
        pin_memory=pin_memory, persistent_workers=persistent,
    )

    timed_ckpt = os.path.join(cfg.checkpoint_dir, "timed_best.pt")
    untimed_ckpt = os.path.join(cfg.checkpoint_dir, "untimed_best.pt")
    model_timed = build_and_load_model(timed_ckpt, use_time=True, cfg=cfg, device=device)
    model_untimed = build_and_load_model(untimed_ckpt, use_time=False, cfg=cfg, device=device)

    logger.info("Valutazione modello timed...")
    res_t = evaluate(model_timed, test_loader, device, cfg.mate_loss_weight)
    logger.info(f"[Timed]   Loss={res_t['loss']:.4f} | Move Acc={res_t['move_acc']*100:.2f}% | "
                f"Mate Acc={res_t['mate_acc']*100:.2f}% | N={res_t['n']}")

    logger.info("Valutazione modello untimed...")
    res_u = evaluate(model_untimed, test_loader, device, cfg.mate_loss_weight)
    logger.info(f"[Untimed] Loss={res_u['loss']:.4f} | Move Acc={res_u['move_acc']*100:.2f}% | "
                f"Mate Acc={res_u['mate_acc']*100:.2f}% | N={res_u['n']}")

    # Salvataggio CSV generale delle metriche
    csv_path = os.path.join(cfg.out_dir, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "n", "loss", "move_acc", "mate_acc"])
        w.writerow(["timed", res_t["n"], f"{res_t['loss']:.6f}",
                    f"{res_t['move_acc']:.6f}", f"{res_t['mate_acc']:.6f}"])
        w.writerow(["untimed", res_u["n"], f"{res_u['loss']:.6f}",
                    f"{res_u['move_acc']:.6f}", f"{res_u['mate_acc']:.6f}"])
    logger.info(f"Salvato {csv_path}")

    # =========================================================================
    # GENERAZIONE DEI PLOT E DELLE METRICHE RICHIESTE
    # =========================================================================

    # 1. Barre verticali con testo (corrette / totale) per ogni n
    plotter.plot_depth_bars(res_t, res_u, max_n=10, filename="bars_per_n.png")

    # 2. Curve per ogni n
    plotter.plot_depth_curves(res_t, res_u, max_n=10, filename="curves_per_n.png")

    # 3. Metriche salvate su file CSV per ogni n
    plotter.save_depth_metrics(res_t, res_u, max_n=10, filename="metrics_per_n.csv")

    # Grafici ausiliari
    plotter.plot_aggregate_bars(res_t, res_u, filename="aggregate_bars.png")
    plotter.plot_confusion_matrix(res_t["mate_true"], res_t["mate_pred"], cfg.num_mate_classes,
                                  "Confusion Matrix Mate-in-N (Timed)", "confusion_mate_timed.png")
    plotter.plot_confusion_matrix(res_u["mate_true"], res_u["mate_pred"], cfg.num_mate_classes,
                                  "Confusion Matrix Mate-in-N (Untimed)", "confusion_mate_untimed.png")

    logger.info("Valutazione completata. Risultati e grafici salvati con successo.")


if __name__ == "__main__":
    main()
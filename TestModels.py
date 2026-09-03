import csv
import logging
import os
import sys
from types import SimpleNamespace
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader

from Training_TestModel.PolicyGNN import legal_move_log_probs, policy_targets_to_global_index
from ModelUtils.EvaluatorPlotter import EvaluatorPlotter
from ModelUtils.Utils import load_model, load_dataset_from_pt, predict_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_evaluator")


def load_config(config_path: str = "test.yaml") -> SimpleNamespace:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File di configurazione {config_path} non trovato.")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)


CONFIG = load_config("Yaml/test.yaml")


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, mate_loss_weight: float) -> dict:
    """
    Valuta il modello sul dataloader e restituisce metriche aggregate.
    """
    model.eval()
    total_loss = 0.0
    total_examples = 0

    move_correct_flags = []
    mate_correct_flags = []
    mate_true_all = []
    mate_pred_all = []
    rating_all = []
    has_rating = True

    with torch.no_grad():
        for batch in loader:
            inner_batch, chain_edge_index, chain_edge_attr = batch
            inner_batch = inner_batch.to(device, non_blocking=True)
            chain_edge_index = chain_edge_index.to(device, non_blocking=True)
            chain_edge_attr = chain_edge_attr.to(device, non_blocking=True)
            num_graphs = inner_batch.num_graphs

            # Usa predict_batch per ottenere predizioni e logit
            preds = predict_batch(model, batch, device)
            move_pred = preds["move_pred"].to(device)  # già su CPU, spostiamo per i calcoli
            move_scores = preds["move_scores"].to(device)
            edge_batch = preds["edge_batch"].to(device)
            mate_logits = preds["mate_logits"].to(device)

            # Calcolo delle loss (necessario per la metrica loss)
            log_probs = legal_move_log_probs(move_scores, edge_batch, num_graphs)
            target_idx = policy_targets_to_global_index(edge_batch, inner_batch.y, num_graphs)
            policy_loss = -log_probs[target_idx].mean()

            mate_target = inner_batch.mate_n.clamp(0, mate_logits.size(-1) - 1)
            mate_loss = torch.nn.functional.cross_entropy(mate_logits, mate_target)
            loss = policy_loss + mate_loss_weight * mate_loss

            # Accuratezza move
            move_correct = (move_pred == target_idx).cpu().numpy()
            move_correct_flags.append(move_correct)

            # Accuratezza mate
            mate_pred = preds["mate_pred"]  # già su CPU
            mate_correct = (mate_pred == mate_target.cpu()).numpy()
            mate_correct_flags.append(mate_correct)
            mate_true_all.append(mate_target.cpu().numpy())
            mate_pred_all.append(mate_pred.numpy())

            # Rating (se presente)
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
        "mate_n": mate_true_np,  # alias per compatibilità
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

    # Caricamento del dataset di test tramite il modulo utils
    test_pt = os.path.join(cfg.data_dir, "merged_test.pt")
    logger.info(f"Caricamento {test_pt}...")
    test_loader = load_dataset_from_pt(
        test_pt,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device
    )
    logger.info(f"Test set caricato: {len(test_loader.dataset)} posizioni.")

    # Caricamento modelli
    timed_ckpt = os.path.join(cfg.checkpoint_dir, "timed_best.pt")
    untimed_ckpt = os.path.join(cfg.checkpoint_dir, "untimed_best.pt")

    model_timed = load_model(
        timed_ckpt,
        use_time=True,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        lambda_decay=cfg.lambda_decay,
        device=device,
    )
    model_untimed = load_model(
        untimed_ckpt,
        use_time=False,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        lambda_decay=cfg.lambda_decay,
        device=device,
    )

    # Inizializzazione plotter
    plotter = EvaluatorPlotter(plots_dir=cfg.plots_dir, out_dir=cfg.out_dir)

    # Valutazione
    logger.info("Valutazione modello timed...")
    res_t = evaluate(model_timed, test_loader, device, cfg.mate_loss_weight)
    logger.info(f"[Timed]   Loss={res_t['loss']:.4f} | Move Acc={res_t['move_acc']*100:.2f}% | "
                f"Mate Acc={res_t['mate_acc']*100:.2f}% | N={res_t['n']}")

    logger.info("Valutazione modello untimed...")
    res_u = evaluate(model_untimed, test_loader, device, cfg.mate_loss_weight)
    logger.info(f"[Untimed] Loss={res_u['loss']:.4f} | Move Acc={res_u['move_acc']*100:.2f}% | "
                f"Mate Acc={res_u['mate_acc']*100:.2f}% | N={res_u['n']}")

    # Salvataggio CSV metriche generali
    csv_path = os.path.join(cfg.out_dir, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "n", "loss", "move_acc", "mate_acc"])
        w.writerow(["timed", res_t["n"], f"{res_t['loss']:.6f}",
                    f"{res_t['move_acc']:.6f}", f"{res_t['mate_acc']:.6f}"])
        w.writerow(["untimed", res_u["n"], f"{res_u['loss']:.6f}",
                    f"{res_u['move_acc']:.6f}", f"{res_u['mate_acc']:.6f}"])
    logger.info(f"Salvato {csv_path}")

    # Generazione plot e metriche per profondità
    plotter.plot_depth_bars(res_t, res_u, max_n=10, filename="bars_per_n.png")
    plotter.plot_depth_curves(res_t, res_u, max_n=10, filename="curves_per_n.png")
    plotter.save_depth_metrics(res_t, res_u, max_n=10, filename="metrics_per_n.csv")
    plotter.plot_aggregate_bars(res_t, res_u, filename="aggregate_bars.png")
    plotter.plot_confusion_matrix(res_t["mate_true"], res_t["mate_pred"], cfg.num_mate_classes,
                                  "Confusion Matrix Mate-in-N (Timed)", "confusion_mate_timed.png")
    plotter.plot_confusion_matrix(res_u["mate_true"], res_u["mate_pred"], cfg.num_mate_classes,
                                  "Confusion Matrix Mate-in-N (Untimed)", "confusion_mate_untimed.png")

    logger.info("Valutazione completata. Risultati e grafici salvati con successo.")


if __name__ == "__main__":
    main()
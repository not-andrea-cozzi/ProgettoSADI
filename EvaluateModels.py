import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from ModelUtils.PolicyGNN import legal_move_log_probs, policy_targets_to_global_index
from ModelUtils.TimeChainGnn import TimedPolicyGNN
from Component.PuzzleSequenceDataset import PuzzleSequenceDataset, timed_collate_fn
from ModelUtils.EvaluatorPlotter import EvaluatorPlotter
from ModelUtils.Utils import _argmax_per_graph
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("unified_evaluator")


def load_config(config_path: str) -> argparse.Namespace:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File di configurazione {config_path} non trovato.")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)


def dataset_tag(dataset_path: str) -> str:
    """Ricava un tag leggibile dal path del dataset, da usare come suffisso
    nei nomi dei file di output (es. 'games_untimed_test_one' da
    'Dataset/Games_v2_untimed/games_untimed_test_one.pt'), cosi' run su
    dataset diversi non si sovrascrivono a vicenda in out_dir."""
    base = os.path.basename(dataset_path)
    name, _ = os.path.splitext(base)
    return name


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
        for inner_batch, chain_edge_index, chain_edge_attr in loader:
            inner_batch = inner_batch.to(device, non_blocking=True)
            chain_edge_index = chain_edge_index.to(device, non_blocking=True)
            chain_edge_attr = chain_edge_attr.to(device, non_blocking=True)
            num_graphs = inner_batch.num_graphs

            # Inferenza esplicita (come in TrainModels.py)
            move_scores, edge_batch, mate_logits = model(inner_batch, chain_edge_index, chain_edge_attr)

            log_probs = legal_move_log_probs(move_scores, edge_batch, num_graphs)
            target_idx = policy_targets_to_global_index(edge_batch, inner_batch.y, num_graphs)
            policy_loss = -log_probs[target_idx].mean()

            mate_target = inner_batch.mate_n.clamp(0, mate_logits.size(-1) - 1)
            mate_loss = torch.nn.functional.cross_entropy(mate_logits, mate_target)
            loss = policy_loss + mate_loss_weight * mate_loss

            # Move Accuracy
            move_pred = _argmax_per_graph(move_scores, edge_batch, num_graphs)
            move_correct = (move_pred == target_idx).cpu().numpy()
            move_correct_flags.append(move_correct)

            # Mate Accuracy
            mate_pred = mate_logits.argmax(dim=-1)
            mate_correct = (mate_pred == mate_target).cpu().numpy()
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

            # Pulizia VRAM
            del inner_batch, chain_edge_index, chain_edge_attr, move_scores, edge_batch, mate_logits, loss

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
        "mate_n": mate_true_np,
        "rating": rating_np,
    }


def main():
    parser = argparse.ArgumentParser(description="Script unificato per la valutazione dei modelli Timed e Untimed.")
    parser.add_argument("--config", type=str, default="Yaml/evaluate.yaml", help="Percorso al file di configurazione YAML")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        logger.info(f"Configurazione caricata da {args.config}")
    except Exception as e:
        logger.error(f"Errore nel caricamento della configurazione da {args.config}: {e}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"GPU attiva: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        logger.warning("CUDA non disponibile: esecuzione su CPU.")

    # Caricamento esplicito del dataset per allineamento col TrainModels
    test_pt = getattr(cfg, "dataset_path", None)
    if not test_pt:
        test_pt = os.path.join(getattr(cfg, "data_dir", ""), "merged_test.pt")

    logger.info(f"Caricamento dataset {test_pt}...")
    test_positions = torch.load(test_pt, weights_only=False)
    test_dataset = PuzzleSequenceDataset(test_positions)

    use_persistent = getattr(cfg, "num_workers", 0) > 0
    prefetch = 2 if use_persistent else None

    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=timed_collate_fn, num_workers=getattr(cfg, "num_workers", 0),
        pin_memory=(device.type == "cuda"), persistent_workers=use_persistent,
        prefetch_factor=prefetch
    )
    logger.info(f"Test set caricato: {len(test_dataset)} posizioni.")

    timed_ckpt = os.path.join(cfg.checkpoint_dir, "timed_best.pt")
    untimed_ckpt = os.path.join(cfg.checkpoint_dir, "untimed_best.pt")

    # Caricamento esplicito del modello Timed
    model_timed = TimedPolicyGNN(
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        lambda_decay=cfg.lambda_decay, use_time=True
    ).to(device)
    ckpt_t = torch.load(timed_ckpt, map_location=device, weights_only=False)
    model_timed.load_state_dict(ckpt_t["model_state"])

    # Caricamento esplicito del modello Untimed
    model_untimed = TimedPolicyGNN(
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        lambda_decay=cfg.lambda_decay, use_time=False
    ).to(device)
    ckpt_u = torch.load(untimed_ckpt, map_location=device, weights_only=False)
    model_untimed.load_state_dict(ckpt_u["model_state"])

    
    os.makedirs(cfg.out_dir, exist_ok=True)
    tag = dataset_tag(test_pt)
    plotter = EvaluatorPlotter(plots_dir=cfg.out_dir, out_dir=cfg.out_dir)

    logger.info("Valutazione modello timed...")
    res_t = evaluate(model_timed, test_loader, device, cfg.mate_loss_weight)
    logger.info(f"[Timed]   Loss={res_t['loss']:.4f} | Move Acc={res_t['move_acc']*100:.2f}% | Mate Acc={res_t['mate_acc']*100:.2f}% | N={res_t['n']}")

    logger.info("Valutazione modello untimed...")
    res_u = evaluate(model_untimed, test_loader, device, cfg.mate_loss_weight)
    logger.info(f"[Untimed] Loss={res_u['loss']:.4f} | Move Acc={res_u['move_acc']*100:.2f}% | Mate Acc={res_u['mate_acc']*100:.2f}% | N={res_u['n']}")

    csv_path = os.path.join(cfg.out_dir, f"test_metrics_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "n", "loss", "move_acc", "mate_acc"])
        w.writerow(["timed", res_t["n"], f"{res_t['loss']:.6f}", f"{res_t['move_acc']:.6f}", f"{res_t['mate_acc']:.6f}"])
        w.writerow(["untimed", res_u["n"], f"{res_u['loss']:.6f}", f"{res_u['move_acc']:.6f}", f"{res_u['mate_acc']:.6f}"])
    logger.info(f"Salvato {csv_path}")

    max_n = getattr(cfg, "max_depth", 10)
    plotter.plot_depth_bars(res_t, res_u, max_n=max_n, filename=f"bars_per_n_{tag}.png")
    plotter.plot_depth_curves(res_t, res_u, max_n=max_n, filename=f"curves_per_n_{tag}.png")
    plotter.save_depth_metrics(res_t, res_u, max_n=max_n, filename=f"metrics_per_n_{tag}.csv")
    plotter.plot_aggregate_bars(res_t, res_u, filename=f"aggregate_bars_{tag}.png")
    plotter.plot_confusion_matrix(
        res_t["mate_true"], res_t["mate_pred"], cfg.num_mate_classes,
        "Confusion Matrix Mate-in-N (Timed)", f"confusion_mate_timed_{tag}.png"
    )
    plotter.plot_confusion_matrix(
        res_u["mate_true"], res_u["mate_pred"], cfg.num_mate_classes,
        "Confusion Matrix Mate-in-N (Untimed)", f"confusion_mate_untimed_{tag}.png"
    )

    logger.info("Valutazione completata. Risultati e grafici salvati con successo.")


if __name__ == "__main__":
    main()
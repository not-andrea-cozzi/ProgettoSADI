import csv
import logging
import os
import sys
from ModelUtils.Utils import (
    load_config_as_namespace,
    load_dataset_from_pt,
    load_model,
    evaluate_model,
)
import torch

from ModelUtils.EvaluatorPlotter import EvaluatorPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_model")

CONFIG_PATH = "Yaml/evaluate.yaml"  # o il percorso che preferisci


def main():
    cfg = load_config_as_namespace(CONFIG_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"GPU attiva: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        logger.warning("CUDA non disponibile: esecuzione su CPU.")

    # 1. Caricamento dataset (il percorso viene dal YAML)
    test_pt = cfg.dataset_path  # es. "Dataset/Validation/validator_test.pt"
    logger.info(f"Caricamento {test_pt}...")
    test_loader = load_dataset_from_pt(
        test_pt,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
    )
    logger.info(f"Test set caricato: {len(test_loader.dataset)} posizioni.")

    # 2. Caricamento modelli (checkpoint e iperparametri dal YAML)
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

    # 3. Plotter
    plotter = EvaluatorPlotter(plots_dir=cfg.plots_dir, out_dir=cfg.out_dir)

    # 4. Valutazione
    logger.info("Valutazione modello timed...")
    res_t = evaluate_model(model_timed, test_loader, device, cfg.mate_loss_weight)
    logger.info(f"[Timed]   Loss={res_t['loss']:.4f} | Move Acc={res_t['move_acc']*100:.2f}% | "
                f"Mate Acc={res_t['mate_acc']*100:.2f}% | N={res_t['n']}")

    logger.info("Valutazione modello untimed...")
    res_u = evaluate_model(model_untimed, test_loader, device, cfg.mate_loss_weight)
    logger.info(f"[Untimed] Loss={res_u['loss']:.4f} | Move Acc={res_u['move_acc']*100:.2f}% | "
                f"Mate Acc={res_u['mate_acc']*100:.2f}% | N={res_u['n']}")

    # 5. Salvataggio CSV metriche
    csv_path = os.path.join(cfg.out_dir, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "n", "loss", "move_acc", "mate_acc"])
        w.writerow(["timed", res_t["n"], f"{res_t['loss']:.6f}",
                    f"{res_t['move_acc']:.6f}", f"{res_t['mate_acc']:.6f}"])
        w.writerow(["untimed", res_u["n"], f"{res_u['loss']:.6f}",
                    f"{res_u['move_acc']:.6f}", f"{res_u['mate_acc']:.6f}"])
    logger.info(f"Salvato {csv_path}")

    # 6. Generazione plot
    max_n = getattr(cfg, "max_depth", 10)
    plotter.plot_depth_bars(res_t, res_u, max_n=max_n, filename="bars_per_n.png")
    plotter.plot_depth_curves(res_t, res_u, max_n=max_n, filename="curves_per_n.png")
    plotter.save_depth_metrics(res_t, res_u, max_n=max_n, filename="metrics_per_n.csv")
    plotter.plot_aggregate_bars(res_t, res_u, filename="aggregate_bars.png")
    plotter.plot_confusion_matrix(
        res_t["mate_true"], res_t["mate_pred"], cfg.num_mate_classes,
        "Confusion Matrix Mate-in-N (Timed)", "confusion_mate_timed.png"
    )
    plotter.plot_confusion_matrix(
        res_u["mate_true"], res_u["mate_pred"], cfg.num_mate_classes,
        "Confusion Matrix Mate-in-N (Untimed)", "confusion_mate_untimed.png"
    )

    logger.info("Valutazione completata. Risultati e grafici salvati con successo.")


if __name__ == "__main__":
    main()
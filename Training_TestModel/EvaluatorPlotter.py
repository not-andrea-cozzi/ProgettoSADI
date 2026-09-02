import csv
import logging
import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("evaluator_plotter")


class EvaluatorPlotter:
    """
    Classe helper per calcolare metriche e generare grafici/CSV per la valutazione dei modelli.
    """

    def __init__(self, plots_dir: str, out_dir: str):
        self.plots_dir = plots_dir
        self.out_dir = out_dir
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def per_mate_depth_metrics(results: dict, max_n: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calcola accuracy, totale ed esatti per ogni n (profondità di matto)."""
        mate_true = results["mate_true"]
        move_c = results["move_correct"]

        depths = np.arange(1, max_n + 1)
        accs = np.full(max_n, np.nan)
        counts = np.zeros(max_n, dtype=np.int64)
        corrects = np.zeros(max_n, dtype=np.int64)

        for i, d in enumerate(depths):
            mask = mate_true == d
            counts[i] = mask.sum()
            if counts[i] > 0:
                corrects[i] = move_c[mask].sum()
                accs[i] = corrects[i] / counts[i]

        return depths, accs, counts, corrects

    @staticmethod
    def per_rating_metrics(results: dict, bin_edges: np.ndarray) -> Dict[str, np.ndarray]:
        """Calcola le metriche suddivise per fascia di rating."""
        rating = results["rating"]
        move_c = results["move_correct"]
        mate_c = results["mate_correct"]

        if rating is None or len(rating) == 0:
            return {}

        idx = np.digitize(rating, bin_edges) - 1
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

    # -------------------------------------------------------------------------
    # METODI DI PLOTTING E SALVATAGGIO
    # -------------------------------------------------------------------------

    def plot_depth_bars(self, res_t: dict, res_u: dict, max_n: int = 10, filename: str = "bars_per_n.png"):
        """Plot 1: Barre verticali per vedere risposte giuste / totale per ogni n."""
        depths, acc_t, counts, corrects_t = self.per_mate_depth_metrics(res_t, max_n=max_n)
        _, acc_u, _, corrects_u = self.per_mate_depth_metrics(res_u, max_n=max_n)

        width = 0.35
        x = np.arange(len(depths))
        fig, ax = plt.subplots(figsize=(10, 6))

        rects1 = ax.bar(x - width / 2, acc_t * 100, width, label="Timed", color="royalblue")
        rects2 = ax.bar(x + width / 2, acc_u * 100, width, label="Untimed", color="coral")

        ax.set_xticks(x)
        ax.set_xticklabels([f"n={d}" for d in depths])
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Profondità di matto (n)")
        ax.set_title("Risposte Giuste / Totale per ogni n")
        ax.set_ylim(0, 115)  # Spazio superiore per le etichette di testo
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        # Aggiunta etichette (es. 45/50) sopra le barre
        for i, rect in enumerate(rects1):
            if counts[i] > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2,
                        f"{corrects_t[i]}/{counts[i]}", ha="center", va="bottom", fontsize=8, rotation=90)

        for i, rect in enumerate(rects2):
            if counts[i] > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2,
                        f"{corrects_u[i]}/{counts[i]}", ha="center", va="bottom", fontsize=8, rotation=90)

        fig.tight_layout()
        out_path = os.path.join(self.plots_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info(f"Salvato {out_path}")

    def plot_depth_curves(self, res_t: dict, res_u: dict, max_n: int = 10, filename: str = "curves_per_n.png"):
        """Plot 2: Curve di accuracy al variare di n."""
        depths, acc_t, _, _ = self.per_mate_depth_metrics(res_t, max_n=max_n)
        _, acc_u, _, _ = self.per_mate_depth_metrics(res_u, max_n=max_n)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(depths, acc_t * 100, marker="o", label="Timed", color="royalblue")
        ax.plot(depths, acc_u * 100, marker="s", label="Untimed", color="coral")
        ax.set_xticks(depths)
        ax.set_xlabel("Profondità di matto (n)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Andamento Accuracy per ogni n")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = os.path.join(self.plots_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info(f"Salvato {out_path}")

    def save_depth_metrics(self, res_t: dict, res_u: dict, max_n: int = 10, filename: str = "metrics_per_n.csv"):
        """Plot 3 (Metriche): Salvataggio su file CSV dei dati esatti divisi per n."""
        depths, acc_t, counts, corrects_t = self.per_mate_depth_metrics(res_t, max_n=max_n)
        _, acc_u, _, corrects_u = self.per_mate_depth_metrics(res_u, max_n=max_n)

        out_path = os.path.join(self.out_dir, filename)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "totale", "corrette_timed", "acc_timed", "corrette_untimed", "acc_untimed"])
            for i in range(len(depths)):
                w.writerow([
                    depths[i], counts[i],
                    corrects_t[i], f"{acc_t[i]:.6f}" if not np.isnan(acc_t[i]) else "",
                    corrects_u[i], f"{acc_u[i]:.6f}" if not np.isnan(acc_u[i]) else ""
                ])
        logger.info(f"Salvato {out_path}")

    # --- Altri plot ausiliari ---

    def plot_aggregate_bars(self, res_t: dict, res_u: dict, filename: str = "aggregate_bars.png"):
        metrics = ["move_acc", "mate_acc"]
        x = np.arange(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 4.5))
        vals_t = [res_t[m] for m in metrics]
        vals_u = [res_u[m] for m in metrics]
        ax.bar(x - width / 2, vals_t, width, label="Timed", color="royalblue")
        ax.bar(x + width / 2, vals_u, width, label="Untimed", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Test set — Timed vs Untimed (N={res_t['n']})")
        for i, (vt, vu) in enumerate(zip(vals_t, vals_u)):
            ax.text(i - width / 2, vt + 0.01, f"{vt:.3f}", ha="center", fontsize=9)
            ax.text(i + width / 2, vu + 0.01, f"{vu:.3f}", ha="center", fontsize=9)
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(self.plots_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info(f"Salvato {out_path}")

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, title: str, filename: str):
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if int(t) < num_classes and int(p) < num_classes:
                cm[int(t), int(p)] += 1
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xlabel("Mate-in-N predetto")
        ax.set_ylabel("Mate-in-N reale")
        ax.set_title(title)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                if cm[i, j] > 0:
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                            ha="center", va="center",
                            color="white" if cm_norm[i, j] > 0.5 else "black",
                            fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Frazione (row-normalized)")
        fig.tight_layout()
        out_path = os.path.join(self.plots_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info(f"Salvato {out_path}")
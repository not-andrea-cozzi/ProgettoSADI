import os
import json
import csv
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from Training_TestModel.PolicyGNN import legal_move_log_probs, policy_targets_to_global_index
from TrainModels import _argmax_per_graph


class TrainingMetricsLogger:
    """
    Gestisce la registrazione e il plotting delle metriche di training e validation
    per il confronto tra la variante Timed e Untimed.
    """
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Struttura dati per le metriche per configurazione
        self.history: Dict[str, Dict[str, List[float]]] = {
            "timed": {
                "train_loss": [], "val_loss": [],
                "train_move_acc": [], "val_move_acc": [],
                "train_mate_acc": [], "val_mate_acc": []
            },
            "untimed": {
                "train_loss": [], "val_loss": [],
                "train_move_acc": [], "val_move_acc": [],
                "train_mate_acc": [], "val_mate_acc": []
            }
        }

    def log_epoch(
        self,
        tag: str,
        epoch: int,
        train_loss: float,
        train_move_acc: float,
        train_mate_acc: float,
        val_loss: float,
        val_move_acc: float,
        val_mate_acc: float
    ) -> None:
        """Registra i valori scalari per l'epoca corrente."""
        if tag not in self.history:
            self.history[tag] = {k: [] for k in self.history["timed"].keys()}

        self.history[tag]["train_loss"].append(train_loss)
        self.history[tag]["val_loss"].append(val_loss)
        self.history[tag]["train_move_acc"].append(train_move_acc)
        self.history[tag]["val_move_acc"].append(val_move_acc)
        self.history[tag]["train_mate_acc"].append(train_mate_acc)
        self.history[tag]["val_mate_acc"].append(val_mate_acc)

    def save_metrics_to_disk(self) -> None:
        """Esporta la cronologia completa in JSON e CSV."""
        json_path = os.path.join(self.output_dir, "training_history.json")
        with open(json_path, "w") as f:
            json.dump(self.history, f, indent=4)

        # Esportazione CSV
        csv_path = os.path.join(self.output_dir, "training_history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tag", "epoch", "train_loss", "val_loss", "train_move_acc", "val_move_acc", "train_mate_acc", "val_mate_acc"])
            for tag, metrics in self.history.items():
                num_epochs = len(metrics["train_loss"])
                for ep in range(num_epochs):
                    writer.writerow([
                        tag, ep + 1,
                        metrics["train_loss"][ep], metrics["val_loss"][ep],
                        metrics["train_move_acc"][ep], metrics["val_move_acc"][ep],
                        metrics["train_mate_acc"][ep], metrics["val_mate_acc"][ep]
                    ])

    def plot_training_curves(self) -> None:
        """Genera e salva i grafici comparativi (Loss, Move Accuracy, Mate Accuracy)."""
        metrics_to_plot = [
            ("loss", "Loss", ["train_loss", "val_loss"]),
            ("move_acc", "Move Prediction Accuracy", ["train_move_acc", "val_move_acc"]),
            ("mate_acc", "Mate-in-N Classification Accuracy", ["train_mate_acc", "val_mate_acc"])
        ]

        for file_suffix, title, keys in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            for tag, color in [("timed", "royalblue"), ("untimed", "coral")]:
                if not self.history[tag][keys[0]]:
                    continue
                epochs = range(1, len(self.history[tag][keys[0]]) + 1)
                ax.plot(epochs, self.history[tag][keys[0]], label=f"{tag.capitalize()} Train", linestyle="--", color=color, alpha=0.7)
                ax.plot(epochs, self.history[tag][keys[1]], label=f"{tag.capitalize()} Val", linestyle="-", color=color, linewidth=2.0)

            ax.set_title(f"Comparison: {title}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend()
            
            out_file = os.path.join(self.plots_dir, f"comparison_{file_suffix}.png")
            fig.tight_layout()
            fig.savefig(out_file, dpi=300)
            plt.close(fig)


class StratifiedEvaluator:
    """
    Valuta i modelli su un held-out test set stratificando le metriche
    in base alla profondità del matto (n da 1 a 10).
    Permette inoltre di integrare i risultati di un LLM baseline.
    """
    def __init__(self, device: torch.device, output_dir: str = "results/eval"):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @torch.no_grad()
    def evaluate_stratified_gnn(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        max_n: int = 10
    ) -> Dict[int, Dict[str, float]]:
        """
        Valuta il modello GNN calcolando Accuracy di mossa e di matto per ciascun valore di n.
        """
        model.eval()
        
        # Statistiche per profondità n
        correct_moves = {n: 0 for n in range(1, max_n + 1)}
        correct_mates = {n: 0 for n in range(1, max_n + 1)}
        total_counts = {n: 0 for n in range(1, max_n + 1)}

        for inner_batch, chain_edge_index, chain_edge_attr in test_loader:
            inner_batch = inner_batch.to(self.device, non_blocking=True)
            chain_edge_index = chain_edge_index.to(self.device, non_blocking=True)
            chain_edge_attr = chain_edge_attr.to(self.device, non_blocking=True)
            num_graphs = inner_batch.num_graphs

            move_scores, edge_batch, mate_logits = model(inner_batch, chain_edge_index, chain_edge_attr)
            target_idx = policy_targets_to_global_index(edge_batch, inner_batch.y, num_graphs)

            move_pred_global = _argmax_per_graph(move_scores, edge_batch, num_graphs)
            is_move_correct = (move_pred_global == target_idx).cpu().numpy()

            mate_target = inner_batch.mate_n.clamp(0, mate_logits.size(-1) - 1)
            is_mate_correct = (mate_logits.argmax(dim=-1) == mate_target).cpu().numpy()

            # Raggruppamento per n
            mate_n_values = inner_batch.mate_n.cpu().numpy()
            for i, n_val in enumerate(mate_n_values):
                if 1 <= n_val <= max_n:
                    correct_moves[n_val] += int(is_move_correct[i])
                    correct_mates[n_val] += int(is_mate_correct[i])
                    total_counts[n_val] += 1

        # Calcolo percentuali
        stratified_results = {}
        for n in range(1, max_n + 1):
            count = total_counts[n]
            stratified_results[n] = {
                "count": count,
                "move_acc": (correct_moves[n] / count) if count > 0 else 0.0,
                "mate_acc": (correct_mates[n] / count) if count > 0 else 0.0
            }

        return stratified_results

    def plot_and_save_stratified_comparison(
        self,
        timed_results: Dict[int, Dict[str, float]],
        untimed_results: Dict[int, Dict[str, float]],
        llm_results: Optional[Dict[int, float]] = None,
        max_n: int = 10
    ) -> None:
        """
        Salva i risultati stratificati in formato JSON e genera il Bar Chart
        comparativo (GNN Timed vs GNN Untimed vs LLM Baseline) per ogni valore di n.
        """
        # 1. Esportazione JSON
        combined_data = {
            "timed": timed_results,
            "untimed": untimed_results,
            "llm": llm_results or {}
        }
        with open(os.path.join(self.output_dir, "stratified_comparison.json"), "w") as f:
            json.dump(combined_data, f, indent=4)

        # 2. Bar Chart per Move Accuracy per n
        n_values = np.arange(1, max_n + 1)
        timed_acc = [timed_results[n]["move_acc"] * 100 for n in n_values]
        untimed_acc = [untimed_results[n]["move_acc"] * 100 for n in n_values]

        width = 0.25 if llm_results is not None else 0.35
        fig, ax = plt.subplots(figsize=(10, 6))

        r1 = np.arange(len(n_values))
        r2 = [x + width for x in r1]
        
        ax.bar(r1, timed_acc, color="royalblue", width=width, edgecolor="grey", label="Timed GNN")
        ax.bar(r2, untimed_acc, color="coral", width=width, edgecolor="grey", label="Untimed GNN")

        if llm_results is not None:
            r3 = [x + width for x in r2]
            llm_acc = [llm_results.get(n, 0.0) * 100 for n in n_values]
            ax.bar(r3, llm_acc, color="mediumseagreen", width=width, edgecolor="grey", label="LLM Baseline")
            ax.set_xticks([r + width for r in range(len(n_values))])
        else:
            ax.set_xticks([r + width / 2 for r in range(len(n_values))])

        ax.set_xticklabels([f"Mate in {n}" for n in n_values])
        ax.set_xlabel("Depth of Mate (n)", fontweight="bold")
        ax.set_ylabel("Move Accuracy (%)", fontweight="bold")
        ax.set_title("Puzzle-Solving Accuracy Stratified by Mate Depth (n)")
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.7)

        fig.tight_layout()
        plot_path = os.path.join(self.output_dir, "stratified_accuracy_by_n.png")
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
import logging
import os
import random
import sys
from types import SimpleNamespace
from typing import Optional, Tuple
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader

from timegnn.train.early_stopping import EarlyStopping
from Component.PuzzleSequenceDataset import PuzzleSequenceDataset, timed_collate_fn
from ModelUtils.PolicyGNN import legal_move_log_probs, policy_targets_to_global_index
from ModelUtils.TimeChainGnn import TimedPolicyGNN
from ModelUtils.MetricLogger import TrainingMetricsLogger
from ModelUtils.Utils import _argmax_per_graph 


def load_config(config_path: str = "train.yaml") -> SimpleNamespace:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File di configurazione {config_path} non trovato.")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)


CONFIG = load_config("Yaml/train.yaml")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("trainer")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(data_dir: str, name: str):
    path = os.path.join(data_dir, f"merged_{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} non trovato. Verifica il percorso dati.")
    return torch.load(path, weights_only=False)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    mate_loss_weight: float,
    train: bool,
    grad_clip: Optional[float] = 1.0,
) -> Tuple[float, float, float]:
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
                assert optimizer is not None, "Optimizer richiesto durante il training."
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        with torch.no_grad():
            move_pred_global = _argmax_per_graph(move_scores, edge_batch, num_graphs)
            move_pred_is_best = move_pred_global == target_idx
            mate_pred_correct = mate_logits.argmax(dim=-1) == mate_target

            total_move_correct += move_pred_is_best.sum().item()
            total_mate_correct += mate_pred_correct.sum().item()
            total_examples += num_graphs
            total_loss += loss.item() * num_graphs

        # Ottimizzazione memoria: pulizia esplicita dei tensori pesanti a fine iterazione
        del loss, move_scores, mate_logits, inner_batch, chain_edge_index, chain_edge_attr

    return (
        total_loss / total_examples,
        total_move_correct / total_examples,
        total_mate_correct / total_examples,
    )


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, best_move_acc: float, best_mate_acc: float, early_stopper: EarlyStopping) -> None:
    tmp_path = f"{path}.tmp"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_move_acc": best_move_acc,
        "best_val_mate_acc": best_mate_acc,
        "es_counter": early_stopper.counter,
        "es_best_loss": early_stopper.best_loss,
    }, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device, early_stopper: EarlyStopping) -> Tuple[int, float, float]:
    if not os.path.exists(path):
        return 0, 0.0, 0.0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    early_stopper.counter = ckpt.get("es_counter", 0)
    early_stopper.best_loss = ckpt.get("es_best_loss", float("inf"))
    return ckpt["epoch"], ckpt["best_val_move_acc"], ckpt["best_val_mate_acc"]


def train_one_config(
    use_time: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: SimpleNamespace,
    metrics_logger: TrainingMetricsLogger
) -> Tuple[torch.nn.Module, float, float]:
    tag = "timed" if use_time else "untimed"
    set_seed(cfg.seed)

    model = TimedPolicyGNN(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        lambda_decay=cfg.lambda_decay,
        use_time=use_time,
    ).to(device)

    if cfg.compile:
        try:
            model = torch.compile(model)
            logger.info(f"[{tag}] torch.compile abilitato.")
        except Exception as err:
            logger.warning(f"[{tag}] torch.compile non riuscito ({err}), fallback su interprete standard.")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    early_stopper = EarlyStopping(patience=cfg.patience, delta=0.0)

    latest_path = os.path.join(cfg.checkpoint_dir, f"{tag}_latest.pt")
    best_path = os.path.join(cfg.checkpoint_dir, f"{tag}_best.pt")

    start_epoch, best_move_acc, best_mate_acc = load_checkpoint(latest_path, model, optimizer, device, early_stopper)

    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        if best_ckpt.get("best_val_move_acc", 0.0) > best_move_acc:
            best_move_acc = best_ckpt["best_val_move_acc"]
            best_mate_acc = best_ckpt["best_val_mate_acc"]

    if start_epoch >= cfg.epochs:
        logger.info(f"[{tag}] Configurazione già completata ({start_epoch}/{cfg.epochs} epoche).")
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state"])
        return model, best_move_acc, best_mate_acc

    if start_epoch > 0:
        logger.info(f"[{tag}] Ripresa del training da epoca {start_epoch + 1}/{cfg.epochs}.")

    effective_grad_clip = cfg.grad_clip if (cfg.grad_clip is not None and cfg.grad_clip > 0) else None

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        tr_loss, tr_m_acc, tr_mate_acc = run_epoch(
            model, train_loader, optimizer, device, cfg.mate_loss_weight, train=True, grad_clip=effective_grad_clip
        )
        val_loss, val_m_acc, val_mate_acc = run_epoch(
            model, val_loader, optimizer=None, device=device, mate_loss_weight=cfg.mate_loss_weight, train=False
        )

        metrics_logger.log_epoch(
            tag, epoch, tr_loss, tr_m_acc, tr_mate_acc, val_loss, val_m_acc, val_mate_acc
        )

        logger.info(
            f"[{tag}] Epoch {epoch:03d}/{cfg.epochs} | "
            f"Train Loss: {tr_loss:.4f} Acc(Move/Mate): {tr_m_acc*100:.2f}%/{tr_mate_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc(Move/Mate): {val_m_acc*100:.2f}%/{val_mate_acc*100:.2f}%"
        )

        if val_m_acc > best_move_acc:
            best_move_acc = val_m_acc
            best_mate_acc = val_mate_acc
            save_checkpoint(best_path, model, optimizer, epoch, best_move_acc, best_mate_acc, early_stopper)

        save_checkpoint(latest_path, model, optimizer, epoch, best_move_acc, best_mate_acc, early_stopper)

        metric_to_monitor = val_loss if cfg.es_metric == "val_loss" else -val_m_acc
        if early_stopper(metric_to_monitor):
            logger.info(f"[{tag}] Early stopping attivato all'epoca {epoch}.")
            break

    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state"])

    return model, best_move_acc, best_mate_acc


def main():
    cfg = CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"Dispositivo attivo: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        logger.warning("CUDA non rilevata, esecuzione su CPU.")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    if not os.path.exists(cfg.time_stats_json):
        logger.warning(f"{cfg.time_stats_json} non trovato: verrà usato il fallback lineare.")

    logger.info("Caricamento dataset pre-processati...")
    train_positions = load_split(cfg.data_dir, "train")
    val_positions = load_split(cfg.data_dir, "val")
    logger.info(f"Dataset caricato: {len(train_positions)} posizioni train, {len(val_positions)} posizioni val.")

    train_dataset = PuzzleSequenceDataset(train_positions)
    val_dataset = PuzzleSequenceDataset(val_positions)

    is_cuda = device.type == "cuda"
    use_persistent = cfg.num_workers > 0
    prefetch = 2 if cfg.num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=timed_collate_fn, num_workers=cfg.num_workers,
        pin_memory=is_cuda, persistent_workers=use_persistent,
        prefetch_factor=prefetch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=timed_collate_fn, num_workers=cfg.num_workers,
        pin_memory=is_cuda, persistent_workers=use_persistent,
        prefetch_factor=prefetch
    )

    metrics_logger = TrainingMetricsLogger(output_dir=cfg.results_dir)

    # 1. Addestramento modello Timed
    logger.info("Avvio training modello TIMED")
    timed_model, t_move_acc, t_mate_acc = train_one_config(
        use_time=True, train_loader=train_loader, val_loader=val_loader,
        device=device, cfg=cfg, metrics_logger=metrics_logger
    )

    # 2. Addestramento modello Untimed
    logger.info("Avvio training modello UNTIMED")
    untimed_model, u_move_acc, u_mate_acc = train_one_config(
        use_time=False, train_loader=train_loader, val_loader=val_loader,
        device=device, cfg=cfg, metrics_logger=metrics_logger
    )

    logger.info("Salvataggio delle metriche di training e generazione dei plot comparativi...")
    metrics_logger.save_metrics_to_disk()
    metrics_logger.plot_training_curves()

    logger.info(
        f"Confronto finale Training (Best Val Move Acc / Mate Acc):\n"
        f" - Con segnale temporale:   Move={t_move_acc*100:.2f}% | Mate={t_mate_acc*100:.2f}%\n"
        f" - Senza segnale temporale: Move={u_move_acc*100:.2f}% | Mate={u_mate_acc*100:.2f}%"
    )
    logger.info("Training completato. Utilizzare EvaluateModel.py per l'analisi stratificata.")

if __name__ == "__main__":
    main()
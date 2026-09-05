import logging
import os
import random
import sys
import time
from types import SimpleNamespace
from typing import Optional, Tuple
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader

from timegnn.train.early_stopping import EarlyStopping
from Component.PuzzleSequenceDataset import timed_collate_fn
from LMDBShards import PuzzleLMDBDataset
from ModelUtils.PolicyGNN import legal_move_log_probs, policy_targets_to_global_index
from ModelUtils.TimeChainGnn import TimedPolicyGNN
from ModelUtils.MetricLogger import TrainingMetricsLogger
from ModelUtils.Utils import _argmax_per_graph, load_checkpoint, save_checkpoint

MAX_MATE_N = 10


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


def benchmark_loader(loader: DataLoader, n_batches: int = 10) -> float:
    """Misura il tempo per batch e logga lo spawn dei worker (utile su Windows)."""
    nw = loader.num_workers
    logger.info(f"Benchmark DataLoader ({n_batches} batch, num_workers={nw})...")
    if nw > 0 and sys.platform.startswith("win"):
        logger.info(
            f"  -> Spawn di {nw} worker in corso. Con LMDB multi-shard (mmap, "
            f"handle per shard aperti lazy, no torch.load completo in RAM) "
            f"dovrebbe essere quasi istantaneo."
        )

    t0 = time.time()
    t_first = None
    for i, _ in enumerate(loader):
        if i == 0:
            t_first = time.time() - t0
            logger.info(f"  -> Primo batch pronto dopo {t_first:.1f}s")
        if (i + 1) % 2 == 0:
            logger.info(f"  -> batch {i+1}/{n_batches} completato")
        if i + 1 >= n_batches:
            break

    elapsed = time.time() - t0
    denom = max(1, n_batches - 1)
    per_batch_ms = (elapsed - (t_first or 0)) / denom * 1000
    logger.info(
        f"[benchmark] Totale {elapsed:.1f}s | spawn={t_first or 0:.1f}s | "
        f"{per_batch_ms:.0f} ms/batch (escluso spawn)"
    )
    if per_batch_ms > 500:
        logger.warning(
            "[benchmark] >500ms/batch: il data loading è il collo di bottiglia. "
            "Con LMDB verifica readahead/OS page cache o riduci num_workers."
        )
    return per_batch_ms


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    mate_loss_weight: float,
    train: bool,
    grad_clip: Optional[float] = 1.0,
) -> Tuple[float, float, float]:
    """Esegue un'epoca (train o validation) e ritorna loss/accuracy medie."""
    model.train(train)
    total_loss = 0.0
    total_move_correct = 0
    total_mate_correct = 0
    total_examples = 0

    t_epoch_start = time.time()
    log_every = max(1, len(loader) // 20)  # ~20 log per epoca

    for batch_idx, (inner_batch, chain_edge_index, chain_edge_attr) in enumerate(loader):
        inner_batch = inner_batch.to(device, non_blocking=True)
        chain_edge_index = chain_edge_index.to(device, non_blocking=True)
        chain_edge_attr = chain_edge_attr.to(device, non_blocking=True)
        num_graphs = inner_batch.num_graphs

        with torch.set_grad_enabled(train):
            move_scores, edge_batch, mate_logits = model(
                inner_batch, chain_edge_index, chain_edge_attr
            )
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

        if train and (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - t_epoch_start
            pct = (batch_idx + 1) / len(loader) * 100
            eta = elapsed / (batch_idx + 1) * (len(loader) - batch_idx - 1)
            logger.info(
                f"  batch {batch_idx+1}/{len(loader)} ({pct:.0f}%) | "
                f"loss={loss.item():.4f} | elapsed={elapsed:.0f}s | ETA={eta:.0f}s"
            )

        del loss, move_scores, mate_logits, inner_batch, chain_edge_index, chain_edge_attr

    return (
        total_loss / total_examples,
        total_move_correct / total_examples,
        total_mate_correct / total_examples,
    )


def train_one_config(
    use_time: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: SimpleNamespace,
    metrics_logger: TrainingMetricsLogger,
) -> Tuple[torch.nn.Module, float, float]:
    """Allena una singola configurazione (timed / untimed) con resume da checkpoint."""
    tag = "timed" if use_time else "untimed"
    set_seed(cfg.seed)

    model = TimedPolicyGNN(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        lambda_decay=cfg.lambda_decay,
        use_time=use_time,
    ).to(device)

    if getattr(cfg, "compile", False):
        try:
            model = torch.compile(model)
            logger.info(f"[{tag}] torch.compile abilitato (prima epoca più lenta).")
        except Exception as err:
            logger.warning(
                f"[{tag}] torch.compile non riuscito ({err}), fallback su interprete standard."
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    early_stopper = EarlyStopping(patience=cfg.patience, delta=0.0)

    latest_path = os.path.join(cfg.checkpoint_dir, f"{tag}_latest.pt")
    best_path = os.path.join(cfg.checkpoint_dir, f"{tag}_best.pt")

    start_epoch, best_move_acc, best_mate_acc = load_checkpoint(
        latest_path, model, optimizer, device, early_stopper
    )

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

    effective_grad_clip = (
        cfg.grad_clip if (cfg.grad_clip is not None and cfg.grad_clip > 0) else None
    )

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        t_ep = time.time()
        tr_loss, tr_m_acc, tr_mate_acc = run_epoch(
            model, train_loader, optimizer, device, cfg.mate_loss_weight,
            train=True, grad_clip=effective_grad_clip,
        )
        val_loss, val_m_acc, val_mate_acc = run_epoch(
            model, val_loader, optimizer=None, device=device,
            mate_loss_weight=cfg.mate_loss_weight, train=False,
        )

        metrics_logger.log_epoch(
            tag, epoch, tr_loss, tr_m_acc, tr_mate_acc, val_loss, val_m_acc, val_mate_acc
        )

        logger.info(
            f"[{tag}] Epoch {epoch:03d}/{cfg.epochs} | "
            f"Train Loss: {tr_loss:.4f} Acc(Move/Mate): {tr_m_acc*100:.2f}%/{tr_mate_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc(Move/Mate): {val_m_acc*100:.2f}%/{val_mate_acc*100:.2f}% | "
            f"tempo={time.time()-t_ep:.0f}s"
        )

        if val_m_acc > best_move_acc:
            best_move_acc = val_m_acc
            best_mate_acc = val_mate_acc
            save_checkpoint(
                best_path, model, optimizer, epoch, best_move_acc, best_mate_acc, early_stopper
            )

        save_checkpoint(
            latest_path, model, optimizer, epoch, best_move_acc, best_mate_acc, early_stopper
        )

        metric_to_monitor = val_loss if cfg.es_metric == "val_loss" else -val_m_acc
        if early_stopper(metric_to_monitor):
            logger.info(f"[{tag}] Early stopping attivato all'epoca {epoch}.")
            break

    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state"])

    return model, best_move_acc, best_mate_acc


def build_loader(dataset, cfg, shuffle: bool, is_cuda: bool) -> DataLoader:
    """DataLoader coerente: persistent_workers/prefetch_factor solo se num_workers > 0."""
    num_workers = int(getattr(cfg, "num_workers", 0) or 0)

    kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=timed_collate_fn,
        num_workers=num_workers,
        pin_memory=is_cuda,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(getattr(cfg, "prefetch_factor", 4))

    return DataLoader(dataset, **kwargs)


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

    logger.info("Caricamento dataset LMDB...")
    lmdb_train_dir = os.path.join(cfg.data_dir, "lmdb_train")
    lmdb_val_dir = os.path.join(cfg.data_dir, "lmdb_val")

    build_hint = (
        "Lancia prima: python LMDBShards.py build --data-dir "
        f"{cfg.data_dir} --splits train val "
        "(opzionali --target-shard-gb / --map-size-gb)."
    )

    for split_dir, split_name in ((lmdb_train_dir, "train"), (lmdb_val_dir, "val")):
        metadata_path = os.path.join(split_dir, "__metadata__.pkl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata LMDB mancante per lo split '{split_name}': {metadata_path}. "
                f"{build_hint}"
            )

    train_dataset = PuzzleLMDBDataset(lmdb_train_dir)
    val_dataset = PuzzleLMDBDataset(lmdb_val_dir)
    logger.info(
        f"Dataset LMDB: {len(train_dataset)} train ({train_dataset.num_shards} shard), "
        f"{len(val_dataset)} val ({val_dataset.num_shards} shard) "
        f"(random access mmap, nessuna cache shard in RAM)"
    )

    is_cuda = device.type == "cuda"
    num_workers = int(getattr(cfg, "num_workers", 0) or 0)

    if num_workers == 0:
        logger.warning(
            "num_workers=0: il data loading gira sul main thread. "
            "Con LMDB puoi tranquillamente salire a 8+ (ogni worker riapre i propri "
            "handle mmap per shard dopo il fork, vedi __getstate__ in LMDBShards.py)."
        )

    train_loader = build_loader(train_dataset, cfg, shuffle=True, is_cuda=is_cuda)
    val_loader = build_loader(val_dataset, cfg, shuffle=False, is_cuda=is_cuda)

    if getattr(cfg, "benchmark_loader", True):
        benchmark_loader(train_loader, n_batches=10)

    metrics_logger = TrainingMetricsLogger(output_dir=cfg.results_dir)

    logger.info("Avvio training modello TIMED")
    timed_model, t_move_acc, t_mate_acc = train_one_config(
        use_time=True, train_loader=train_loader, val_loader=val_loader,
        device=device, cfg=cfg, metrics_logger=metrics_logger,
    )

    logger.info("Avvio training modello UNTIMED")
    untimed_model, u_move_acc, u_mate_acc = train_one_config(
        use_time=False, train_loader=train_loader, val_loader=val_loader,
        device=device, cfg=cfg, metrics_logger=metrics_logger,
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
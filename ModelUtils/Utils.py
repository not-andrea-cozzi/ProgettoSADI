import csv
import json
import logging
import os
import random
import sys
from types import SimpleNamespace
from typing import Dict, List, Set, Optional, Any

import chess
import numpy as np
import pandas as pd
import torch
import yaml
import zstandard as zstd
from torch.utils.data import DataLoader
from tqdm import tqdm

from Component.PuzzleSequenceDataset import PuzzleSequenceDataset, timed_collate_fn
from Training_TestModel.PolicyGNN import legal_move_log_probs, policy_targets_to_global_index
from Training_TestModel.TimeChainGnn import TimedPolicyGNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("utils")


# -------------------- YAML --------------------
def load_yaml_config(path: str) -> dict:
    """Carica un file YAML e restituisce un dizionario."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File YAML {path} non trovato.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config_as_namespace(path: str) -> SimpleNamespace:
    """Carica YAML come SimpleNamespace (utile per compatibilità)."""
    return SimpleNamespace(**load_yaml_config(path))


# -------------------- Decompressione CSV --------------------
def resolve_puzzle_csv(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Se path è .csv.zst lo decomprime accanto a sé (skip se già fatto)."""
    if not path.endswith(".zst"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} non trovato.")
        return path

    out_csv = path[: -len(".zst")]
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        logger.info(f"{out_csv} già presente, decompressione saltata.")
        return out_csv

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    tmp_out = out_csv + ".tmp"
    dctx = zstd.ZstdDecompressor()
    total_size = os.path.getsize(path)
    try:
        with open(path, "rb") as f_in, open(tmp_out, "wb") as f_out:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Decomprimo {os.path.basename(path)}") as pbar:
                reader = dctx.stream_reader(f_in)
                while True:
                    chunk = reader.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.n = f_in.tell()
                    pbar.refresh()
        os.replace(tmp_out, out_csv)
    except BaseException:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        raise
    return out_csv


# -------------------- Estrazione ID dai .pt --------------------
def extract_puzzle_ids_from_pt(path: str) -> Set[str]:
    """Estrae i PuzzleId (o problem_id) da un file .pt (lista di Data)."""
    ids: Set[str] = set()
    if not os.path.exists(path):
        logger.warning(f"{path} non trovato, nessun ID escluso da qui.")
        return ids
    data_list = torch.load(path, weights_only=False)
    for d in data_list:
        pid = getattr(d, "puzzle_id", None)
        if pid is not None:
            ids.add(str(pid))
        prob_id = getattr(d, "problem_id", None)
        if prob_id is not None:
            ids.add(str(prob_id))
    return ids


def collect_seen_puzzle_ids(
    dataset_dir: str,
    merged_subfolder: str,
    holdout_subfolder: str,
    holdout_filename: str
) -> Set[str]:
    """Raccoglie tutti i PuzzleId già usati in train/val/test + holdout."""
    seen: Set[str] = set()

    merged_dir = os.path.join(dataset_dir, merged_subfolder)
    for split in ("train", "val", "test"):
        p = os.path.join(merged_dir, f"merged_{split}.pt")
        found = extract_puzzle_ids_from_pt(p)
        logger.info(f"  merged_{split}.pt -> {len(found)} id esclusi")
        seen |= found

    holdout_path = os.path.join(dataset_dir, holdout_subfolder, holdout_filename)
    found = extract_puzzle_ids_from_pt(holdout_path)
    logger.info(f"  {holdout_filename} -> {len(found)} id esclusi")
    seen |= found

    logger.info(f"Totale PuzzleId già visti da escludere: {len(seen)}")
    return seen


# -------------------- Mate extraction --------------------
def extract_mate_n(themes: str) -> int:
    """Estrae il numero di mosse per il mate (es. 'mateIn3' -> 3)."""
    for t in themes.split():
        if t.startswith("mateIn"):
            return int(t.replace("mateIn", ""))
    return 0


def simulate_clock(rating: float, avg_time_by_rating: Dict[int, float]) -> float:
    """Simula il tempo a disposizione in base al rating."""
    if avg_time_by_rating:
        bucket = round(float(rating) / 100) * 100
        return avg_time_by_rating.get(bucket, 15.0)
    return 5.0 + (float(rating) / 3000.0) * 55.0


# -------------------- Costruzione dataset di validazione --------------------
def build_validator_dataset(
    csv_paths: List[str],
    seen_ids: Set[str],
    mate_range: tuple,
    target_samples: int,
    avg_time_by_rating: Dict[int, float],
    seed: int = 42,
    chunksize: int = 50_000,
) -> List:
    """
    Costruisce un nuovo dataset di test usando puzzle mai visti, nel range mate specificato.
    Unifica la logica di TestModelValidator e Validator.
    """
    lo, hi = mate_range
    theme_pattern = "|".join(f"mateIn{n}" for n in range(lo, hi + 1))
    already_excluded = set(seen_ids)

    candidate_rows = []
    for csv_path in csv_paths:
        if len(candidate_rows) >= target_samples * 3:
            break
        reader = pd.read_csv(csv_path, chunksize=chunksize)
        pbar = tqdm(desc=f"Scansione {os.path.basename(csv_path)} [solo mai visti]", unit=" righe")
        for chunk in reader:
            mask = chunk["Themes"].str.contains(theme_pattern, na=False)
            mask &= ~chunk["PuzzleId"].astype(str).isin(already_excluded)
            filtered = chunk[mask]
            candidate_rows.extend(filtered.to_dict("records"))
            already_excluded.update(filtered["PuzzleId"].astype(str).tolist())
            pbar.update(len(filtered))
            if len(candidate_rows) >= target_samples * 3:
                break
        pbar.close()

    logger.info(f"Righe candidate (mai viste, mate {lo}-{hi}, {len(csv_paths)} fonti): {len(candidate_rows)}")
    random.Random(seed).shuffle(candidate_rows)

    from DatasetCreator.GraphBuilder import GraphBuilder  # import locale per evitare dipendenze circolari

    data_list = []
    for row in tqdm(candidate_rows, desc="Costruzione grafi test set"):
        if len(data_list) >= target_samples:
            break

        uci_moves = str(row["Moves"]).split()
        if not uci_moves:
            continue

        try:
            board = chess.Board(row["FEN"])
        except Exception:
            continue

        mate_n_iniziale = extract_mate_n(str(row["Themes"]))
        if mate_n_iniziale == 0:
            continue

        clock = simulate_clock(row["Rating"], avg_time_by_rating)

        first_move_uci = uci_moves[0]
        try:
            first_move = chess.Move.from_uci(first_move_uci)
        except Exception:
            continue
        if first_move not in board.legal_moves:
            continue
        board.push(first_move)

        added_for_this_puzzle = False
        for ply_idx, uci in enumerate(uci_moves[1:], start=1):
            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                break

            if ply_idx % 2 == 0:
                if move in board.legal_moves:
                    board.push(move)
                continue

            legal = list(board.legal_moves)
            if move not in legal:
                break

            best_idx = legal.index(move)
            current_mate_n = max(1, mate_n_iniziale - (ply_idx // 2))

            label = {"mate_n": current_mate_n, "best_move_idx": best_idx}
            try:
                d = GraphBuilder.board_to_pyg_data(
                    board,
                    clock_seconds=clock * (1 + 0.1 * ply_idx),
                    label=label,
                )
            except Exception:
                break

            d.puzzle_id = row["PuzzleId"]
            d.rating = float(row["Rating"])
            data_list.append(d)
            added_for_this_puzzle = True

            board.push(move)

            if len(data_list) >= target_samples:
                break

        if not added_for_this_puzzle:
            continue

    return data_list


# -------------------- Modelli e predizione --------------------
def load_model(
    ckpt_path: str,
    use_time: bool,
    hidden_dim: int,
    num_layers: int,
    lambda_decay: float,
    device: torch.device,
) -> TimedPolicyGNN:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} non trovato.")

    model = TimedPolicyGNN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        lambda_decay=lambda_decay,
        use_time=use_time,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]

    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()
    logger.info(
        f"Caricato {ckpt_path} (epoca={ckpt.get('epoch','?')}, "
        f"val_move_acc={ckpt.get('best_val_move_acc', 0):.4f}, "
        f"val_mate_acc={ckpt.get('best_val_mate_acc', 0):.4f})"
    )
    return model


def load_dataset_from_pt(
    pt_path: str,
    batch_size: int = 64,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> DataLoader:
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"File {pt_path} non trovato.")
    data_list = torch.load(pt_path, weights_only=False)
    dataset = PuzzleSequenceDataset(data_list)
    pin_memory = (device is not None and device.type == "cuda")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=timed_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


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


def predict_batch(
    model: TimedPolicyGNN, batch, device: torch.device
) -> Dict[str, torch.Tensor]:
    inner_batch, chain_edge_index, chain_edge_attr = batch
    inner_batch = inner_batch.to(device, non_blocking=True)
    chain_edge_index = chain_edge_index.to(device, non_blocking=True)
    chain_edge_attr = chain_edge_attr.to(device, non_blocking=True)
    num_graphs = inner_batch.num_graphs

    with torch.no_grad():
        move_scores, edge_batch, mate_logits = model(
            inner_batch, chain_edge_index, chain_edge_attr
        )
        move_pred = _argmax_per_graph(move_scores, edge_batch, num_graphs)
        mate_pred = mate_logits.argmax(dim=-1)

    return {
        "move_pred": move_pred.cpu(),
        "mate_pred": mate_pred.cpu(),
        "move_scores": move_scores.cpu(),
        "mate_logits": mate_logits.cpu(),
        "edge_batch": edge_batch.cpu(),
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mate_loss_weight: float,
) -> dict:
    """
    Valuta il modello sul dataloader e restituisce metriche aggregate.
    (Funzione estratta da TestModels.py)
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

    from Training_TestModel.PolicyGNN import legal_move_log_probs, policy_targets_to_global_index

    with torch.no_grad():
        for batch in loader:
            inner_batch, chain_edge_index, chain_edge_attr = batch
            inner_batch = inner_batch.to(device, non_blocking=True)
            chain_edge_index = chain_edge_index.to(device, non_blocking=True)
            chain_edge_attr = chain_edge_attr.to(device, non_blocking=True)
            num_graphs = inner_batch.num_graphs

            preds = predict_batch(model, batch, device)
            move_pred = preds["move_pred"].to(device)
            move_scores = preds["move_scores"].to(device)
            edge_batch = preds["edge_batch"].to(device)
            mate_logits = preds["mate_logits"].to(device)

            # Loss
            log_probs = legal_move_log_probs(move_scores, edge_batch, num_graphs)
            target_idx = policy_targets_to_global_index(edge_batch, inner_batch.y, num_graphs)
            policy_loss = -log_probs[target_idx].mean()

            mate_target = inner_batch.mate_n.clamp(0, mate_logits.size(-1) - 1)
            mate_loss = torch.nn.functional.cross_entropy(mate_logits, mate_target)
            loss = policy_loss + mate_loss_weight * mate_loss

            # Move acc
            move_correct = (move_pred == target_idx).cpu().numpy()
            move_correct_flags.append(move_correct)

            # Mate acc
            mate_pred = preds["mate_pred"]
            mate_correct = (mate_pred == mate_target.cpu()).numpy()
            mate_correct_flags.append(mate_correct)
            mate_true_all.append(mate_target.cpu().numpy())
            mate_pred_all.append(mate_pred.numpy())

            # Rating
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
        "mate_n": mate_true_np,
        "rating": rating_np,
    }
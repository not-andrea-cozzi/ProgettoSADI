from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
import chess
from torch.utils.data import DataLoader

from Component.PuzzleSequenceDataset import PuzzleSequenceDataset, timed_collate_fn
from ModelUtils.TimeChainGnn import TimedPolicyGNN
from ModelUtils.PolicyGNN import legal_move_log_probs, policy_targets_to_global_index
from ModelUtils.EvaluatorPlotter import EvaluatorPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("val_heldout")


def load_config(config_path: str = "Yaml/val.yaml") -> SimpleNamespace:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"File di configurazione {config_path} non trovato."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return SimpleNamespace(**raw)



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


def build_and_load_model(ckpt_path: str, use_time: bool, cfg: SimpleNamespace, device: torch.device) -> TimedPolicyGNN:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} non trovato. Esegui prima TrainModels.py.")
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
    logger.info(
        f"Caricato {ckpt_path} (epoca={ckpt.get('epoch', '?')}, "
        f"val_move_acc={ckpt.get('best_val_move_acc', 0):.4f}, "
        f"val_mate_acc={ckpt.get('best_val_mate_acc', 0):.4f})"
    )
    return model


def load_holdout(path: str) -> List[Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} non trovato. Esegui prima MainDatasetCreator.py con step 'external_holdout'."
        )
    data_list = torch.load(path, weights_only=False)
    if not data_list:
        raise RuntimeError(f"{path} è vuoto: nessun problema da valutare.")
    return data_list


@torch.no_grad()
def evaluate_gnn_on_holdout(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()

    problem_ids: List[str] = []
    fens: List[str] = []
    mate_true: List[int] = []
    best_uci: List[str] = []
    pred_uci: List[str] = []
    move_correct: List[bool] = []

    for inner_batch, chain_edge_index, chain_edge_attr in loader:
        inner_batch = inner_batch.to(device, non_blocking=True)
        chain_edge_index = chain_edge_index.to(device, non_blocking=True)
        chain_edge_attr = chain_edge_attr.to(device, non_blocking=True)
        num_graphs = inner_batch.num_graphs

        move_scores, edge_batch, mate_logits = model(inner_batch, chain_edge_index, chain_edge_attr)

        target_idx = policy_targets_to_global_index(edge_batch, inner_batch.y, num_graphs)
        move_pred_global = _argmax_per_graph(move_scores, edge_batch, num_graphs)

        src_nodes = inner_batch.edge_index[0][inner_batch.edge_attr == 0]
        dst_nodes = inner_batch.edge_index[1][inner_batch.edge_attr == 0]

        graph_node_offset = torch.arange(num_graphs, device=device) * 64

        for g in range(num_graphs):
            fen = getattr(inner_batch, "fen", None)
            true_uci = getattr(inner_batch, "best_move_uci", None)
            pid = getattr(inner_batch, "problem_id", None)
            mate_n_val = getattr(inner_batch, "mate_n", None)

            fen_g = fen[g] if isinstance(fen, (list, tuple)) else fen
            true_uci_g = true_uci[g] if isinstance(true_uci, (list, tuple)) else true_uci
            pid_g = pid[g] if isinstance(pid, (list, tuple)) else pid
            mate_n_g = int(mate_n_val[g]) if isinstance(mate_n_val, torch.Tensor) else int(mate_n_val)

            board = chess.Board(fen_g)
            legal_moves_g = list(board.legal_moves)

            edge_mask_g = edge_batch == g
            local_scores = move_scores[edge_mask_g]
            local_src = (src_nodes[edge_mask_g] - graph_node_offset[g]).cpu().tolist()
            local_dst = (dst_nodes[edge_mask_g] - graph_node_offset[g]).cpu().tolist()

            if local_scores.numel() == 0 or not legal_moves_g:
                pred_move_str = ""
            else:
                best_local = int(torch.argmax(local_scores).item())
                from_sq, to_sq = local_src[best_local], local_dst[best_local]
                candidates = [
                    m for m in legal_moves_g if m.from_square == from_sq and m.to_square == to_sq
                ]
                pred_move_str = candidates[0].uci() if candidates else ""

            problem_ids.append(str(pid_g))
            fens.append(fen_g)
            mate_true.append(mate_n_g)
            best_uci.append(str(true_uci_g))
            pred_uci.append(pred_move_str)
            move_correct.append(bool(pred_move_str == str(true_uci_g)))

    return {
        "problem_id": np.array(problem_ids, dtype=object),
        "fen": np.array(fens, dtype=object),
        "mate_n": np.array(mate_true, dtype=np.int64),
        "best_move_uci": np.array(best_uci, dtype=object),
        "pred_move_uci": np.array(pred_uci, dtype=object),
        "move_correct": np.array(move_correct, dtype=bool),
    }


_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)


def _build_llm_prompt(fen: str) -> str:
    return (
       "You are a chess engine. Given the FEN position below, output ONLY "
        "the best move in UCI format: four or five lowercase characters, "
        "e.g. 'e2e4' or 'e7e8q'. Do NOT use algebraic notation (no piece "
        "letters like 'R' or 'N', no '+', no '#', no 'x'). Output ONLY the "
        "UCI move, nothing else.\n\n"
        f"FEN: {fen}\n"
        "Best move (UCI):"
    )


def _extract_uci(text: str) -> Optional[str]:
    if not text:
        return None
    match = _UCI_RE.search(text.strip())
    return match.group(1).lower() if match else None
_SAN_RE = re.compile(
    r"\b(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b"
)


def _try_parse_move(text: str, board: "chess.Board") -> str:
    """Prova prima a interpretare 'text' come UCI, poi come SAN, validando
    contro le mosse legali di 'board'. Ritorna '' se non trova nulla di valido."""
    if not text:
        return ""
    uci_candidate = _extract_uci(text)
    if uci_candidate:
        try:
            move = chess.Move.from_uci(uci_candidate)
            if move in board.legal_moves:
                return move.uci()
        except ValueError:
            pass
    for san_candidate in _SAN_RE.findall(text):
        try:
            move = board.parse_san(san_candidate)
            return move.uci()
        except ValueError:
            continue
    return ""


class GroqLLMSolver:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 32,
        reasoning_effort: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        request_delay_seconds: float = 0.0,
    ):
        import requests

        self._requests = requests
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.request_delay_seconds = request_delay_seconds
    def solve(self, fen: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": _build_llm_prompt(fen)}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._requests.post(
                    self.base_url, headers=headers, json=payload, timeout=self.timeout_seconds
                )
                if resp.status_code == 429:
                    raise RuntimeError(f"rate limited (429): {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]
                raw_text = message.get("content", "") or ""
                pred = _extract_uci(raw_text)
                if not pred:
                    # fallback: gpt-oss puo' lasciare 'content' vuoto se il
                    # ragionamento consuma tutto il budget di token prima
                    # di scrivere la risposta finale, oppure ragiona in SAN
                    reasoning_text = message.get("reasoning", "") or ""
                    if reasoning_text:
                        raw_text = reasoning_text
                        pred = _extract_uci(reasoning_text)
                if self.request_delay_seconds > 0:
                    time.sleep(self.request_delay_seconds)
                return {"raw_text": raw_text, "pred_move_uci": pred or "", "error": None}
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)

        return {"raw_text": "", "pred_move_uci": "", "error": str(last_err)}

def evaluate_llm_on_holdout(
    data_list: List[Any],
    solver: Optional[GroqLLMSolver],
    cache_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    problem_ids: List[str] = []
    mate_true: List[int] = []
    best_uci: List[str] = []
    pred_uci: List[str] = []
    move_correct: List[bool] = []
    raw_texts: List[str] = []
    errors: List[str] = []

    cache: Dict[str, Dict[str, str]] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"Cache LLM caricata da {cache_path}: {len(cache)} risposte già disponibili.")

    for i, d in enumerate(data_list):
        pid = str(getattr(d, "problem_id", i))
        fen = getattr(d, "fen", None)
        true_uci = str(getattr(d, "best_move_uci", ""))
        mate_n = int(getattr(d, "mate_n", 0))

        if fen is None:
            logger.warning(f"[LLM] problema {pid} senza campo 'fen': saltato.")
            continue

        if solver is None:
            problem_ids.append(pid)
            mate_true.append(mate_n)
            best_uci.append(true_uci)
            pred_uci.append("")
            move_correct.append(False)
            raw_texts.append("")
            errors.append("llm_disabled")
            continue

        if pid in cache:
            result = cache[pid]
        else:
            result = solver.solve(fen)
            cache[pid] = result
            if cache_path and (i % 10 == 0):
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)

        pred = result.get("pred_move_uci", "") or ""
        if not pred:
            board = chess.Board(fen)
            pred = _try_parse_move(result.get("raw_text", "") or "", board)
        problem_ids.append(pid)
        mate_true.append(mate_n)
        best_uci.append(true_uci)
        pred_uci.append(pred)
        move_correct.append(bool(pred == true_uci))
        raw_texts.append(result.get("raw_text", ""))
        errors.append(result.get("error") or "")

        if (i + 1) % 20 == 0:
            logger.info(f"[LLM] valutati {i + 1}/{len(data_list)} problemi.")

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    return {
        "problem_id": np.array(problem_ids, dtype=object),
        "mate_n": np.array(mate_true, dtype=np.int64),
        "best_move_uci": np.array(best_uci, dtype=object),
        "pred_move_uci": np.array(pred_uci, dtype=object),
        "move_correct": np.array(move_correct, dtype=bool),
        "raw_text": np.array(raw_texts, dtype=object),
        "error": np.array(errors, dtype=object),
    }


def build_comparison_table(
    timed_res: Dict[str, np.ndarray],
    untimed_res: Dict[str, np.ndarray],
    llm_res: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    by_pid_timed = {pid: i for i, pid in enumerate(timed_res["problem_id"])}
    by_pid_untimed = {pid: i for i, pid in enumerate(untimed_res["problem_id"])}
    by_pid_llm = {pid: i for i, pid in enumerate(llm_res["problem_id"])}

    all_pids = list(dict.fromkeys(
        list(timed_res["problem_id"]) + list(untimed_res["problem_id"]) + list(llm_res["problem_id"])
    ))

    rows = []
    for pid in all_pids:
        row: Dict[str, Any] = {"problem_id": pid}

        if pid in by_pid_timed:
            i = by_pid_timed[pid]
            row["fen"] = timed_res["fen"][i]
            row["mate_n"] = int(timed_res["mate_n"][i])
            row["risposta_giusta"] = timed_res["best_move_uci"][i]
            row["timed_pred"] = timed_res["pred_move_uci"][i]
            row["timed_correct"] = bool(timed_res["move_correct"][i])
        else:
            row.setdefault("fen", "")
            row.setdefault("mate_n", -1)
            row.setdefault("risposta_giusta", "")
            row["timed_pred"] = ""
            row["timed_correct"] = False

        if pid in by_pid_untimed:
            i = by_pid_untimed[pid]
            row["untimed_pred"] = untimed_res["pred_move_uci"][i]
            row["untimed_correct"] = bool(untimed_res["move_correct"][i])
            if not row.get("risposta_giusta"):
                row["risposta_giusta"] = untimed_res["best_move_uci"][i]
                row["mate_n"] = int(untimed_res["mate_n"][i])
        else:
            row["untimed_pred"] = ""
            row["untimed_correct"] = False

        if pid in by_pid_llm:
            i = by_pid_llm[pid]
            row["llm_pred"] = llm_res["pred_move_uci"][i]
            row["llm_correct"] = bool(llm_res["move_correct"][i])
            row["llm_error"] = llm_res["error"][i]
            if not row.get("risposta_giusta"):
                row["risposta_giusta"] = llm_res["best_move_uci"][i]
                row["mate_n"] = int(llm_res["mate_n"][i])
        else:
            row["llm_pred"] = ""
            row["llm_correct"] = False
            row["llm_error"] = ""

        rows.append(row)

    return rows


def save_comparison_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fieldnames = [
        "problem_id", "fen", "mate_n", "risposta_giusta",
        "timed_pred", "timed_correct",
        "untimed_pred", "untimed_correct",
        "llm_pred", "llm_correct", "llm_error",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    logger.info(f"Salvato {out_path} ({len(rows)} righe).")


def save_summary_csv(rows: List[Dict[str, Any]], out_path: str, max_n: int = 10) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "totale", "timed_corrette", "timed_acc", "untimed_corrette", "untimed_acc",
                          "llm_corrette", "llm_acc"])

        for n in range(1, max_n + 1):
            subset = [r for r in rows if r["mate_n"] == n]
            total = len(subset)
            if total == 0:
                writer.writerow([n, 0, 0, "", 0, "", 0, ""])
                continue
            t_c = sum(1 for r in subset if r["timed_correct"])
            u_c = sum(1 for r in subset if r["untimed_correct"])
            l_c = sum(1 for r in subset if r["llm_correct"])
            writer.writerow([
                n, total,
                t_c, f"{t_c / total:.6f}",
                u_c, f"{u_c / total:.6f}",
                l_c, f"{l_c / total:.6f}",
            ])

        total = len(rows)
        if total > 0:
            t_c = sum(1 for r in rows if r["timed_correct"])
            u_c = sum(1 for r in rows if r["untimed_correct"])
            l_c = sum(1 for r in rows if r["llm_correct"])
            writer.writerow(["all", total,
                              t_c, f"{t_c / total:.6f}",
                              u_c, f"{u_c / total:.6f}",
                              l_c, f"{l_c / total:.6f}"])

    logger.info(f"Salvato {out_path}.")


def _to_plotter_results_dict(rows: List[Dict[str, Any]], pred_key: str, correct_key: str) -> Dict[str, np.ndarray]:
    mate_true = np.array([r["mate_n"] for r in rows], dtype=np.int64)
    move_correct = np.array([1 if r[correct_key] else 0 for r in rows], dtype=np.int64)
    return {"mate_true": mate_true, "move_correct": move_correct}


def plot_three_way_bars(rows: List[Dict[str, Any]], plots_dir: str, max_n: int = 10,
                         filename: str = "bars_per_n_timed_untimed_llm.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)

    n_values = np.arange(1, max_n + 1)

    def acc_for(correct_key: str) -> List[float]:
        out = []
        for n in n_values:
            subset = [r for r in rows if r["mate_n"] == n]
            if not subset:
                out.append(0.0)
                continue
            out.append(100.0 * sum(1 for r in subset if r[correct_key]) / len(subset))
        return out

    timed_acc = acc_for("timed_correct")
    untimed_acc = acc_for("untimed_correct")
    llm_acc = acc_for("llm_correct")

    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    r1 = np.arange(len(n_values))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]

    ax.bar(r1, timed_acc, color="royalblue", width=width, edgecolor="grey", label="Timed GNN")
    ax.bar(r2, untimed_acc, color="coral", width=width, edgecolor="grey", label="Untimed GNN")
    ax.bar(r3, llm_acc, color="mediumseagreen", width=width, edgecolor="grey", label="LLM (gpt-oss-20b)")

    ax.set_xticks([x + width for x in r1])
    ax.set_xticklabels([f"Mate in {n}" for n in n_values])
    ax.set_xlabel("Profondità di matto (n)", fontweight="bold")
    ax.set_ylabel("Move Accuracy (%)", fontweight="bold")
    ax.set_title("Held-Out Esterno — Timed GNN vs Untimed GNN vs LLM, per profondità di matto")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.7)

    fig.tight_layout()
    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Salvato {out_path}")


def plot_three_way_aggregate(rows: List[Dict[str, Any]], plots_dir: str,
                              filename: str = "aggregate_bars_timed_untimed_llm.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = len(rows)
    if total == 0:
        logger.warning("Nessuna riga da plottare in plot_three_way_aggregate.")
        return

    labels = ["Timed GNN", "Untimed GNN", "LLM (Groq)"]
    accs = [
        sum(1 for r in rows if r["timed_correct"]) / total,
        sum(1 for r in rows if r["untimed_correct"]) / total,
        sum(1 for r in rows if r["llm_correct"]) / total,
    ]
    colors = ["royalblue", "coral", "mediumseagreen"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, accs, color=colors, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Move Accuracy")
    ax.set_title(f"Held-Out Esterno — Accuracy Aggregata (N={total})")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    fig.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Salvato {out_path}")


def main():
    config_path = os.environ.get("VAL_HELDOUT_CONFIG", "Yaml/val.yaml")
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"GPU attiva: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        logger.warning("CUDA non disponibile: esecuzione su CPU.")

    logger.info(f"Caricamento held-out esterno da {cfg.holdout_path}...")
    holdout_positions = load_holdout(cfg.holdout_path)
    logger.info(f"Held-out caricato: {len(holdout_positions)} posizioni.")

    missing_fields = [
        f for f in ("fen", "best_move_uci", "mate_n", "problem_id")
        if not hasattr(holdout_positions[0], f)
    ]
    if missing_fields:
        raise RuntimeError(
            f"L'held-out non contiene i campi richiesti {missing_fields}. "
            f"Rigenera external_holdout.pt con la versione corrente di ExternalHoldoutBuilder."
        )

    holdout_dataset = PuzzleSequenceDataset(holdout_positions)

    is_cuda = device.type == "cuda"
    holdout_loader = DataLoader(
        holdout_dataset, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=timed_collate_fn, num_workers=cfg.num_workers,
        pin_memory=is_cuda,
    )

    timed_ckpt = os.path.join(cfg.checkpoint_dir, "timed_best.pt")
    untimed_ckpt = os.path.join(cfg.checkpoint_dir, "untimed_best.pt")
    model_timed = build_and_load_model(timed_ckpt, use_time=True, cfg=cfg, device=device)
    model_untimed = build_and_load_model(untimed_ckpt, use_time=False, cfg=cfg, device=device)

    logger.info("Valutazione GNN timed su held-out esterno...")
    timed_res = evaluate_gnn_on_holdout(model_timed, holdout_loader, device)
    logger.info(f"[Timed]   Move Acc={timed_res['move_correct'].mean() * 100:.2f}% | N={len(timed_res['problem_id'])}")

    logger.info("Valutazione GNN untimed su held-out esterno...")
    untimed_res = evaluate_gnn_on_holdout(model_untimed, holdout_loader, device)
    logger.info(f"[Untimed] Move Acc={untimed_res['move_correct'].mean() * 100:.2f}% | N={len(untimed_res['problem_id'])}")

    solver = None
    if getattr(cfg, "llm_enabled", False):
        # Utilizza None come fallback predefinito
        api_key = os.environ.get(cfg.llm_api_key_env, "")
        if not api_key:
            logger.warning(
                f"Variabile d'ambiente {cfg.llm_api_key_env} non impostata: "
                f"valutazione LLM saltata."
                )
        else:
            solver = GroqLLMSolver(
                base_url=cfg.llm_base_url,
                model=cfg.llm_model,
                api_key=api_key,
                temperature=getattr(cfg, "llm_temperature", 0.0),
                max_tokens=getattr(cfg, "llm_max_tokens", 32),
                reasoning_effort=getattr(cfg, "llm_reasoning_effort", None),
                timeout_seconds=getattr(cfg, "llm_timeout_seconds", 30),
                max_retries=getattr(cfg, "llm_max_retries", 3),
                retry_backoff_seconds=getattr(cfg, "llm_retry_backoff_seconds", 2.0),
                request_delay_seconds=getattr(cfg, "llm_request_delay_seconds", 0.0),
            )
            logger.info(f"LLM solver pronto: provider={cfg.llm_provider} model={cfg.llm_model}")
    else:
        logger.info("llm_enabled=false in val.yaml: valutazione LLM saltata.")

    cache_path = os.path.join(cfg.out_dir, "llm_responses_cache.json")
    logger.info("Valutazione LLM su held-out esterno...")
    llm_res = evaluate_llm_on_holdout(holdout_positions, solver, cache_path=cache_path)
    if solver is not None:
        logger.info(f"[LLM]     Move Acc={llm_res['move_correct'].mean() * 100:.2f}% | N={len(llm_res['problem_id'])}")

    rows = build_comparison_table(timed_res, untimed_res, llm_res)

    os.makedirs(cfg.out_dir, exist_ok=True)
    save_comparison_csv(rows, os.path.join(cfg.out_dir, "heldout_comparison.csv"))
    save_summary_csv(rows, os.path.join(cfg.out_dir, "heldout_summary_per_n.csv"),
                      max_n=cfg.max_mate_depth_eval)

    plotter = EvaluatorPlotter(plots_dir=cfg.plots_dir, out_dir=cfg.out_dir)

    res_t_for_plotter = _to_plotter_results_dict(rows, "timed_pred", "timed_correct")
    res_u_for_plotter = _to_plotter_results_dict(rows, "untimed_pred", "untimed_correct")

    plotter.plot_depth_bars(res_t_for_plotter, res_u_for_plotter, max_n=cfg.max_mate_depth_eval,
                             filename="heldout_bars_timed_vs_untimed.png")
    plotter.plot_depth_curves(res_t_for_plotter, res_u_for_plotter, max_n=cfg.max_mate_depth_eval,
                               filename="heldout_curves_timed_vs_untimed.png")
    plotter.save_depth_metrics(res_t_for_plotter, res_u_for_plotter, max_n=cfg.max_mate_depth_eval,
                                filename="heldout_metrics_timed_vs_untimed.csv")

    plot_three_way_bars(rows, cfg.plots_dir, max_n=cfg.max_mate_depth_eval)
    plot_three_way_aggregate(rows, cfg.plots_dir)

    logger.info("Valutazione held-out completata. CSV e plot salvati in "
                f"{cfg.out_dir} e {cfg.plots_dir}.")


if __name__ == "__main__":
    main()
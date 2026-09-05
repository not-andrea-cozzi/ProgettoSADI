"""
RatingStats.py

Sostituisce NormalizationEncoder.py nello schema compresso.

Con lo schema compresso (vedi DatasetCreator/GraphBuilder.py):
  - clock_norm e' gia' log-normalizzato in [0, 1] alla fonte (non serve
    nessuno scaler aggiuntivo).
  - rating e' salvato in scala Elo grezza (float16, NaN se assente): la
    normalizzazione (es. (rating - mean) / std) va fatta DENTRO il forward
    del modello, non come step separato di preprocessing su disco, perche'
    non c'e' piu' nessun 'x' denso su cui applicare uno StandardScaler
    colonna per colonna: l'unico valore continuo rimasto e' rating, ed e'
    un singolo scalare per sample.

"""
import json
import math
import os
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data


def compute_rating_stats(data_list: List[Data]) -> Dict[str, float]:
    """Calcola mean/std del rating sui sample dove e' disponibile (non NaN).

    Ritorna anche n_total e n_valid per trasparenza su quanti sample hanno
    effettivamente un rating (utile per capire quanto ci si puo' fidare
    della normalizzazione, specialmente se molti sample games hanno rating
    NaN per via di header PGN mancanti/non parsabili)."""
    values: List[float] = []
    for item in data_list:
        if not hasattr(item, "rating") or item.rating is None:
            continue
        val = item.rating
        val = float(val.item()) if isinstance(val, torch.Tensor) else float(val)
        if val == val:  # esclude NaN
            values.append(val)

    n_total = len(data_list)
    n_valid = len(values)

    if n_valid == 0:
        raise ValueError(
            "Nessun rating valido (non-NaN) trovato nel dataset: impossibile "
            "calcolare mean/std. Verificare che almeno una parte dei sample "
            "abbia il campo 'rating' popolato."
        )

    mean = sum(values) / n_valid
    variance = sum((v - mean) ** 2 for v in values) / n_valid
    std = math.sqrt(variance)

    return {
        "mean": mean,
        "std": std if std > 1e-6 else 1.0,  # evita divisione per ~0 se il rating fosse quasi costante
        "n_total": n_total,
        "n_valid": n_valid,
        "coverage": n_valid / n_total if n_total else 0.0,
    }


def save_rating_stats(stats: Dict[str, float], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    os.replace(tmp_path, out_path)


def load_rating_stats(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rating(rating: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Applica (rating - mean) / std, mappando NaN a 0.0 dopo la
    normalizzazione (non prima: 0.0 grezzo sarebbe un rating Elo assurdo e
    verrebbe normalizzato a un valore fuorviante, tipicamente molto
    negativo). Va chiamata dentro il forward del modello, non in
    preprocessing, cosi' il dataset su disco resta invariato indipendente
    dai parametri di normalizzazione scelti."""
    rating = rating.to(torch.float32)
    has_rating = ~torch.isnan(rating)
    safe_std = std if std > 1e-6 else 1.0
    normalized = (rating - mean) / safe_std
    return torch.where(has_rating, normalized, torch.zeros_like(normalized))
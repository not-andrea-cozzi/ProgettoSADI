#!/usr/bin/env python3
"""
merge_games_simple.py

Unisce i file .pt di due esecuzioni del games_pipeline, mescola tutto e
risuddivide in train/val/test con proporzioni 80/10/10.
Tutti i percorsi sono hardcoded: modificali qui sotto.
"""

import torch
import random
import logging
import sys
from collections import defaultdict

# ==================== CONFIGURAZIONE (MODIFICA QUI) ====================
# File della prima run
RUN1_TRAIN = "Dataset/Temp_Games_lichess/games_train.pt"
RUN1_VAL   = "Dataset/Temp_Games_lichess/games_val.pt"
RUN1_TEST  = "Dataset/Temp_Games_lichess/games_test.pt"

# File della seconda run
RUN2_TRAIN = "Dataset/games2_train.pt"
RUN2_VAL   = "Dataset/games2_val.pt"
RUN2_TEST  = "Dataset/games2_test.pt"

# File di output
OUTPUT_TRAIN = "Dataset/games_train.pt"
OUTPUT_VAL   = "Dataset/games_val.pt"
OUTPUT_TEST  = "Dataset/games_test.pt"

# Proporzioni (devono sommare a 1)
RATIO_TRAIN = 0.8
RATIO_VAL   = 0.1
RATIO_TEST  = 0.1   # verrà calcolato automaticamente come resto

# Seme per la riproducibilità della mescolanza
SEED = 42
# ======================================================================

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("merge")


def main():
    # 1. Carica tutti i file e concatenali in un'unica lista
    all_paths = [RUN1_TRAIN, RUN1_VAL, RUN1_TEST, RUN2_TRAIN, RUN2_VAL, RUN2_TEST]
    all_data = []

    for path in all_paths:
        logger.info(f"Caricamento {path}...")
        data = torch.load(path, weights_only=False)
        if not isinstance(data, list):
            raise TypeError(f"Il file {path} non contiene una lista, ma {type(data)}")
        all_data.extend(data)
        logger.info(f"  Aggiunti {len(data)} elementi. Totale: {len(all_data)}")

    n_total = len(all_data)
    logger.info(f"Totale elementi combinati: {n_total}")

    # 2. Raggruppa per mate_n
    groups = defaultdict(list)
    for item in all_data:
        # 'mate_n' è un attributo dell'oggetto Data (PyG)
        mate = item.mate_n if hasattr(item, 'mate_n') else None
        if mate is None:
            raise ValueError("Oggetto senza attributo 'mate_n'")
        groups[mate].append(item)

    # Verifica che i mate_n siano nell'intervallo 1..10
    mate_values = sorted(groups.keys())
    logger.info(f"Valori di mate_n presenti: {mate_values}")

    # 3. Per ogni gruppo, split bilanciato
    torch.manual_seed(SEED)
    train_data, val_data, test_data = [], [], []

    for mate, items in groups.items():
        count = len(items)
        # Calcola le quantità per questo gruppo
        n_train = int(RATIO_TRAIN * count)
        n_val = int(RATIO_VAL * count)
        n_test = count - n_train - n_val
        # Se per arrotondamento n_test è negativo, correggi (caso raro)
        if n_test < 0:
            n_train = max(0, n_train)
            n_val = max(0, n_val)
            n_test = count - n_train - n_val

        logger.info(f"mate_n={mate}: totale={count}, train={n_train}, val={n_val}, test={n_test}")

        # Mescola gli indici del gruppo
        perm = torch.randperm(count)
        items_shuffled = [items[i] for i in perm.tolist()]

        # Suddividi
        train_data.extend(items_shuffled[:n_train])
        val_data.extend(items_shuffled[n_train:n_train + n_val])
        test_data.extend(items_shuffled[n_train + n_val:])

    # 4. Mescola globalmente (opzionale, ma per maggior casualità)
    # Mescoliamo ogni split per evitare che i campioni siano raggruppati per mate_n
    # all'interno dello stesso split.
    def shuffle_list(lst):
        perm = torch.randperm(len(lst))
        return [lst[i] for i in perm.tolist()]

    train_data = shuffle_list(train_data)
    val_data = shuffle_list(val_data)
    test_data = shuffle_list(test_data)

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 5. Salva i file
    logger.info(f"Salvataggio {OUTPUT_TRAIN}...")
    torch.save(train_data, OUTPUT_TRAIN)
    logger.info(f"Salvataggio {OUTPUT_VAL}...")
    torch.save(val_data, OUTPUT_VAL)
    logger.info(f"Salvataggio {OUTPUT_TEST}...")
    torch.save(test_data, OUTPUT_TEST)

    logger.info("Operazione completata con successo (bilanciata per mate_n)!")


if __name__ == "__main__":
    main()
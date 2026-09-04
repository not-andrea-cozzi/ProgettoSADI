#!/usr/bin/env python3
"""
inspect_pt.py

Carica un file .pt e stampa la sua struttura interna (campi/attributi).
Modifica la variabile FILE_PATH qui sotto con il percorso del tuo file.
"""

import torch
import logging
import sys

# ==================== CONFIGURAZIONE (MODIFICA QUI) ====================
FILE_PATH = "Dataset/games_val.pt"  # <-- Metti qui il tuo file
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("inspect")


def print_structure(data, indent=0):
    """Stampa ricorsivamente la struttura di un oggetto Python."""
    prefix = "  " * indent
    data_type = type(data)

    # Caso lista
    if isinstance(data, list):
        print(f"{prefix}Lista di {len(data)} elementi")
        if len(data) > 0:
            print(f"{prefix}Primo elemento:")
            print_structure(data[0], indent + 1)
        return

    # Caso dizionario
    if isinstance(data, dict):
        print(f"{prefix}Dizionario con {len(data)} chiavi:")
        for key, value in data.items():
            print(f"{prefix}  Chiave '{key}' -> {type(value)}")
            if isinstance(value, (dict, list, tuple)) and len(value) > 0:
                print(f"{prefix}    Primo livello:")
                print_structure(value, indent + 2)
        return

    # Caso tupla
    if isinstance(data, tuple):
        print(f"{prefix}Tupla di {len(data)} elementi")
        if len(data) > 0:
            print(f"{prefix}Primo elemento:")
            print_structure(data[0], indent + 1)
        return

    # Caso tensore PyTorch
    if isinstance(data, torch.Tensor):
        print(f"{prefix}Tensore shape={data.shape}, dtype={data.dtype}, device={data.device}")
        return

    # Caso oggetto generico (classe personalizzata)
    # Usiamo vars() per vedere gli attributi (__dict__)
    if hasattr(data, '__dict__'):
        attributi = vars(data)
        print(f"{prefix}Oggetto di tipo {data_type.__name__} con {len(attributi)} attributi:")
        for key, value in attributi.items():
            # Stampo anche un pezzetto del valore per capire cosa contiene
            value_repr = value
            if isinstance(value, torch.Tensor):
                value_repr = f"Tensor shape={value.shape}"
            elif isinstance(value, list):
                value_repr = f"Lista di {len(value)} elementi"
            elif isinstance(value, dict):
                value_repr = f"Dizionario con {len(value)} chiavi"
            print(f"{prefix}  {key}: {type(value).__name__} -> {value_repr}")
        return

    # Altro (int, str, float, None, ecc.)
    print(f"{prefix}Valore: {data} (tipo {data_type.__name__})")


def main():
    logger.info(f"Caricamento del file: {FILE_PATH}")

    if not os.path.exists(FILE_PATH):  # Aggiungo import os per sicurezza
        logger.error(f"File non trovato: {FILE_PATH}")
        sys.exit(1)

    # Carica il file
    data = torch.load(FILE_PATH, weights_only=False)

    print("\n" + "="*60)
    print(f"STRUTTURA DEL FILE: {FILE_PATH}")
    print("="*60)

    print(f"Tipo radice: {type(data)}")
    print_structure(data)

    print("\n" + "="*60)
    logger.info("Ispezione completata.")


if __name__ == "__main__":
    import os  # import aggiunto per il check di esistenza
    main()
import math
import shutil
import torch

IN_PATH = "Dataset/Holdout/external_holdout.pt"
BACKUP_PATH = "Dataset/Holdout/external_holdout_ORIGINALE_BUGGATO.pt"
CLOCK_SECONDS = 6.66  # media avg_time_by_rating.json fascia 1000-1600
CLOCK_CAP_SECONDS = 600.0  # stessa costante di GraphBuilder


def clock_norm(clock_seconds: float) -> float:
    cap = CLOCK_CAP_SECONDS
    denom = math.log1p(cap)
    return min(math.log1p(max(clock_seconds, 0.0)) / denom, 1.0)


def main():
    print(f"Backup del file originale in {BACKUP_PATH}...")
    shutil.copy(IN_PATH, BACKUP_PATH)

    print(f"Caricamento {IN_PATH}...")
    data_list = torch.load(IN_PATH, weights_only=False)
    print(f"Caricate {len(data_list)} posizioni.")

    norm_value = clock_norm(CLOCK_SECONDS)
    print(f"clock_seconds={CLOCK_SECONDS} -> clock_norm={norm_value:.6f}")

    for d in data_list:
        d.x[:, 3] = norm_value

    print(f"Salvataggio {IN_PATH} corretto...")
    torch.save(data_list, IN_PATH)
    print("Fatto.")


if __name__ == "__main__":
    main()

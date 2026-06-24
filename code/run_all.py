from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

DATASETS = {
    "german_numeric": "german-data-numeric",
    "south_german": "SouthGermanCredit.asc",
    "australian": "australian.dat",
    "taiwan": "default of credit card clients.xls",
    "give_me_some_credit": "cs-training.csv",
    "fico_heloc": "heloc_dataset_v1.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all code dataset experiments.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--datasets", nargs="*", choices=sorted(DATASETS), default=list(DATASETS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[10, 20, 30, 40, 50])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        folder = CODE_DIR / dataset
        data_path = args.data_root / DATASETS[dataset]
        command = [
            sys.executable,
            "run_experiment.py",
            "--data-path",
            str(data_path),
            "--seeds",
            *[str(seed) for seed in args.seeds],
        ]
        print(f"[{dataset}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=folder, check=True)


if __name__ == "__main__":
    main()

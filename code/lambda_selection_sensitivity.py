from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
DATASETS = [
    "german_numeric",
    "south_german",
    "australian",
    "taiwan",
    "give_me_some_credit",
    "fico_heloc",
]
MODEL_LABELS = {
    "convex": "Convex",
    "feature_specific": "Feature-specific",
    "feature_specific_weighting": "Feature-specific",
}
DATASET_LABELS = {
    "german_numeric": "German Numeric Full",
    "south_german": "South German Full",
    "australian": "Australian Full",
    "taiwan": "Taiwan Informative 10000",
    "give_me_some_credit": "Give Me Some Credit Informative 10000",
    "fico_heloc": "FICO HELOC Full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize lambda-selection candidates from code/<dataset>/results/lambda_validation_results.csv."
    )
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=DATASETS)
    parser.add_argument("--output-dir", type=Path, default=CODE_DIR / "results")
    parser.add_argument(
        "--candidate-sets",
        nargs="*",
        choices=["all", "nonzero"],
        default=["all", "nonzero"],
    )
    return parser.parse_args()


def normalize_experiment_name(value: str) -> str:
    return "feature_specific" if value == "feature_specific_weighting" else value


def load_validation_rows(datasets: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []
    for dataset in datasets:
        path = CODE_DIR / dataset / "results" / "lambda_validation_results.csv"
        if not path.exists():
            missing.append(path)
            continue
        frame = pd.read_csv(path)
        frame["dataset"] = dataset
        frame["dataset_label"] = DATASET_LABELS[dataset]
        frame["experiment"] = frame["experiment"].map(normalize_experiment_name)
        frame["model"] = frame["experiment"].map(MODEL_LABELS).fillna(frame["experiment"])
        frames.append(frame)
    if not frames:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "No lambda validation files were found. Run code/run_all.py first.\n"
            f"Expected files:\n{missing_text}"
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def select_rows(validation: pd.DataFrame, candidate_sets: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    criteria = {
        "validation_f1": ("F1", False),
        "validation_business_cost": ("Business Cost", True),
        "validation_balanced_accuracy": ("Balanced Accuracy", False),
    }
    for keys, group in validation.groupby(["dataset", "dataset_label", "seed", "experiment", "model"], sort=False):
        dataset, label, seed, experiment, model = keys
        for candidate_set in candidate_sets:
            candidates = group.copy()
            if candidate_set == "nonzero":
                candidates = candidates[pd.to_numeric(candidates["lambda"], errors="coerce").gt(0.0)]
            if candidates.empty:
                continue
            for criterion, (column, ascending) in criteria.items():
                selected = candidates.sort_values(
                    [column, "lambda"],
                    ascending=[ascending, True],
                ).iloc[0]
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": label,
                        "seed": seed,
                        "experiment": experiment,
                        "model": model,
                        "candidate_set": candidate_set,
                        "selection_criterion": criterion,
                        "selected_lambda": float(selected["lambda"]),
                        "validation_F1": float(selected["F1"]),
                        "validation_Balanced Accuracy": float(selected["Balanced Accuracy"]),
                        "validation_Business Cost": float(selected["Business Cost"]),
                        "validation_FN": float(selected["FN"]),
                        "validation_FP": float(selected["FP"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = raw.groupby(["dataset", "dataset_label", "experiment", "model", "candidate_set", "selection_criterion"], sort=False)
    for keys, group in grouped:
        dataset, label, experiment, model, candidate_set, criterion = keys
        row: dict[str, Any] = {
            "dataset": dataset,
            "dataset_label": label,
            "experiment": experiment,
            "model": model,
            "candidate_set": candidate_set,
            "selection_criterion": criterion,
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ", ".join(str(seed) for seed in sorted(group["seed"].unique())),
            "selected_lambda_values": ", ".join(f"{value:.2f}" for value in group.sort_values("seed")["selected_lambda"]),
        }
        for metric in ["validation_F1", "validation_Balanced Accuracy", "validation_Business Cost", "validation_FN", "validation_FP"]:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = load_validation_rows(args.datasets)
    raw = select_rows(validation, args.candidate_sets)
    raw_path = args.output_dir / "lambda_selection_sensitivity_raw.csv"
    summary_path = args.output_dir / "lambda_selection_sensitivity_summary.csv"
    raw.to_csv(raw_path, index=False)
    summarize(raw).to_csv(summary_path, index=False)
    print(f"raw_results={raw_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()

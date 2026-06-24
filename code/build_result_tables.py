from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
SEEDS = [10, 20, 30, 40, 50]
DATASETS = [
    "german_numeric",
    "south_german",
    "australian",
    "taiwan",
    "give_me_some_credit",
    "fico_heloc",
]
MODEL_LABELS = {
    "baseline": "Baseline QSVC",
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
        description="Build result tables from code/<dataset>/results outputs."
    )
    parser.add_argument("--output-dir", type=Path, default=CODE_DIR / "results")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=DATASETS)
    return parser.parse_args()


def normalize_experiment_name(value: str) -> str:
    return "feature_specific" if value == "feature_specific_weighting" else value


def normalize_frame(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    frame = frame.copy()
    if "dataset" not in frame.columns:
        frame["dataset"] = dataset_name
    frame["dataset"] = frame["dataset"].fillna(dataset_name)
    frame["dataset_label"] = frame["dataset"].map(DATASET_LABELS).fillna(frame["dataset"])
    frame["experiment"] = frame["experiment"].map(normalize_experiment_name)
    frame["model"] = frame["experiment"].map(MODEL_LABELS).fillna(frame["experiment"])
    if "selected_lambda" not in frame.columns:
        frame["selected_lambda"] = frame.get("lambda", 0.0)
    if "Business Cost" not in frame.columns and "business_cost" in frame.columns:
        frame["Business Cost"] = frame["business_cost"]
    for column in ["FN", "FP", "Business Cost", "F1", "Precision", "Recall", "Balanced Accuracy", "Accuracy"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_main_raw(datasets: list[str], seeds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []
    for dataset in datasets:
        path = CODE_DIR / dataset / "results" / "raw_results.csv"
        if not path.exists():
            missing.append(path)
            continue
        frame = pd.read_csv(path)
        if "seed" in frame.columns:
            frame = frame[pd.to_numeric(frame["seed"], errors="coerce").isin(seeds)]
        frames.append(normalize_frame(frame, dataset))
    if not frames:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "No dataset raw result files were found. Run code/run_all.py first.\n"
            f"Expected files:\n{missing_text}"
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns = ["Accuracy", "F1", "Precision", "Recall", "Balanced Accuracy", "FN", "FP", "Business Cost"]
    group_cols = ["dataset", "dataset_label", "experiment", "model"]
    for (dataset, label, experiment, model), group in raw.groupby(group_cols, sort=False):
        row: dict[str, Any] = {
            "dataset": dataset,
            "dataset_label": label,
            "experiment": experiment,
            "model": model,
            "n_seeds": int(group["seed"].nunique()) if "seed" in group.columns else len(group),
            "seeds": ", ".join(str(seed) for seed in sorted(group["seed"].unique())) if "seed" in group.columns else "",
            "selected_lambda_values": ", ".join(
                f"{value:.2f}" for value in group.sort_values("seed")["selected_lambda"].astype(float)
            )
            if "seed" in group.columns
            else "",
        }
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    baseline = summary[summary["experiment"].eq("baseline")][["dataset", "Business Cost_mean"]].rename(
        columns={"Business Cost_mean": "baseline_business_cost_mean"}
    )
    summary = summary.merge(baseline, on="dataset", how="left")
    summary["relative_cost_reduction_from_means_pct"] = (
        (summary["baseline_business_cost_mean"] - summary["Business Cost_mean"])
        / summary["baseline_business_cost_mean"]
        * 100.0
    )
    return summary


def cost_ratio_sensitivity(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ratio in [2, 5, 10]:
        work = raw.copy()
        work["ratio_cost"] = ratio * work["FN"] + work["FP"]
        baseline = work[work["experiment"].eq("baseline")][["dataset", "seed", "ratio_cost"]].rename(
            columns={"ratio_cost": "baseline_ratio_cost"}
        )
        work = work.merge(baseline, on=["dataset", "seed"], how="left")
        work["ratio_rcr_pct"] = (work["baseline_ratio_cost"] - work["ratio_cost"]) / work["baseline_ratio_cost"] * 100.0
        for (dataset, label, experiment, model), group in work.groupby(
            ["dataset", "dataset_label", "experiment", "model"], sort=False
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": label,
                    "cost_ratio": f"{ratio}FN+FP",
                    "experiment": experiment,
                    "model": model,
                    "cost_mean": float(group["ratio_cost"].mean()),
                    "cost_std": float(group["ratio_cost"].std(ddof=1)) if len(group) > 1 else 0.0,
                    "relative_cost_reduction_pct_mean": float(group["ratio_rcr_pct"].mean()),
                    "relative_cost_reduction_pct_std": float(group["ratio_rcr_pct"].std(ddof=1)) if len(group) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_main_raw(args.datasets, args.seeds)
    summary = summarize_metrics(raw)
    sensitivity = cost_ratio_sensitivity(raw)

    outputs = {
        "manuscript_main_raw.csv": raw,
        "manuscript_main_summary.csv": summary,
        "cost_ratio_sensitivity.csv": sensitivity,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)
        write_markdown_table(frame, args.output_dir / filename.replace(".csv", ".md"))

    print(f"output_dir={args.output_dir}")
    for filename in outputs:
        print(args.output_dir / filename)


if __name__ == "__main__":
    main()

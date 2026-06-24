from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
PIPELINE_MODULE = None

DATA_FILES = {
    "german_numeric": "german-data-numeric",
    "south_german": "SouthGermanCredit.asc",
    "australian": "australian.dat",
    "taiwan": "default of credit card clients.xls",
    "give_me_some_credit": "cs-training.csv",
    "fico_heloc": "heloc_dataset_v1.csv",
}


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    experiment: str
    criterion: str | None


VARIANTS = (
    Variant("baseline", "Baseline QSVC", "baseline", None),
    Variant("convex_main", "Convex, main setting", "convex", None),
    Variant("feature_main", "Feature-specific, main setting", "feature_specific", None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize quantum state embeddings with t-SNE.")
    parser.add_argument("--datasets", nargs="*", choices=sorted(DATA_FILES), default=list(DATA_FILES))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=CODE_DIR / "quantum_tsne_results")
    parser.add_argument("--per-class", type=int, default=250, help="Maximum test-set samples per class for t-SNE.")
    parser.add_argument("--perplexity", type=float, default=30.0)
    return parser.parse_args()


def load_pipeline_module():
    global PIPELINE_MODULE
    if PIPELINE_MODULE is not None:
        return PIPELINE_MODULE
    module_path = CODE_DIR / "_shared" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pipeline module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline"] = module
    spec.loader.exec_module(module)
    PIPELINE_MODULE = module
    return module


def load_config(dataset: str, pipeline_module):
    sys.modules["pipeline"] = pipeline_module
    config_path = CODE_DIR / dataset / "config.py"
    spec = importlib.util.spec_from_file_location(f"{dataset}_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def parse_qsvc_shape(shape_name: str, pipeline):
    match = re.match(r"(?P<name>.+)_reps(?P<reps>\d+)_(?P<ent>.+)_C(?P<c>[0-9.]+)", str(shape_name))
    if not match:
        return None
    return pipeline.QSVCShape(
        match.group("name"),
        int(match.group("reps")),
        match.group("ent"),
        float(match.group("c")),
    )


def shape_from_main_row(row: pd.Series, pipeline):
    if "qsvc_shape" in row and pd.notna(row["qsvc_shape"]):
        parsed = parse_qsvc_shape(str(row["qsvc_shape"]), pipeline)
        if parsed is not None:
            return parsed
    if all(column in row and pd.notna(row[column]) for column in ["feature_map", "reps", "entanglement", "C"]):
        return pipeline.QSVCShape(
            str(row["feature_map"]),
            int(row["reps"]),
            str(row["entanglement"]),
            float(row["C"]),
        )
    return None


def selected_params(dataset: str, seed: int, variant: Variant, pipeline):
    main_path = CODE_DIR / "results" / "manuscript_main_raw.csv"
    frame = pd.read_csv(main_path)
    rows = frame[frame["dataset"].eq(dataset) & frame["seed"].eq(seed) & frame["experiment"].eq(variant.experiment)]
    if rows.empty:
        raise ValueError(f"No manuscript row for dataset={dataset}, seed={seed}, experiment={variant.experiment}")

    row = rows.iloc[0]
    selected_lambda = float(row["selected_lambda"]) if variant.experiment != "baseline" else 0.0

    shape_row = row
    shape_override = shape_from_main_row(shape_row, pipeline)
    if variant.experiment == "baseline" and shape_override is None:
        convex_rows = frame[frame["dataset"].eq(dataset) & frame["seed"].eq(seed) & frame["experiment"].eq("convex")]
        if not convex_rows.empty:
            shape_row = convex_rows.iloc[0]
            shape_override = shape_from_main_row(shape_row, pipeline)

    selected_c = float(shape_row["C"]) if "C" in shape_row and pd.notna(shape_row["C"]) else None
    return selected_lambda, selected_c, shape_override


def shape_for_variant(config, pipeline, variant: Variant, selected_c: float | None, shape_override=None):
    if shape_override is not None:
        return shape_override
    if variant.experiment == "feature_specific" and config.tune_c:
        shape = config.qsvc_shapes["feature_specific"]
    elif config.tune_c:
        shape = config.qsvc_shapes["convex"]
    else:
        shape = config.fixed_qsvc_shape
    return shape.with_c(float(selected_c)) if selected_c is not None else shape


def balanced_test_subset(split, pipeline, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    y_values = split.y_test.to_numpy()
    for class_value in sorted(np.unique(y_values)):
        indices = np.flatnonzero(y_values == class_value)
        take = min(per_class, len(indices))
        selected.extend(rng.choice(indices, size=take, replace=False).tolist())
    selected = np.array(sorted(selected), dtype=int)
    return pipeline.DatasetSplit(
        split.X_train,
        split.X_test.iloc[selected].reset_index(drop=True),
        split.y_train,
        split.y_test.iloc[selected].reset_index(drop=True),
    )


def quantum_distance_from_states(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fidelity = np.clip((np.abs(states @ states.conj().T) ** 2).real, 0.0, 1.0)
    distance = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - np.sqrt(fidelity))))
    np.fill_diagonal(distance, 0.0)
    return fidelity.astype(np.float32, copy=False), distance.astype(np.float32, copy=False)


def class_fidelity_stats(fidelity: np.ndarray, y: np.ndarray) -> dict[str, float]:
    same = y.reshape(-1, 1) == y.reshape(1, -1)
    not_diag = ~np.eye(len(y), dtype=bool)
    same_values = fidelity[same & not_diag]
    diff_values = fidelity[~same]
    return {
        "same_class_fidelity_mean": float(same_values.mean()),
        "different_class_fidelity_mean": float(diff_values.mean()),
        "same_minus_diff_fidelity": float(same_values.mean() - diff_values.mean()),
    }


def knn_purity(distance: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    k = min(k, len(y) - 1)
    masked = distance.copy()
    np.fill_diagonal(masked, np.inf)
    nearest = np.argpartition(masked, kth=k - 1, axis=1)[:, :k]
    return float(np.mean(y[nearest] == y.reshape(-1, 1)))


def run_tsne(distance: np.ndarray, seed: int, requested_perplexity: float) -> np.ndarray:
    n_samples = distance.shape[0]
    perplexity = min(float(requested_perplexity), max(5.0, (n_samples - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        perplexity=perplexity,
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    )
    return tsne.fit_transform(distance)


def plot_dataset(dataset_label: str, points: pd.DataFrame, output_path: Path) -> None:
    variants = list(points["variant"].drop_duplicates())
    fig, axes = plt.subplots(1, len(variants), figsize=(5.0 * len(variants), 4.5), constrained_layout=True)
    if len(variants) == 1:
        axes = [axes]
    colors = {0: "#2563eb", 1: "#dc2626"}
    labels = {0: "Good / non-default", 1: "Bad / default"}
    for ax, variant in zip(axes, variants):
        frame = points[points["variant"].eq(variant)]
        for class_value in [0, 1]:
            class_frame = frame[frame["class"].eq(class_value)]
            ax.scatter(
                class_frame["tsne_1"],
                class_frame["tsne_2"],
                s=14,
                alpha=0.68,
                linewidths=0,
                c=colors[class_value],
                label=labels[class_value],
            )
        ax.set_title(variant, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9, markerscale=1.2)
    fig.suptitle(f"{dataset_label}: t-SNE of quantum state embeddings", fontsize=13)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline_module()
    summary_rows: list[dict[str, object]] = []

    for dataset in args.datasets:
        config = load_config(dataset, pipeline)
        data_path = args.data_root / DATA_FILES[dataset]
        split = pipeline.make_train_test_split(
            config,
            data_path,
            seed=args.seed,
            sample_size=config.default_sample_size,
            subset_strategy=config.default_subset_strategy,
        )
        subset = balanced_test_subset(split, pipeline, per_class=args.per_class, seed=args.seed)
        dataset_points: list[pd.DataFrame] = []

        for variant in VARIANTS:
            selected_lambda, selected_c, shape_override = selected_params(dataset, args.seed, variant, pipeline)
            shape = shape_for_variant(config, pipeline, variant, selected_c, shape_override)
            if variant.experiment == "baseline":
                _, X_eval_input = pipeline.fit_baseline_inputs(subset, args.seed)
            else:
                _, X_eval_input = pipeline.fit_preprocess_risk_transform(
                    config,
                    subset,
                    args.seed,
                    variant.experiment,
                    selected_lambda,
                )
            states = pipeline.statevector_matrix(X_eval_input, shape)
            fidelity, distance = quantum_distance_from_states(states)
            coords = run_tsne(distance, args.seed, args.perplexity)
            y_values = subset.y_test.to_numpy()
            point_frame = pd.DataFrame(
                {
                    "dataset": dataset,
                    "dataset_label": config.display_name,
                    "seed": args.seed,
                    "variant": variant.label,
                    "experiment": variant.experiment,
                    "selected_lambda": selected_lambda,
                    "selected_C": selected_c,
                    "class": y_values,
                    "tsne_1": coords[:, 0],
                    "tsne_2": coords[:, 1],
                }
            )
            point_frame.to_csv(args.output_dir / f"{dataset}_{variant.key}_seed{args.seed}_tsne_points.csv", index=False)
            dataset_points.append(point_frame)

            metrics = class_fidelity_stats(fidelity, y_values)
            metrics["tsne_silhouette"] = float(silhouette_score(coords, y_values)) if len(np.unique(y_values)) > 1 else np.nan
            metrics["quantum_distance_silhouette"] = (
                float(silhouette_score(distance, y_values, metric="precomputed")) if len(np.unique(y_values)) > 1 else np.nan
            )
            metrics["knn10_purity_quantum_distance"] = knn_purity(distance, y_values, k=10)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": config.display_name,
                    "seed": args.seed,
                    "variant": variant.label,
                    "experiment": variant.experiment,
                    "selected_lambda": selected_lambda,
                    "selected_C": selected_c,
                    "n_samples": len(y_values),
                    "n_good": int(np.sum(y_values == 0)),
                    "n_bad": int(np.sum(y_values == 1)),
                    **metrics,
                }
            )

        dataset_points_frame = pd.concat(dataset_points, ignore_index=True)
        plot_path = args.output_dir / f"{dataset}_quantum_tsne_seed{args.seed}.png"
        plot_dataset(config.display_name, dataset_points_frame, plot_path)
        print(f"plot={plot_path}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / f"quantum_tsne_summary_seed{args.seed}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()

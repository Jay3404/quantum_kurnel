from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
DEFAULT_DATASETS = [
    "german_numeric",
    "south_german",
    "australian",
    "taiwan",
    "give_me_some_credit",
    "fico_heloc",
]
DATA_FILES = {
    "german_numeric": "german-data-numeric",
    "south_german": "SouthGermanCredit.asc",
    "australian": "australian.dat",
    "taiwan": "default of credit card clients.xls",
    "give_me_some_credit": "cs-training.csv",
    "fico_heloc": "heloc_dataset_v1.csv",
}
MODEL_ORDER = [
    "Logistic Regression",
    "RBF-SVM",
    "Random Forest",
    "XGBoost",
    "Classical SVC on PCA features",
]
_PIPELINE_MODULE = None
_PIPELINE_LOCK = Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Appendix classical reference baselines.")
    parser.add_argument("--datasets", nargs="*", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=CODE_DIR / "classical_baselines" / "results")
    parser.add_argument("--seeds", nargs="*", type=int, default=[10, 20, 30, 40, 50])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--subset-strategy", choices=["none", "random", "informative"], default=None)
    parser.add_argument(
        "--input-space",
        choices=["pca7", "original"],
        default="pca7",
        help="Use PCA 7 inputs for all models by default to match the main QSVC preprocessing.",
    )
    return parser.parse_args()


def load_pipeline_module():
    global _PIPELINE_MODULE
    with _PIPELINE_LOCK:
        if _PIPELINE_MODULE is not None:
            return _PIPELINE_MODULE
        module_path = CODE_DIR / "_shared" / "pipeline.py"
        spec = importlib.util.spec_from_file_location("pipeline", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load pipeline module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["pipeline"] = module
        spec.loader.exec_module(module)
        _PIPELINE_MODULE = module
        return _PIPELINE_MODULE


def load_config(dataset: str, pipeline_module):
    sys.modules["pipeline"] = pipeline_module
    config_path = CODE_DIR / dataset / "config.py"
    spec = importlib.util.spec_from_file_location(f"{dataset}_final_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def score_for_bad(model: Any, X: Any, bad_label: int) -> np.ndarray:
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) == 2 and list(classes)[1] != bad_label:
            score = -score
        return np.asarray(score)
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        return model.predict_proba(X)[:, classes.index(bad_label)]
    raise ValueError(f"Model does not expose decision_function or predict_proba: {type(model)}")


def build_xgboost(y_train: pd.Series, seed: int):
    from xgboost import XGBClassifier

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = negatives / positives if positives else 1.0
    return make_pipeline(
        SimpleImputer(strategy="median"),
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=1,
        ),
    )


def classical_models(y_train: pd.Series, seed: int) -> dict[str, Any]:
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "RBF-SVM": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        ),
        "XGBoost": build_xgboost(y_train, seed),
    }


def pca_models(y_train: pd.Series, seed: int) -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        ),
        "RBF-SVM": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "XGBoost": build_xgboost(y_train, seed).named_steps["xgbclassifier"],
        "Classical SVC on PCA features": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed),
    }


def evaluate_one(
    dataset: str,
    seed: int,
    data_root: Path,
    input_space: str,
    sample_size_override: int | None,
    subset_strategy_override: str | None,
) -> list[dict[str, Any]]:
    pipeline = load_pipeline_module()
    config = load_config(dataset, pipeline)
    data_path = data_root / DATA_FILES[dataset]
    sample_size = config.default_sample_size if sample_size_override is None else sample_size_override
    subset_strategy = config.default_subset_strategy if subset_strategy_override is None else subset_strategy_override
    split = pipeline.make_train_test_split(
        config,
        data_path=data_path,
        seed=seed,
        sample_size=sample_size,
        subset_strategy=subset_strategy,
    )

    rows: list[dict[str, Any]] = []
    if input_space == "pca7":
        X_train_pca, X_test_pca = pipeline.fit_pca7(split, seed)
        for model_name, model in pca_models(split.y_train, seed).items():
            started = time.perf_counter()
            model.fit(X_train_pca, split.y_train)
            y_pred = model.predict(X_test_pca)
            y_score = score_for_bad(model, X_test_pca, pipeline.BAD_LABEL)
            metrics = pipeline.calculate_metrics(split.y_test, y_pred, y_score)
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_name": dataset,
                    "display_name": config.display_name,
                    "seed": seed,
                    "sample_size": sample_size if sample_size > 0 else "full",
                    "subset_strategy": subset_strategy,
                    "model": model_name,
                    "input_space": "pca_7",
                    "runtime_seconds": round(time.perf_counter() - started, 3),
                    **metrics,
                }
            )
        return rows

    for model_name, model in classical_models(split.y_train, seed).items():
        started = time.perf_counter()
        model.fit(split.X_train, split.y_train)
        y_pred = model.predict(split.X_test)
        y_score = score_for_bad(model, split.X_test, pipeline.BAD_LABEL)
        metrics = pipeline.calculate_metrics(split.y_test, y_pred, y_score)
        rows.append(
            {
                "dataset": dataset,
                "dataset_name": dataset,
                "display_name": config.display_name,
                "seed": seed,
                "sample_size": sample_size if sample_size > 0 else "full",
                "subset_strategy": subset_strategy,
                "model": model_name,
                "input_space": "original_features",
                "runtime_seconds": round(time.perf_counter() - started, 3),
                **metrics,
            }
        )

    started = time.perf_counter()
    X_train_pca, X_test_pca = pipeline.fit_pca7(split, seed)
    pca_svc = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed)
    pca_svc.fit(X_train_pca, split.y_train)
    y_pred = pca_svc.predict(X_test_pca)
    y_score = score_for_bad(pca_svc, X_test_pca, pipeline.BAD_LABEL)
    metrics = pipeline.calculate_metrics(split.y_test, y_pred, y_score)
    rows.append(
        {
            "dataset": dataset,
            "dataset_name": dataset,
            "display_name": config.display_name,
            "seed": seed,
            "sample_size": sample_size if sample_size > 0 else "full",
            "subset_strategy": subset_strategy,
            "model": "Classical SVC on PCA features",
            "input_space": "pca_7",
            "runtime_seconds": round(time.perf_counter() - started, 3),
            **metrics,
        }
    )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "Accuracy",
        "F1",
        "Precision",
        "Recall",
        "Specificity",
        "Balanced Accuracy",
        "ROC-AUC",
        "PR-AUC",
        "FN",
        "FP",
        "Business Cost",
    ]
    rows = []
    for (dataset, display_name, model), group in raw.groupby(["dataset", "display_name", "model"], sort=False):
        row = {
            "dataset": dataset,
            "display_name": display_name,
            "model": model,
            "model_order": MODEL_ORDER.index(model),
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ", ".join(str(seed) for seed in sorted(group["seed"].unique())),
            "sample_size": group["sample_size"].iloc[0],
            "subset_strategy": group["subset_strategy"].iloc[0],
            "input_space": group["input_space"].iloc[0],
        }
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "model_order"]).drop(columns=["model_order"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (dataset, seed, args.data_root, args.input_space, args.sample_size, args.subset_strategy)
        for dataset in args.datasets
        for seed in args.seeds
    ]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(evaluate_one, dataset, seed, data_root, input_space, sample_size, subset_strategy): (dataset, seed)
            for dataset, seed, data_root, input_space, sample_size, subset_strategy in jobs
        }
        for future in as_completed(futures):
            dataset, seed = futures[future]
            result_rows = future.result()
            rows.extend(result_rows)
            print(f"completed dataset={dataset} seed={seed} rows={len(result_rows)}", flush=True)

    raw = pd.DataFrame(rows).sort_values(["dataset", "seed", "model"])
    summary = summarize(raw)
    raw_path = args.output_dir / "raw_results.csv"
    summary_path = args.output_dir / "summary_mean_std.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"raw_results={raw_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()

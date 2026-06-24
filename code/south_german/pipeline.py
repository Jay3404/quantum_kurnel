from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TEST_SIZE = 0.2
INNER_VALIDATION_SIZE = 0.2
PCA_DIM = 7
OOF_SPLITS = 5
BAD_LABEL = 1
LAMBDA_VALUES = [round(value * 0.05, 2) for value in range(21)]
DEFAULT_C_GRID = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
LAMBDA_SELECTION_NOTE = (
    "Lambda was selected using inner validation within the training set. "
    "The held-out test set was used only once for final evaluation and was not involved in lambda selection."
)


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class QSVCShape:
    feature_map_name: str = "zz"
    reps: int = 1
    entanglement: str = "full"
    c_value: float = 1.0

    def with_c(self, c_value: float) -> "QSVCShape":
        return QSVCShape(
            feature_map_name=self.feature_map_name,
            reps=self.reps,
            entanglement=self.entanglement,
            c_value=float(c_value),
        )

    @property
    def label(self) -> str:
        return f"{self.feature_map_name}_reps{self.reps}_{self.entanglement}_C{self.c_value:.4f}"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name: str
    display_name: str
    loader: str
    default_data_path: Path
    default_sample_size: int = 0
    default_subset_strategy: str = "none"
    default_seeds: tuple[int, ...] = (10, 20, 30, 40, 50)
    tune_c: bool = True
    c_grid: tuple[float, ...] = tuple(DEFAULT_C_GRID)
    fixed_qsvc_shape: QSVCShape = QSVCShape("zz", 1, "linear", 1.0)
    qsvc_shapes: dict[str, QSVCShape] = field(default_factory=lambda: {
        "convex": QSVCShape("zz", 1, "full", 1.0),
        "feature_specific": QSVCShape("zz", 1, "full", 1.0),
    })
    convex_risk_model: str = "logistic_regression"
    feature_risk_model: str = "logistic_regression"
    lr_params: dict[str, object] = field(default_factory=lambda: {"C": 1.0, "class_weight": None})
    xgb_params: dict[str, object] = field(default_factory=dict)
    experiments: tuple[str, ...] = ("baseline", "convex", "feature_specific")


def parse_args(config: ExperimentConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"QSVC experiment for {config.display_name}")
    parser.add_argument("--data-path", type=Path, default=config.default_data_path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(config.default_seeds))
    parser.add_argument("--sample-size", type=int, default=config.default_sample_size)
    parser.add_argument("--subset-strategy", choices=["none", "random", "informative"], default=config.default_subset_strategy)
    parser.add_argument("--experiments", nargs="*", choices=["baseline", "convex", "feature_specific"], default=list(config.experiments))
    parser.add_argument("--lambda-values", nargs="*", type=float, default=LAMBDA_VALUES)
    parser.add_argument("--c-grid", nargs="*", type=float, default=list(config.c_grid))
    return parser.parse_args()


def load_dataset(config: ExperimentConfig, data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {data_path}. Pass --data-path /path/to/file.")

    if config.loader == "german_numeric":
        df = pd.read_csv(data_path, sep=r"\s+", header=None)
        X = df.iloc[:, :-1].copy().reset_index(drop=True)
        X.columns = [f"f{index:02d}" for index in range(1, X.shape[1] + 1)]
        y = encode_bad_labels(df.iloc[:, -1].astype(int) == 2)
        return X, y

    if config.loader == "south_german":
        df = pd.read_csv(data_path, sep=r"\s+")
        y = encode_bad_labels(df["kredit"].astype(int) == 0)
        X = df.drop(columns=["kredit"]).copy().reset_index(drop=True)
        return X, y

    if config.loader == "australian":
        df = pd.read_csv(data_path, sep=r"\s+", header=None)
        X = df.iloc[:, :-1].copy().reset_index(drop=True)
        X.columns = [f"a{index:02d}" for index in range(1, X.shape[1] + 1)]
        y = encode_bad_labels(df.iloc[:, -1].astype(int) == 0)
        return X, y

    if config.loader == "taiwan":
        df = pd.read_excel(data_path, header=1)
        y = encode_bad_labels(df["default payment next month"].astype(int) == 1)
        X = df.drop(columns=["ID", "default payment next month"]).copy().reset_index(drop=True)
        return X, y

    if config.loader == "give_me_some_credit":
        df = pd.read_csv(data_path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        y = df["SeriousDlqin2yrs"].astype(int).reset_index(drop=True)
        X = df.drop(columns=["SeriousDlqin2yrs"]).copy().reset_index(drop=True)
        return X, y

    if config.loader == "fico_heloc":
        df = pd.read_csv(data_path)
        risk = df["RiskPerformance"].astype(str).str.strip().str.lower()
        y = risk.map({"bad": 1, "good": 0})
        if y.isna().any():
            raise ValueError("Unexpected HELOC RiskPerformance values; expected Good/Bad.")
        X = df.drop(columns=["RiskPerformance"]).replace([-9.0, -8.0, -7.0], np.nan)
        return X.reset_index(drop=True), y.astype(int).reset_index(drop=True)

    raise ValueError(f"Unknown loader: {config.loader}")


def encode_bad_labels(bad_mask: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(np.where(np.asarray(bad_mask, dtype=bool), BAD_LABEL, 1 - BAD_LABEL)).astype(int).reset_index(drop=True)


def make_train_test_split(
    config: ExperimentConfig,
    data_path: Path,
    seed: int,
    sample_size: int,
    subset_strategy: str,
) -> DatasetSplit:
    np.random.seed(seed)
    X, y = load_dataset(config, data_path)
    X, y = select_subset(X, y, sample_size=sample_size, subset_strategy=subset_strategy, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=seed,
    )
    return DatasetSplit(
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def select_subset(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int,
    subset_strategy: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if sample_size <= 0 or sample_size >= len(y):
        return X.reset_index(drop=True), y.reset_index(drop=True)

    if subset_strategy == "random":
        X_sub, _, y_sub, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            stratify=y,
            random_state=seed,
        )
        return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)

    if subset_strategy == "informative":
        return informative_stratified_subset(X, y, sample_size, seed)

    raise ValueError("sample_size > 0 requires --subset-strategy random or informative.")


def informative_stratified_subset(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=min(PCA_DIM, X.shape[1]), random_state=seed)
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    pca_scores = pca.fit_transform(X_scaled)
    leverage_scores = np.sum(np.square(pca_scores), axis=1)

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    total_count = len(y)

    for class_value in sorted(y.unique()):
        class_indices = np.flatnonzero(y.to_numpy() == class_value)
        target_count = int(round(sample_size * len(class_indices) / total_count))
        target_count = min(max(target_count, 1), len(class_indices))
        ranking = pd.DataFrame(
            {
                "index": class_indices,
                "leverage": leverage_scores[class_indices],
                "tie": rng.uniform(size=len(class_indices)),
            }
        ).sort_values(["leverage", "tie"], ascending=[False, True])
        selected_indices.extend(ranking.head(target_count)["index"].tolist())

    if len(selected_indices) < sample_size:
        remaining = sorted(set(range(len(y))) - set(selected_indices))
        remaining_ranking = pd.DataFrame(
            {
                "index": remaining,
                "leverage": leverage_scores[remaining],
            }
        ).sort_values("leverage", ascending=False)
        selected_indices.extend(remaining_ranking.head(sample_size - len(selected_indices))["index"].tolist())
    elif len(selected_indices) > sample_size:
        selected_indices = sorted(selected_indices, key=lambda index: leverage_scores[index], reverse=True)[:sample_size]

    selected_indices = sorted(selected_indices)
    return X.iloc[selected_indices].reset_index(drop=True), y.iloc[selected_indices].reset_index(drop=True)


def build_inner_validation_split(split: DatasetSplit, seed: int) -> DatasetSplit:
    X_inner_train, X_inner_valid, y_inner_train, y_inner_valid = train_test_split(
        split.X_train,
        split.y_train,
        test_size=INNER_VALIDATION_SIZE,
        stratify=split.y_train,
        random_state=seed,
    )
    return DatasetSplit(
        X_inner_train.reset_index(drop=True),
        X_inner_valid.reset_index(drop=True),
        y_inner_train.reset_index(drop=True),
        y_inner_valid.reset_index(drop=True),
    )


def fit_pca7(split: DatasetSplit, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_columns = list(split.X_train.columns)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=PCA_DIM, random_state=seed)

    X_train_imputed = pd.DataFrame(imputer.fit_transform(split.X_train[numeric_columns]), columns=numeric_columns)
    X_test_imputed = pd.DataFrame(imputer.transform(split.X_test[numeric_columns]), columns=numeric_columns)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    columns = [f"pca_{index + 1:02d}" for index in range(PCA_DIM)]
    return pd.DataFrame(X_train_pca, columns=columns), pd.DataFrame(X_test_pca, columns=columns)


def scale_to_quantum_range(X_train_pca: pd.DataFrame, X_test_pca: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    min_values = X_train_pca.min(axis=0)
    ranges = (X_train_pca.max(axis=0) - min_values).replace(0, 1.0)
    X_train_unit = ((X_train_pca - min_values) / ranges).clip(0.0, 1.0)
    X_test_unit = ((X_test_pca - min_values) / ranges).clip(0.0, 1.0)
    return X_train_unit.to_numpy() * np.pi, X_test_unit.to_numpy() * np.pi


def risk_model_name(config: ExperimentConfig, experiment: str) -> str:
    if experiment == "convex":
        return config.convex_risk_model
    if experiment == "feature_specific":
        return config.feature_risk_model
    raise ValueError(f"Unknown risk-aware experiment: {experiment}")


def build_risk_model(config: ExperimentConfig, model_name: str, seed: int):
    if model_name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                random_state=seed,
                **config.lr_params,
            ),
        )

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            **config.xgb_params,
        )

    raise ValueError(f"Unknown risk model: {model_name}")


def predict_bad_probability(model, X: pd.DataFrame) -> np.ndarray:
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    return model.predict_proba(X)[:, list(classes).index(BAD_LABEL)]


def make_oof_risk(config: ExperimentConfig, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> np.ndarray:
    min_class_count = int(y_train.value_counts().min())
    n_splits = min(OOF_SPLITS, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_risk = np.zeros(len(y_train), dtype=float)
    for fit_idx, valid_idx in cv.split(X_train, y_train):
        model = build_risk_model(config, model_name, seed)
        model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        oof_risk[valid_idx] = predict_bad_probability(model, X_train.iloc[valid_idx])
    return oof_risk


def make_eval_risk(
    config: ExperimentConfig,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    model = build_risk_model(config, model_name, seed)
    model.fit(X_train, y_train)
    return predict_bad_probability(model, X_eval)


def fit_lr_weights(config: ExperimentConfig, X_train_pca: pd.DataFrame, y_train: pd.Series, seed: int) -> np.ndarray:
    model = build_risk_model(config, "logistic_regression", seed)
    model.fit(X_train_pca, y_train)
    coefficients = model.named_steps["logisticregression"].coef_.reshape(-1)
    absolute_coefficients = np.abs(coefficients)
    total = float(absolute_coefficients.sum())
    if total == 0.0:
        return np.ones_like(absolute_coefficients) / len(absolute_coefficients)
    return absolute_coefficients / total


def apply_risk_transform(
    experiment: str,
    X_train_pca: pd.DataFrame,
    X_eval_pca: pd.DataFrame,
    train_risk: np.ndarray,
    eval_risk: np.ndarray,
    lambda_value: float,
    weights: np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if experiment == "convex":
        X_train_values = X_train_pca.to_numpy() * (1.0 + lambda_value * train_risk.reshape(-1, 1))
        X_eval_values = X_eval_pca.to_numpy() * (1.0 + lambda_value * eval_risk.reshape(-1, 1))
    elif experiment == "feature_specific":
        if weights is None:
            raise ValueError("feature_specific requires logistic-regression feature weights.")
        weight_row = weights.reshape(1, -1)
        X_train_values = X_train_pca.to_numpy() * (1.0 + lambda_value * weight_row * train_risk.reshape(-1, 1))
        X_eval_values = X_eval_pca.to_numpy() * (1.0 + lambda_value * weight_row * eval_risk.reshape(-1, 1))
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    return pd.DataFrame(X_train_values, columns=X_train_pca.columns), pd.DataFrame(X_eval_values, columns=X_eval_pca.columns)


def fit_preprocess_risk_transform(
    config: ExperimentConfig,
    split: DatasetSplit,
    seed: int,
    experiment: str,
    lambda_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    X_train_pca, X_eval_pca = fit_pca7(split, seed)
    model_name = risk_model_name(config, experiment)
    train_risk = make_oof_risk(config, model_name, X_train_pca, split.y_train, seed)
    eval_risk = make_eval_risk(config, model_name, X_train_pca, split.y_train, X_eval_pca, seed)
    weights = fit_lr_weights(config, X_train_pca, split.y_train, seed) if experiment == "feature_specific" else None
    X_train_transformed, X_eval_transformed = apply_risk_transform(
        experiment,
        X_train_pca,
        X_eval_pca,
        train_risk,
        eval_risk,
        lambda_value,
        weights,
    )
    return scale_to_quantum_range(X_train_transformed, X_eval_transformed)


def fit_baseline_inputs(split: DatasetSplit, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X_train_pca, X_eval_pca = fit_pca7(split, seed)
    return scale_to_quantum_range(X_train_pca, X_eval_pca)


def build_feature_map(shape: QSVCShape, feature_dimension: int):
    if shape.feature_map_name == "zz":
        return ZZFeatureMap(feature_dimension=feature_dimension, reps=shape.reps, entanglement=shape.entanglement)
    if shape.feature_map_name == "pauli_zz":
        return PauliFeatureMap(
            feature_dimension=feature_dimension,
            reps=shape.reps,
            entanglement=shape.entanglement,
            paulis=["Z", "ZZ"],
        )
    raise ValueError(f"Unknown feature map: {shape.feature_map_name}")


def statevector_matrix(input_values: np.ndarray, shape: QSVCShape) -> np.ndarray:
    feature_map = build_feature_map(shape, feature_dimension=input_values.shape[1])
    ordered_parameters = getattr(feature_map, "ordered_parameters", None)
    parameters = list(ordered_parameters) if ordered_parameters is not None else list(feature_map.parameters)
    vectors = np.empty((len(input_values), 2 ** input_values.shape[1]), dtype=np.complex64)
    for row_index, row in enumerate(input_values):
        circuit = feature_map.assign_parameters(
            {parameter: float(value) for parameter, value in zip(parameters, row)},
            inplace=False,
        )
        vectors[row_index] = Statevector.from_instruction(circuit).data.astype(np.complex64, copy=False)
    return vectors


def kernel_from_states(left_states: np.ndarray, right_states: np.ndarray) -> np.ndarray:
    inner = left_states @ right_states.conj().T
    return np.clip((np.abs(inner) ** 2).real, 0.0, 1.0).astype(np.float32, copy=False)


def decision_score_for_bad(model: SVC, kernel: np.ndarray) -> np.ndarray:
    y_score = model.decision_function(kernel)
    if list(model.classes_)[1] != BAD_LABEL:
        y_score = -y_score
    return y_score


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1 - BAD_LABEL, BAD_LABEL]).ravel()
    y_true_bad = (np.asarray(y_true) == BAD_LABEL).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=BAD_LABEL, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=BAD_LABEL, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "F1": f1_score(y_true, y_pred, pos_label=BAD_LABEL, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true_bad, y_score),
        "PR-AUC": average_precision_score(y_true_bad, y_score),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Business Cost": float(5 * fn + fp),
    }


def run_qsvc(
    X_train_input: np.ndarray,
    X_eval_input: np.ndarray,
    y_train: pd.Series,
    y_eval: pd.Series,
    shape: QSVCShape,
) -> dict[str, float]:
    train_states = statevector_matrix(X_train_input, shape)
    eval_states = statevector_matrix(X_eval_input, shape)
    train_kernel = kernel_from_states(train_states, train_states)
    eval_kernel = kernel_from_states(eval_states, train_states)
    model = SVC(kernel="precomputed", C=shape.c_value, class_weight="balanced", random_state=42)
    model.fit(train_kernel, y_train)
    y_pred = model.predict(eval_kernel)
    y_score = decision_score_for_bad(model, eval_kernel)
    return calculate_metrics(y_eval, y_pred, y_score)


def evaluate_c_grid(
    X_train_input: np.ndarray,
    X_eval_input: np.ndarray,
    y_train: pd.Series,
    y_eval: pd.Series,
    shape: QSVCShape,
    lambda_value: float,
    c_grid: list[float],
) -> list[dict[str, float]]:
    train_states = statevector_matrix(X_train_input, shape)
    eval_states = statevector_matrix(X_eval_input, shape)
    train_kernel = kernel_from_states(train_states, train_states)
    eval_kernel = kernel_from_states(eval_states, train_states)
    rows = []
    for c_value in c_grid:
        model = SVC(kernel="precomputed", C=float(c_value), class_weight="balanced", random_state=42)
        model.fit(train_kernel, y_train)
        y_pred = model.predict(eval_kernel)
        y_score = decision_score_for_bad(model, eval_kernel)
        rows.append({"lambda": float(lambda_value), "C": float(c_value), **calculate_metrics(y_eval, y_pred, y_score)})
    return rows


def select_lambda_and_c(
    config: ExperimentConfig,
    split: DatasetSplit,
    seed: int,
    experiment: str,
    lambda_values: list[float],
    c_grid: list[float],
) -> tuple[float, QSVCShape, float, pd.DataFrame]:
    inner = build_inner_validation_split(split, seed)
    base_shape = config.qsvc_shapes[experiment]
    rows: list[dict[str, float]] = []
    for lambda_value in sorted(float(value) for value in lambda_values):
        X_inner_train, X_inner_valid = fit_preprocess_risk_transform(config, inner, seed, experiment, lambda_value)
        rows.extend(evaluate_c_grid(X_inner_train, X_inner_valid, inner.y_train, inner.y_test, base_shape, lambda_value, c_grid))
    validation = pd.DataFrame(rows)
    best = validation.sort_values(
        ["F1", "lambda", "C", "Recall", "Balanced Accuracy", "ROC-AUC"],
        ascending=[False, True, True, False, False, False],
    ).iloc[0]
    selected_lambda = float(best["lambda"])
    selected_shape = base_shape.with_c(float(best["C"]))
    return selected_lambda, selected_shape, float(best["F1"]), validation


def select_lambda_fixed_c(
    config: ExperimentConfig,
    split: DatasetSplit,
    seed: int,
    experiment: str,
    lambda_values: list[float],
    shape: QSVCShape,
) -> tuple[float, float, pd.DataFrame]:
    inner = build_inner_validation_split(split, seed)
    rows = []
    selected_lambda = None
    best_f1 = -np.inf
    for lambda_value in sorted(float(value) for value in lambda_values):
        X_inner_train, X_inner_valid = fit_preprocess_risk_transform(config, inner, seed, experiment, lambda_value)
        metrics = run_qsvc(X_inner_train, X_inner_valid, inner.y_train, inner.y_test, shape)
        rows.append({"lambda": float(lambda_value), "C": shape.c_value, **metrics})
        if metrics["F1"] > best_f1:
            best_f1 = float(metrics["F1"])
            selected_lambda = float(lambda_value)
    if selected_lambda is None:
        raise ValueError("lambda_values must contain at least one value.")
    return selected_lambda, best_f1, pd.DataFrame(rows)


def result_row(
    config: ExperimentConfig,
    seed: int,
    sample_size: int,
    subset_strategy: str,
    experiment: str,
    selected_lambda: float,
    inner_validation_f1: float | None,
    shape: QSVCShape,
    metrics: dict[str, float],
    runtime_seconds: float,
) -> dict[str, object]:
    return {
        "dataset": config.dataset_name,
        "dataset_name": config.dataset_name,
        "display_name": config.display_name,
        "sample_size": sample_size if sample_size > 0 else "full",
        "subset_strategy": subset_strategy,
        "seed": seed,
        "experiment": experiment,
        "model": {"baseline": "Baseline QSVC", "convex": "Convex", "feature_specific": "Feature-specific"}[experiment],
        "feature_map": shape.feature_map_name,
        "reps": shape.reps,
        "entanglement": shape.entanglement,
        "C": shape.c_value,
        "lambda": selected_lambda,
        "selected_lambda": selected_lambda,
        "inner_validation_f1": inner_validation_f1,
        "runtime_seconds": round(runtime_seconds, 3),
        "lambda_selection_note": LAMBDA_SELECTION_NOTE,
        **metrics,
    }


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw = raw.drop(columns=[column for column in ["baseline_cost", "cost_reduction_pct"] if column in raw.columns])
    baseline_cost = raw[raw["experiment"] == "baseline"][["seed", "Business Cost"]].rename(columns={"Business Cost": "baseline_cost"})
    raw = raw.merge(baseline_cost, on="seed", how="left")
    raw["cost_reduction_pct"] = np.where(
        raw["experiment"] == "baseline",
        0.0,
        (raw["baseline_cost"] - raw["Business Cost"]) / raw["baseline_cost"] * 100.0,
    )
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
        "cost_reduction_pct",
    ]
    rows = []
    for (experiment, model), group in raw.groupby(["experiment", "model"], sort=False):
        row = {
            "experiment": experiment,
            "model": model,
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ", ".join(str(seed) for seed in sorted(group["seed"].unique())),
            "selected_lambda_values": ", ".join(f"{value:.2f}" for value in group.sort_values("seed")["selected_lambda"].tolist()),
        }
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_suite(config: ExperimentConfig, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    validation_frames: list[pd.DataFrame] = []

    for seed in args.seeds:
        split = make_train_test_split(config, args.data_path, seed, args.sample_size, args.subset_strategy)
        seed_rows: list[dict[str, object]] = []
        selected_convex_shape = config.fixed_qsvc_shape

        if "convex" in args.experiments:
            started_at = time.perf_counter()
            if config.tune_c:
                selected_lambda, selected_shape, inner_f1, validation = select_lambda_and_c(
                    config,
                    split,
                    seed,
                    "convex",
                    args.lambda_values,
                    args.c_grid,
                )
            else:
                selected_shape = config.fixed_qsvc_shape
                selected_lambda, inner_f1, validation = select_lambda_fixed_c(
                    config,
                    split,
                    seed,
                    "convex",
                    args.lambda_values,
                    selected_shape,
                )
            selected_convex_shape = selected_shape
            validation.insert(0, "seed", seed)
            validation.insert(1, "experiment", "convex")
            validation_frames.append(validation)
            X_train_input, X_test_input = fit_preprocess_risk_transform(config, split, seed, "convex", selected_lambda)
            metrics = run_qsvc(X_train_input, X_test_input, split.y_train, split.y_test, selected_shape)
            seed_rows.append(result_row(config, seed, args.sample_size, args.subset_strategy, "convex", selected_lambda, inner_f1, selected_shape, metrics, time.perf_counter() - started_at))

        if "baseline" in args.experiments:
            started_at = time.perf_counter()
            X_train_input, X_test_input = fit_baseline_inputs(split, seed)
            metrics = run_qsvc(X_train_input, X_test_input, split.y_train, split.y_test, selected_convex_shape)
            seed_rows.insert(0, result_row(config, seed, args.sample_size, args.subset_strategy, "baseline", 0.0, None, selected_convex_shape, metrics, time.perf_counter() - started_at))

        if "feature_specific" in args.experiments:
            started_at = time.perf_counter()
            if config.tune_c:
                selected_lambda, selected_shape, inner_f1, validation = select_lambda_and_c(
                    config,
                    split,
                    seed,
                    "feature_specific",
                    args.lambda_values,
                    args.c_grid,
                )
            else:
                selected_shape = config.fixed_qsvc_shape
                selected_lambda, inner_f1, validation = select_lambda_fixed_c(
                    config,
                    split,
                    seed,
                    "feature_specific",
                    args.lambda_values,
                    selected_shape,
                )
            validation.insert(0, "seed", seed)
            validation.insert(1, "experiment", "feature_specific")
            validation_frames.append(validation)
            X_train_input, X_test_input = fit_preprocess_risk_transform(config, split, seed, "feature_specific", selected_lambda)
            metrics = run_qsvc(X_train_input, X_test_input, split.y_train, split.y_test, selected_shape)
            seed_rows.append(result_row(config, seed, args.sample_size, args.subset_strategy, "feature_specific", selected_lambda, inner_f1, selected_shape, metrics, time.perf_counter() - started_at))

        all_rows.extend(seed_rows)
        print(f"completed seed={seed} rows={len(seed_rows)}", flush=True)

    raw = pd.DataFrame(all_rows)
    baseline_cost = raw[raw["experiment"] == "baseline"][["seed", "Business Cost"]].rename(columns={"Business Cost": "baseline_cost"})
    raw = raw.merge(baseline_cost, on="seed", how="left")
    raw["cost_reduction_pct"] = np.where(
        raw["experiment"] == "baseline",
        0.0,
        (raw["baseline_cost"] - raw["Business Cost"]) / raw["baseline_cost"] * 100.0,
    )

    raw_path = args.output_dir / "raw_results.csv"
    summary_path = args.output_dir / "summary_mean_std.csv"
    validation_path = args.output_dir / "lambda_validation_results.csv"
    metadata_path = args.output_dir / "metadata.json"

    raw.to_csv(raw_path, index=False)
    summarize_results(raw).to_csv(summary_path, index=False)
    if validation_frames:
        pd.concat(validation_frames, ignore_index=True).to_csv(validation_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": config.dataset_name,
                "display_name": config.display_name,
                "data_path": str(args.data_path),
                "sample_size": args.sample_size if args.sample_size > 0 else "full",
                "subset_strategy": args.subset_strategy,
                "seeds": args.seeds,
                "lambda_values": args.lambda_values,
                "lambda_selection_note": LAMBDA_SELECTION_NOTE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"raw_results={raw_path}")
    print(f"summary={summary_path}")


def main(config: ExperimentConfig) -> None:
    args = parse_args(config)
    run_suite(config, args)

from __future__ import annotations

from pathlib import Path

from pipeline import ExperimentConfig, QSVCShape


ROOT = Path(__file__).resolve().parents[2]

CONFIG = ExperimentConfig(
    dataset_name="german_numeric",
    display_name="German Numeric Full",
    loader="german_numeric",
    default_data_path=ROOT / "data" / "german-data-numeric",
    default_sample_size=0,
    default_subset_strategy="none",
    tune_c=True,
    qsvc_shapes={
        "convex": QSVCShape("zz", 1, "full", 1.0),
        "feature_specific": QSVCShape("zz", 1, "full", 1.0),
    },
    convex_risk_model="xgboost",
    feature_risk_model="logistic_regression",
    lr_params={"C": 0.5384094201016693, "class_weight": "balanced"},
    xgb_params={
        "n_estimators": 250,
        "max_depth": 2,
        "learning_rate": 0.020301330868858616,
        "scale_pos_weight": 2.0872819046064457,
        "subsample": 0.9120572031542851,
        "colsample_bytree": 0.9187021504122962,
        "min_child_weight": 6.39889242680162,
        "gamma": 0.07404465173409036,
        "reg_alpha": 0.7169314570885452,
        "reg_lambda": 0.905541708337954,
        "n_jobs": 1,
    },
)

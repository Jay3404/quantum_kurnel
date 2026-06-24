from __future__ import annotations

from pathlib import Path

from pipeline import ExperimentConfig, QSVCShape


ROOT = Path(__file__).resolve().parents[2]

CONFIG = ExperimentConfig(
    dataset_name="taiwan",
    display_name="Taiwan Informative 10000",
    loader="taiwan",
    default_data_path=ROOT / "data" / "default of credit card clients.xls",
    default_sample_size=10000,
    default_subset_strategy="informative",
    tune_c=True,
    qsvc_shapes={
        "convex": QSVCShape("zz", 1, "full", 1.0),
        "feature_specific": QSVCShape("zz", 2, "linear", 1.0),
    },
    convex_risk_model="logistic_regression",
    feature_risk_model="logistic_regression",
    lr_params={"C": 0.005337032762603957, "class_weight": "balanced"},
    xgb_params={
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.0761909360685659,
        "scale_pos_weight": 3.087057856161148,
        "subsample": 0.7580441126574113,
        "colsample_bytree": 0.8081933572384011,
        "min_child_weight": 1.2011605302193642,
        "gamma": 0.490425361367648,
        "reg_alpha": 1.5455492111035936,
        "reg_lambda": 3.025803015490338,
        "n_jobs": 1,
    },
)

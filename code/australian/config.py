from __future__ import annotations

from pathlib import Path

from pipeline import ExperimentConfig, QSVCShape


ROOT = Path(__file__).resolve().parents[2]

CONFIG = ExperimentConfig(
    dataset_name="australian",
    display_name="Australian Full",
    loader="australian",
    default_data_path=ROOT / "data" / "australian.dat",
    default_sample_size=0,
    default_subset_strategy="none",
    tune_c=True,
    qsvc_shapes={
        "convex": QSVCShape("pauli_zz", 1, "linear", 1.0),
        "feature_specific": QSVCShape("zz", 1, "full", 1.0),
    },
    convex_risk_model="xgboost",
    feature_risk_model="logistic_regression",
    lr_params={"C": 0.22737894714575696, "class_weight": None},
    xgb_params={
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.06434672169189819,
        "scale_pos_weight": 1.0,
        "subsample": 0.9578949625586654,
        "colsample_bytree": 0.8971435912982683,
        "min_child_weight": 1.1905331847383778,
        "gamma": 0.9002148367830045,
        "reg_alpha": 0.8031289227965891,
        "reg_lambda": 3.092507594310814,
        "n_jobs": 1,
    },
)

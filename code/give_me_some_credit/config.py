from __future__ import annotations

from pathlib import Path

from pipeline import ExperimentConfig, QSVCShape


ROOT = Path(__file__).resolve().parents[2]

CONFIG = ExperimentConfig(
    dataset_name="give_me_some_credit",
    display_name="Give Me Some Credit Informative 10000",
    loader="give_me_some_credit",
    default_data_path=ROOT / "data" / "cs-training.csv",
    default_sample_size=10000,
    default_subset_strategy="informative",
    tune_c=False,
    fixed_qsvc_shape=QSVCShape("zz", 1, "linear", 1.0),
    convex_risk_model="logistic_regression",
    feature_risk_model="logistic_regression",
    lr_params={"C": 1.0, "class_weight": "balanced"},
)

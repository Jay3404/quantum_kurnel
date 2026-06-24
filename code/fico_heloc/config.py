from __future__ import annotations

from pathlib import Path

from pipeline import ExperimentConfig, QSVCShape


ROOT = Path(__file__).resolve().parents[2]

CONFIG = ExperimentConfig(
    dataset_name="fico_heloc",
    display_name="FICO HELOC Full",
    loader="fico_heloc",
    default_data_path=ROOT / "data" / "heloc_dataset_v1.csv",
    default_sample_size=0,
    default_subset_strategy="none",
    tune_c=False,
    fixed_qsvc_shape=QSVCShape("zz", 1, "linear", 1.0),
    convex_risk_model="logistic_regression",
    feature_risk_model="logistic_regression",
    lr_params={"C": 1.0, "class_weight": None},
)

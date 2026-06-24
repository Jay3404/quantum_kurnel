# Result Bundle

This document explains how to read the paper-facing result files in this folder. It does not replace the raw CSV outputs; it points readers to the tables that correspond to the experiment protocol.

## Primary Result Files

| File | Purpose |
| --- | --- |
| `results/manuscript_main_summary.md` | Main QSVC comparison across baseline, convex risk-aware, and feature-specific risk-aware models |
| `results/manuscript_main_raw.csv` | Per-seed raw main QSVC results |
| `results/classical_reference_with_qsvc.md` | Classical baselines beside QSVC variants |
| `results/cost_ratio_sensitivity.md` | Sensitivity to alternative FN:FP cost ratios |
| `results/lambda_selection_sensitivity_summary.csv` | Summary of lambda-selection behavior |
| `supplementary_results/` | Additional Taiwan and Give Me Some Credit sensitivity tables |
| `quantum_tsne_visualization.py` | Script for regenerating quantum-kernel t-SNE visualization artifacts locally |

## Main Reading of the Results

The results support a conditional claim:

| Dataset | Risk-aware result pattern |
| --- | --- |
| German Numeric | Convex and feature-specific risk-aware QSVC reduce business cost relative to baseline QSVC |
| South German | Convex risk-aware QSVC improves F1, recall, balanced accuracy, and business cost relative to baseline QSVC |
| Australian | Convex risk-aware QSVC improves business cost, but feature-specific weighting is worse than baseline |
| Taiwan | Convex risk-aware QSVC is close to baseline; cost improvement is not robust under the main 5FN+FP setting |
| Give Me Some Credit | Risk-aware QSVC gives only small movement and does not beat the strongest classical reference |
| FICO HELOC | QSVC variants are close, while classical Random Forest/XGBoost are strong references |

This is why the paper should describe the proposed method as a dataset-dependent risk-aware representation mechanism rather than a universally superior classifier.

## Business Cost Convention

The main table uses:

```text
Business Cost = 5 * FN + FP
```

where the bad/default class is treated as the positive class. The sensitivity table also reports `2FN+FP` and `10FN+FP`.

## Leakage-Control Note

The experiment protocol uses inner validation within the training split to choose lambda values. The held-out test split is used only once for final evaluation.

This note is also written into each experiment's `metadata.json` output:

```text
Lambda was selected using inner validation within the training set. The held-out test set was used only once for final evaluation and was not involved in lambda selection.
```

## Regenerating Summary Tables

After rerunning the dataset experiments, rebuild result tables from the repository root:

```bash
python code/build_result_tables.py
```

Run classical reference baselines:

```bash
python code/classical_baselines.py --data-root data --seeds 10 20 30 40 50
```

Run cost-ratio sensitivity:

```bash
python code/lambda_selection_sensitivity.py
```

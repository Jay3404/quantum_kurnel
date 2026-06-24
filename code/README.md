# Code

This folder is the paper-facing experiment package for the Soft Computing submission. It contains the reproducible QSVC protocol, classical reference baselines, result tables, and visualization scripts.

## Folder Map

| Path | Purpose |
| --- | --- |
| `_shared/pipeline.py` | Shared preprocessing, risk-score estimation, risk-aware transforms, QSVC kernel evaluation, lambda selection, and metrics |
| `german_numeric/` | Statlog German numeric runner |
| `south_german/` | South German Credit runner |
| `australian/` | Australian Credit Approval runner |
| `taiwan/` | Taiwan default runner |
| `give_me_some_credit/` | Give Me Some Credit runner |
| `fico_heloc/` | FICO HELOC runner |
| `classical_baselines.py` | Classical model reference experiments |
| `build_result_tables.py` | Rebuilds result tables after rerunning the dataset experiments |
| `results/` | Main generated result tables |
| `supplementary_results/` | Additional sensitivity tables used during paper analysis |
| `RESULTS.md` | Paper-facing guide to the result bundle |

Each dataset folder contains:

| File | Role |
| --- | --- |
| `run_experiment.py` | Dataset entry point |
| `config.py` | Dataset-specific loader, sampling, risk model, QSVC shape, and seed settings |
| `pipeline.py` | Thin wrapper that imports the shared pipeline |
| `requirements.txt` | Minimal Python package requirements |
| `README.md` | Dataset-specific command and notes |

## Experiment Protocol

Default seeds:

```bash
10 20 30 40 50
```

Default lambda grid:

```text
0.00, 0.05, ..., 1.00
```

Default quantum input dimension:

```text
PCA_DIM = 7
```

Main business-cost metric:

```text
Business Cost = 5 * FN + FP
```

Leakage-control statement:

```text
Lambda was selected using inner validation within the training set. The held-out test set was used only once for final evaluation and was not involved in lambda selection.
```

## Install

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r code/requirements.txt
```

If you use the existing Poetry setup instead:

```bash
poetry install
```

## Run One Dataset

```bash
cd code/australian
python run_experiment.py --data-path ../../data/australian.dat
```

## Run All QSVC Experiments

From the repository root:

```bash
python code/run_all.py --data-root data --seeds 10 20 30 40 50
```

## Outputs

Each dataset runner writes to its `results/` directory by default.

| Output | Description |
| --- | --- |
| `raw_results.csv` | Per-seed test metrics |
| `summary_mean_std.csv` | Mean/std summary across seeds |
| `lambda_validation_results.csv` | Inner-validation lambda sweep |
| `metadata.json` | Configuration, seed list, and leakage-control note |

## Dataset Defaults

| Dataset | Folder | Default data file | Default sample setting |
| --- | --- | --- | --- |
| German Numeric | `german_numeric` | `data/german-data-numeric` | Full dataset |
| South German | `south_german` | `data/SouthGermanCredit.asc` | Full dataset |
| Australian | `australian` | `data/australian.dat` | Full dataset |
| Taiwan | `taiwan` | `data/default of credit card clients.xls` | Informative 10,000-sample subset |
| Give Me Some Credit | `give_me_some_credit` | `data/cs-training.csv` | Informative 10,000-sample subset |
| FICO HELOC | `fico_heloc` | `data/heloc_dataset_v1.csv` | Full dataset |

See [`../data/README.md`](../data/README.md) for data-source and redistribution notes.

## Classical Reference Baselines

Run from the repository root:

```bash
python code/classical_baselines.py --data-root data --seeds 10 20 30 40 50
```

The default classical comparison uses PCA-7 inputs to match the main QSVC preprocessing.

## Regenerate Result Tables

```bash
python code/build_result_tables.py
```

Then inspect:

```text
code/results/
```

# FICO HELOC Experiment

This folder contains the QSVC experiment configuration for the FICO HELOC benchmark.

## Data

Expected local file:

```text
../../data/heloc_dataset_v1.csv
```

The target is `RiskPerformance`, with `Bad` treated as the positive class. Special values `-9`, `-8`, and `-7` are treated as missing values by the loader.

## Default Protocol

| Setting | Value |
| --- | --- |
| Sample scope | Full dataset |
| Seeds | `10 20 30 40 50` |
| PCA dimension | `7` |
| Lambda grid | `0.00` to `1.00` by `0.05` |
| Main cost | `5 * FN + FP` |
| Risk model | Logistic Regression |
| QSVC shape | Fixed `ZZFeatureMap`, 1 rep, linear entanglement |

## Run

From this folder:

```bash
python run_experiment.py --data-path ../../data/heloc_dataset_v1.csv
```

From the repository root:

```bash
python code/fico_heloc/run_experiment.py --data-path data/heloc_dataset_v1.csv
```

Outputs are saved in `results/`.

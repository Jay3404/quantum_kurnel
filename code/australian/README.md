# Australian Credit Approval Experiment

This folder contains the QSVC experiment configuration for the Australian Credit Approval benchmark.

## Data

Expected local file:

```text
../../data/australian.dat
```

The final label column is encoded so that the negative/bad class is treated as the positive class for bad-risk metrics.

## Default Protocol

| Setting | Value |
| --- | --- |
| Sample scope | Full dataset |
| Seeds | `10 20 30 40 50` |
| PCA dimension | `7` |
| Lambda grid | `0.00` to `1.00` by `0.05` |
| Main cost | `5 * FN + FP` |
| Convex risk model | XGBoost |
| Feature-specific risk model | Logistic Regression |

## Run

From this folder:

```bash
python run_experiment.py --data-path ../../data/australian.dat
```

From the repository root:

```bash
python code/australian/run_experiment.py --data-path data/australian.dat
```

Outputs are saved in `results/`.

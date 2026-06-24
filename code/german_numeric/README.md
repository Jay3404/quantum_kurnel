# German Numeric Experiment

This folder contains the QSVC experiment configuration for the numeric Statlog German Credit benchmark.

## Data

Expected local file:

```text
../../data/german-data-numeric
```

The final numeric class is encoded so that class `2` is treated as the positive bad-credit class.

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
python run_experiment.py --data-path ../../data/german-data-numeric
```

From the repository root:

```bash
python code/german_numeric/run_experiment.py --data-path data/german-data-numeric
```

Outputs are saved in `results/`.

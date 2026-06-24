# South German Credit Experiment

This folder contains the QSVC experiment configuration for the South German Credit benchmark.

## Data

Expected local file:

```text
../../data/SouthGermanCredit.asc
```

The target is `kredit`, with `0` treated as the positive bad-credit class.

## Default Protocol

| Setting | Value |
| --- | --- |
| Sample scope | Full dataset |
| Seeds | `10 20 30 40 50` |
| PCA dimension | `7` |
| Lambda grid | `0.00` to `1.00` by `0.05` |
| Main cost | `5 * FN + FP` |
| Risk model | Logistic Regression |

## Run

From this folder:

```bash
python run_experiment.py --data-path ../../data/SouthGermanCredit.asc
```

From the repository root:

```bash
python code/south_german/run_experiment.py --data-path data/SouthGermanCredit.asc
```

Outputs are saved in `results/`.

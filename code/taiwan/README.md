# Taiwan Default Experiment

This folder contains the QSVC experiment configuration for the UCI Taiwan default of credit card clients benchmark.

## Data

Expected local file:

```text
../../data/default of credit card clients.xls
```

The target is `default payment next month`, with `1` treated as the positive default/bad-risk class.

## Default Protocol

| Setting | Value |
| --- | --- |
| Sample scope | Informative 10,000-sample subset |
| Subset strategy | Stratified selection by PCA leverage score |
| Seeds | `10 20 30 40 50` |
| PCA dimension | `7` |
| Lambda grid | `0.00` to `1.00` by `0.05` |
| Main cost | `5 * FN + FP` |
| Risk model | Logistic Regression |

## Run

From this folder:

```bash
python run_experiment.py --data-path "../../data/default of credit card clients.xls"
```

From the repository root:

```bash
python code/taiwan/run_experiment.py --data-path "data/default of credit card clients.xls"
```

Outputs are saved in `results/`.

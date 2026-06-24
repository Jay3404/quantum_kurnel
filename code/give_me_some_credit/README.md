# Give Me Some Credit Experiment

This folder contains the QSVC experiment configuration for the Give Me Some Credit benchmark.

## Data

Expected local file:

```text
../../data/cs-training.csv
```

The target is `SeriousDlqin2yrs`, with `1` treated as the positive default/bad-risk class.

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
| QSVC shape | Fixed `ZZFeatureMap`, 1 rep, linear entanglement |

## Run

From this folder:

```bash
python run_experiment.py --data-path ../../data/cs-training.csv
```

From the repository root:

```bash
python code/give_me_some_credit/run_experiment.py --data-path data/cs-training.csv
```

Outputs are saved in `results/`.

# Data Inventory

This directory is the expected local data root for the final experiments. The public GitHub repository should keep raw third-party datasets out of version control unless redistribution is explicitly permitted by the original source.

If a dataset cannot be redistributed, keep it out of Git and ask users to download it from the official source, then place it in this directory with the filename shown below.

## Required Files

| Dataset | Required file | Used by | Default scope | Target definition in code |
| --- | --- | --- | --- | --- |
| Statlog German Credit numeric | `german-data-numeric` | `code/german_numeric` | Full dataset | Bad credit is encoded from numeric class `2` |
| South German Credit | `SouthGermanCredit.asc` | `code/south_german` | Full dataset | Bad credit is `kredit == 0` |
| Australian Credit Approval | `australian.dat` | `code/australian` | Full dataset | Bad/negative class is encoded from final label `0` |
| Taiwan default of credit card clients | `default of credit card clients.xls` | `code/taiwan` | Informative 10,000-sample subset | Default is `default payment next month == 1` |
| Give Me Some Credit | `cs-training.csv` | `code/give_me_some_credit` | Informative 10,000-sample subset | Default is `SeriousDlqin2yrs == 1` |
| FICO HELOC | `heloc_dataset_v1.csv` | `code/fico_heloc` | Full dataset | Bad risk is `RiskPerformance == Bad` |

## Source Notes

| Dataset | Typical source |
| --- | --- |
| Statlog German Credit numeric | UCI Machine Learning Repository, Statlog German Credit Data |
| South German Credit | UCI Machine Learning Repository, South German Credit update |
| Australian Credit Approval | UCI Machine Learning Repository, Statlog Australian Credit Approval |
| Taiwan default of credit card clients | UCI Machine Learning Repository |
| Give Me Some Credit | Kaggle competition data |
| FICO HELOC | FICO Explainable Machine Learning Challenge / HELOC dataset |

## Public-Release Guidance

Do not assume that a public benchmark can automatically be redistributed in this repository. For journal-review reproducibility, the safest public GitHub pattern is:

1. Keep raw third-party files out of Git unless redistribution is explicitly allowed.
2. Document the expected filename and target column.
3. Provide exact commands that accept `--data-path`.
4. Keep derived result tables under `code/results/` when they are generated from reproducible scripts.

The runners fail with a clear `FileNotFoundError` if a required dataset is missing.

## Local File Check

From the repository root:

```bash
find data -maxdepth 1 -type f -print
```

Run a single dataset with an explicit file path:

```bash
python code/australian/run_experiment.py --data-path data/australian.dat
```

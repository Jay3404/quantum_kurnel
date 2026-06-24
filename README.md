# Risk-Aware Quantum Kernel Experiments for Credit Risk

This repository contains the reproducibility package for a Soft Computing submission on cost-sensitive credit risk classification with risk-aware quantum kernel embeddings.

The core idea is deliberately narrow: a classical risk model estimates sample-level bad/default probability, and that risk signal is injected into the input representation used by a Quantum Support Vector Classifier (QSVC). The goal is not to claim universal quantum superiority. The goal is to test whether a risk-aware quantum feature map can reduce false-negative-heavy business cost on public credit benchmarks.

## Quick Path

Start here for the paper code, result tables, and reproduction commands.

| Path | Purpose |
| --- | --- |
| [`code/`](code/) | Reproducible experiment code for the six benchmark datasets |
| [`code/README.md`](code/README.md) | Main execution protocol and command examples |
| [`code/RESULTS.md`](code/RESULTS.md) | Curated result tables and paper-facing interpretation |
| [`data/README.md`](data/README.md) | Dataset inventory, required filenames, targets, and redistribution notes |

## Method Summary

For each dataset and seed, the experiment protocol is:

1. Split the data into train and held-out test partitions.
2. Preprocess tabular features and compress them to seven PCA dimensions.
3. Fit a classical risk estimator on the training data.
4. Transform the PCA representation with one of two risk-aware embeddings.
5. Rescale inputs to the quantum feature-map range.
6. Fit QSVC models and evaluate only once on the held-out test set.

The main evaluated models are:

| Model | Description |
| --- | --- |
| Baseline QSVC | QSVC on the shared PCA representation without risk injection |
| Convex risk-aware QSVC | Interpolates between the baseline representation and a risk-amplified representation |
| Feature-specific risk-aware QSVC | Uses feature importance from the classical risk model to weight the risk injection by feature |
| Classical baselines | Logistic Regression, RBF-SVM, Random Forest, XGBoost, and classical SVC on PCA features |

Lambda selection is performed on an inner validation split within the training set. The held-out test set is not used for lambda selection.

## Main Experimental Scope

The code covers six public credit-risk datasets:

| Dataset folder | Default setting |
| --- | --- |
| `code/german_numeric` | Statlog German numeric, full dataset |
| `code/south_german` | South German Credit, full dataset |
| `code/australian` | Australian Credit Approval, full dataset |
| `code/taiwan` | Taiwan default of credit card clients, informative 10,000-sample subset |
| `code/give_me_some_credit` | Give Me Some Credit, informative 10,000-sample subset |
| `code/fico_heloc` | FICO HELOC, full dataset |

The larger datasets are evaluated on controlled subsets where exact quantum-kernel simulation would otherwise be computationally expensive.

## Key Result Reading

Use the results as dataset-dependent evidence, not as a universal performance claim.

| Dataset | Strongest paper-facing takeaway |
| --- | --- |
| German Numeric | Risk-aware QSVC reduces false-negative-heavy business cost relative to baseline QSVC |
| South German | Risk-aware QSVC gives one of the clearest cost-sensitive improvements |
| Australian | Convex risk-aware QSVC improves cost, while feature-specific weighting is not consistently helpful |
| Taiwan | Risk-aware QSVC is roughly competitive, with only small cost movement |
| Give Me Some Credit | Risk-aware QSVC is not a clear winner against stronger classical baselines |
| FICO HELOC | Baseline/risk-aware QSVC are close; Random Forest and XGBoost are strong classical references |

The main paper message should therefore be: risk-aware quantum embeddings can be useful under some credit-risk data geometries, especially when the induced representation improves bad-class recall enough to offset false-positive cost.

## Quick Reproduction

Install dependencies for the code:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r code/requirements.txt
```

Run all QSVC experiments from the repository root:

```bash
python code/run_all.py --data-root data --seeds 10 20 30 40 50
```

Run one dataset:

```bash
cd code/australian
python run_experiment.py --data-path ../../data/australian.dat
```

Outputs are written to each dataset's `results/` directory:

| File | Meaning |
| --- | --- |
| `raw_results.csv` | Per-seed test metrics |
| `summary_mean_std.csv` | Mean/std summary across seeds |
| `lambda_validation_results.csv` | Inner-validation lambda sweep results |
| `metadata.json` | Configuration and leakage-control metadata |

## Repository Layout

```text
.
├── data/                      # Dataset files or local placeholders; see data/README.md
├── code/                      # Reproducibility package
│   ├── _shared/               # Shared pipeline implementation
│   ├── australian/            # Dataset-specific runner and config
│   ├── fico_heloc/
│   ├── german_numeric/
│   ├── give_me_some_credit/
│   ├── south_german/
│   ├── taiwan/
│   ├── results/
│   ├── supplementary_results/
│   └── run_all.py
├── pyproject.toml             # Poetry environment metadata
└── poetry.lock
```

## Public-Release Notes

Before making the repository public, confirm:

| Item | Status |
| --- | --- |
| Dataset redistribution | Check each source license; do not publish third-party data if redistribution is not allowed |
| License | Add a project license selected by the authors |
| Citation | Add final DOI/arXiv/manuscript citation when available |
| Results | Treat `code/results/` as the paper-facing result bundle |

## Suggested Citation

The final citation should be added after journal submission metadata is fixed.

```bibtex
@misc{risk_aware_quantum_kernel_credit,
  title = {A Hybrid Risk-Aware Quantum-Classical Framework for Cost-Sensitive Credit Risk Classification},
  author = {Author information to be added},
  year = {2026},
  note = {Code repository for Soft Computing submission}
}
```

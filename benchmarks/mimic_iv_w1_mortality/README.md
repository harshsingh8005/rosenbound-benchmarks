# MIMIC-IV W1 in-hospital mortality reproducer

## What this reproduces

The **W1** benchmark predicts **in-hospital mortality** — death during the
hospital stay — for intensive-care admissions in **MIMIC-IV** (Johnson et al.,
2023). The cohort is the first ICU stay of each hospital admission; the label
is the admission's `hospital_expire_flag`. Every predictor is measured within
the first 24 hours of the ICU stay, so nothing from after the prediction point
can leak into the model.

This reproducer estimates the headline discrimination and calibration numbers
using only published, non-proprietary methods, so they can be checked without
any access to the closed platform. It runs out of the box on the open-access
100-patient demo and, unchanged, on the full credentialed corpus.

## Methods

- **Classifier** (`src/model.py`) — LightGBM gradient-boosted trees (Ke et al.,
  2017), parameterised for a small, class-imbalanced cohort.
- **Calibration** (`src/calibration.py`) — Beta calibration (Kull, Silva Filho
  & Flach, 2017) fit by maximum a posteriori under a Gaussian prior (L2), so the
  recalibration is stable on small validation folds.
- **Evaluation** (`src/evaluate.py`) — AUROC and average precision for
  discrimination; Brier score and 10-bin expected calibration error (ECE) for
  calibration.

Because the demo cohort is small (~130 admissions, ~15 deaths), a single
train/test split is dominated by which few deaths land in the test fold. The
run instead pools out-of-fold predictions from a stratified 5-fold
cross-validation and averages the metrics over 40 seed repeats, so each
admission is scored once by a model that never saw it and the reported numbers
are stable and reproducible.

## How to run (demo)

```bash
pip install -r requirements.txt
python data/mimic_iv/fetch.py            # open-access 100-patient demo (~15 MB)
cd benchmarks/mimic_iv_w1_mortality
python run.py                            # canonical demo run (seed 42)
```

`python run.py --ci` runs the same fixed-seed canonical configuration used by
continuous integration. The run writes `results.json` and asserts the
calibrated metrics against `expected_results.json`, exiting non-zero on drift.

## How to run (credentialed full corpus)

The full MIMIC-IV corpus is credentialed-access data and is never committed
here. With a completed PhysioNet CITI certificate and a signed data use
agreement (see [`../../data/mimic_iv/README.md`](../../data/mimic_iv/README.md)),
point the reproducer at the tables and enable the gate:

```bash
export MIMIC_IV_CREDENTIALED=1
export MIMIC_IV_PATH=/path/to/mimic-iv        # directory holding hosp/ and icu/
python run.py --credentialed
```

## Or use the notebook

```bash
jupyter notebook notebook.ipynb
```

The notebook loads the cohort, engineers the features, fits the model, shows
the reliability curve before and after calibration, and compares to the
expected numbers, cell by cell.

## Expected numbers

The canonical demo run reproduces the contract in `expected_results.json`
(calibrated metrics, seed 42):

| Metric            | Demo target | Tolerance |
|-------------------|------------:|----------:|
| AUROC             | 0.6197      | ±0.05     |
| Average precision | 0.2069      | ±0.05     |
| Brier score       | 0.1056      | ±0.02     |
| ECE (10-bin)      | 0.0472      | ±0.02     |

## Interpretation

The demo cohort is two orders of magnitude smaller than the full corpus, so its
discrimination is both lower and noisier than the full-corpus figure — an
honest reflection of what 130 admissions support. The value of the demo run is
that it is a working, checkable pipeline: the Beta + Bayes calibrator measurably
tightens calibration (it roughly halves the uncalibrated ECE), and the whole
thing is deterministic and public. The full-corpus AUROC recorded in
`expected_results.json` → `internal_reference` is reached on the 546K-admission
corpus, not on this demo.

## Internal vs public

> The full internal system augments this pipeline with additional
> risk-stratification and calibration components; the numbers in
> `expected_results.json` → `internal_reference` are from that system on the
> full 546K-admission corpus. Those extensions are commercial; contact the
> maintainers for a platform demonstration.

## Data provenance

Acquisition and licence details are in
[`../../data/mimic_iv/README.md`](../../data/mimic_iv/README.md); the cohort
definition and determinism controls are in
[`../../docs/data_provenance.md`](../../docs/data_provenance.md).

## Citations

- Johnson, A. E. W., et al. (2023). MIMIC-IV, a freely accessible electronic
  health record dataset. *Scientific Data*, 10(1). PhysioNet.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision
  tree. *Advances in Neural Information Processing Systems*, 30.
- Kull, M., Silva Filho, T., & Flach, P. (2017). Beta calibration: a
  well-founded and easily implemented improvement on logistic calibration for
  binary classifiers. *AISTATS / JMLR*.
- Purushotham, S., Meng, C., Che, Z., & Liu, Y. (2018). Benchmarking deep
  learning models on large healthcare datasets. *Journal of Biomedical
  Informatics*, 83.

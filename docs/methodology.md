# Methodology

This document describes the public methods used by each reproducer. None of
the methods here are proprietary; the patent-protected components of the
Rosenbound platform are out of scope for this repository and are enumerated in
[`patent_scope.md`](patent_scope.md).

## ACIC22 Track-2

The Track-2 challenge scores average-treatment-effect estimation across many
semi-synthetic cohorts with known ground truth. The reproducer estimates the
effect on each cohort with doubly-robust methods and reports aggregate accuracy
and interval calibration.

**Estimators.**

- **AIPW** (augmented inverse-probability weighting) — combines an outcome
  regression with a propensity model so the estimate stays consistent when
  either nuisance model is correctly specified.
- **DR-ATT** — a doubly-robust average-treatment-effect-on-the-treated
  estimator over the region of covariate overlap.
- **Rosenbaum sensitivity** — quantifies how strong an unobserved confounder
  would need to be (the Γ bound) to overturn the estimated effect.

**Metrics.** For each cohort the estimator produces a point estimate and an
uncertainty interval. Aggregated across all cohorts:

- **Bias** — mean signed error of the point estimate against the known effect.
- **RMSE** — root-mean-square error of the point estimate.
- **Coverage** — fraction of cohorts whose 95% interval contains the true
  effect (a well-calibrated interval is near 0.95).
- **Average width** — mean width of the uncertainty intervals.

Each cohort's panel is reduced to a cross-sectional problem with a
practice-level difference-in-differences change score before estimation, and
the metrics are aggregated over a fixed, seeded cohort sample. The expected
values and their tolerances are pinned in
`benchmarks/acic22_track2/expected_results.json`, alongside the internal
reference figures the closed platform reaches.

## MIMIC-IV W1 in-hospital mortality

W1 predicts whether a patient dies during a hospital admission, from features
available at and shortly after admission.

**Model.** A classical **LightGBM** gradient-boosted classifier on the
admission-level feature matrix. No transformer or large-language-model
component is used anywhere in the prediction path.

**Calibration.** Raw model scores are recalibrated with a **Beta + Bayes**
calibrator so that predicted probabilities align with observed event rates.

**Metrics.**

- **AUROC** — area under the receiver-operating-characteristic curve on the
  held-out test split, measuring discrimination.
- **ECE** — expected calibration error, the average gap between predicted
  probability and observed frequency across probability bins, measuring
  calibration quality.

AUROC and ECE are reported to four decimal places. The expected values are
pinned in `benchmarks/mimic_iv_w1_mortality/expected_results.json`.

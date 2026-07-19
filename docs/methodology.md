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

**Features.** Three blocks: admission descriptors (gender, admission
type/location, insurance) one-hot encoded against the training vocabulary;
numeric demographics (`anchor_age`) and a comorbidity index; and first-24h
vital and lab aggregates, median-imputed. The comorbidity index,
`charlson_history`, is a Charlson index (Quan 2005 ICD-9/ICD-10 code sets,
Charlson 1987 weights) built **only from the subject's prior admissions
discharged before the current admit time**. This matters because
`hosp/diagnoses_icd` carries no timestamp and its rows are discharge-coded
billing diagnoses of the admission — counting the current admission's diagnoses
would leak information unavailable at the 24-hour prediction point. The encoder
vocabulary and imputation medians are fit on training folds only.

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

## Reproducibility tiers

Each reproducer distinguishes what an outside party can verify from what remains
proprietary. Four levels apply across the repository:

- **Demo.** Runs on open MIMIC-IV demo data or a sampled FAERS slice. Fully
  verifiable in continuous integration — a `git clone` plus `pip install`
  reproduces the pinned numbers. This is the only tier the committed targets and
  tolerances gate.
- **Credentialed public-method.** The same public-method pipeline (LightGBM +
  Beta calibration, no proprietary components) run on a full credentialed
  corpus. Expected values are recorded as future work — unpinned until validated
  across independent runs — so results at this level are informational rather
  than gated. See the `credentialed_public_method` key in each
  `expected_results.json`.
- **Externally verified.** Third-party-certified metrics
  for the full-corpus pipeline obtained without exposing proprietary source, via
  a MedPerf (MLCommons) container submission that runs on data-owner
  infrastructure and an ACIC Data Challenge predictions-only entry. Both routes
  are in setup; results will be linked when live.
- **Internal (proprietary-augmented).** The proprietary Rosenbound pipeline,
  not reproducible from this repository. Full-corpus figures are recorded under
  `internal_reference` with method-category rationale in
  [`patent_scope.md`](patent_scope.md); external verification for them goes
  through the externally-verified tier above.

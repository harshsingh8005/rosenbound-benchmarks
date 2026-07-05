# Data provenance

This document records the exact dataset versions, cohort filters, and
determinism controls each reproducer relies on. Acquisition instructions live
under `data/{source}/README.md`; this document describes what the reproducers
do with the data once it is on disk.

## Dataset versions

| Source          | Version / scope                         | Access               |
|-----------------|-----------------------------------------|----------------------|
| MIMIC-IV        | v3.1 — 546K admissions / 364K patients  | PhysioNet credentialed |
| ACIC22 Track-2  | 3,400-cohort canonical                  | public               |
| FAERS           | 2.83M-row backfill                      | public (future work) |

The ACIC22 Track-2 and MIMIC-IV W1 reproducers ship in this repository. FAERS
is listed for completeness; it is the basis of a separate future reproducer and
is not exercised by any code here.

The MIMIC-IV W1 reproducer runs by default on the open-access **MIMIC-IV
Clinical Database Demo v2.2** (100 patients, Open Data Commons ODbL), which
carries the same `hosp/` and `icu/` table layout as the full credentialed v3.1
corpus. The demo requires no credentials; the full corpus does.

## Cohort filters

**ACIC22 Track-2.** The reproducer scores a fixed, seeded 100-cohort sample of
the 3,400-cohort release by default; `--full` runs every cohort with no
exclusions. Each cohort's panel is reduced to a cross-sectional problem by a
practice-level difference-in-differences change score (post-period mean outcome
minus pre-period mean), and the effect on the treated is estimated on that
change score. The scored estimand is the sample average treatment effect on the
treated (SATT), taken from the challenge's estimand-truth table.

**MIMIC-IV W1.** The in-hospital mortality cohort is one row per **first ICU
stay of a hospital admission**: ICU stays are ordered by intake time and the
earliest is kept for each admission. The label is in-hospital expiry recorded on
the admission (`hospital_expire_flag`). Every predictor — demographics, the
diagnosis count, and the first-24h vital and lab aggregates — is measured within
the first 24 hours of the ICU stay, so no post-prediction information leaks into
the features. The one-hot vocabulary and imputation medians are fit on training
folds only. By default the reproducer runs on the open-access **MIMIC-IV
Clinical Database Demo v2.2** (100 patients); the identical code path runs on
the full credentialed v3.1 corpus when pointed at it. The inclusion predicates
and feature windows are applied deterministically so the cohort is reconstructed
identically on every run.

## Determinism

Every reproducer pins the random seeds that govern train/test splitting, model
initialization, and any bootstrap resampling. The seed values are declared
alongside the reproducer entry point and are held fixed so that a given dataset
version yields the exact metrics recorded in the corresponding
`expected_results.json`. A metric that drifts from the pinned contract fails the
run rather than being silently accepted.

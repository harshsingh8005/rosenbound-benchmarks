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

## Cohort filters

**ACIC22 Track-2.** The reproducer scores a fixed, seeded 100-cohort sample of
the 3,400-cohort release by default; `--full` runs every cohort with no
exclusions. Each cohort's panel is reduced to a cross-sectional problem by a
practice-level difference-in-differences change score (post-period mean outcome
minus pre-period mean), and the effect on the treated is estimated on that
change score. The scored estimand is the sample average treatment effect on the
treated (SATT), taken from the challenge's estimand-truth table.

**MIMIC-IV W1.** The in-hospital mortality cohort is derived from the 546K
MIMIC-IV v3.1 admissions by admission-level inclusion filters (adult admissions
with the features required by the model available at admission). The label is
in-hospital expiry recorded on the admission. The exact inclusion predicates
and feature windows are defined in the reproducer and are applied
deterministically so the cohort is reconstructed identically on every run.

## Determinism

Every reproducer pins the random seeds that govern train/test splitting, model
initialization, and any bootstrap resampling. The seed values are declared
alongside the reproducer entry point and are held fixed so that a given dataset
version yields the exact metrics recorded in the corresponding
`expected_results.json`. A metric that drifts from the pinned contract fails the
run rather than being silently accepted.

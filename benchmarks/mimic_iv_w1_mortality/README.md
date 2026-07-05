# MIMIC-IV W1 in-hospital mortality reproducer

## What this benchmark is

The **W1** benchmark predicts **in-hospital mortality** (expiry during the
hospital admission) from admission-level features on **MIMIC-IV v3.1**. The
classifier is a classical gradient-boosted model (LightGBM) whose predicted
probabilities are recalibrated with a Beta + Bayes calibrator, and it is scored
on a held-out test split by discrimination (AUROC) and calibration (expected
calibration error, ECE).

This reproducer re-derives the reported number on the same data using
published, non-proprietary methods, so it can be confirmed independently of the
closed Rosenbound platform.

## Running it

```bash
cd benchmarks/mimic_iv_w1_mortality
python run.py
```

This benchmark requires MIMIC-IV v3.1, which is credentialed-access data. You
must hold a completed PhysioNet CITI certificate and a signed MIMIC-IV data use
agreement, and the tables must be present under `../../data/mimic_iv/`. See
`data/mimic_iv/README.md` for the acquisition steps.

Pass `--ci` for the shortened run used by continuous integration. The CI job is
skipped entirely on forks that lack the `PHYSIONET_USER` / `PHYSIONET_PASS`
secrets.

## Expected wall-time

Feature construction dominates on first run; the model fit itself is minutes.
Budget time proportional to how much of `labevents` / `chartevents` your disk
and memory can stream.

## Expected outputs

The run reproduces the lock recorded in `expected_results.json`:

| Metric      | Expected   |
|-------------|------------|
| Test AUROC  | 0.9476     |
| Calibrator  | beta_bayes |
| ECE         | 0.0024     |

The run exits 0 when every metric matches the contract to the recorded
precision, and non-zero on drift.

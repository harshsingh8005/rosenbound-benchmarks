# ACIC22 Track-2 reproducer

## What this benchmark is

The **2022 Atlantic Causal Inference Conference (ACIC22) data challenge**
evaluates average-treatment-effect estimation on semi-synthetic datasets where
the ground-truth effect is known. **Track-2** provides many independent cohorts
and scores each estimator on how accurately and how well-calibrated its effect
estimates and uncertainty intervals are.

The Rosenbound platform reports a locked result on the full **3,400-cohort
canonical**. This reproducer re-derives that result on the same public data
using published, non-proprietary estimators (AIPW / DR-ATT with Rosenbaum
sensitivity), so the number can be confirmed independently of the closed
platform.

## Running it

```bash
cd benchmarks/acic22_track2
python run.py
```

The reproducer reads the extracted cohorts from `../../data/acic22/` (see
`data/acic22/README.md` for how to obtain them), aggregates the metrics across
all cohorts, writes a `results.json`, and asserts it against
`expected_results.json`.

Pass `--ci` for the shortened run used by continuous integration, which
completes inside the GitHub Actions job cap.

## Expected wall-time

Under **10 minutes** on a modern laptop for the full 3,400-cohort canonical.

## Expected outputs

The run reproduces the V3 lock recorded in `expected_results.json`:

| Metric               | Expected |
|----------------------|----------|
| Bias                 | +19.26   |
| RMSE                 | 28.80    |
| Coverage (%)         | 77.53    |
| Average interval width | 78.04  |
| Cohorts              | 3,400    |

The run exits 0 when every metric matches the contract to the recorded
precision, and non-zero on drift.

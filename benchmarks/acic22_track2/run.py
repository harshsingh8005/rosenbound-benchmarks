"""Reproducer entry-point for the ACIC22 Track-2 V3 lock.

When implemented, run.py:
  * loads the 3,400-cohort canonical from ../../data/acic22/
  * runs the AIPW / DR-ATT / Rosenbaum estimators on each cohort
  * aggregates bias, RMSE, coverage, and interval width
  * writes results.json next to expected_results.json
  * asserts every metric matches expected_results.json to the
    precision recorded there (bias/rmse/width to 2 dp,
    coverage_percent to 2 dp)
  * exits 0 on match, non-zero on drift

The --ci flag (used by GitHub Actions) enables a short-canonical
subset that completes inside the CI job cap.
"""

import sys


def main() -> int:
    raise NotImplementedError(
        "This reproducer is not yet implemented. See "
        "benchmarks/acic22_track2/README.md for context."
    )


if __name__ == "__main__":
    sys.exit(main())

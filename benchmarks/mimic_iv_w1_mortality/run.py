"""Reproducer entry-point for the MIMIC-IV W1 in-hospital mortality lock.

When implemented, run.py:
  * loads the MIMIC-IV v3.1 tables from ../../data/mimic_iv/
  * builds the admission-level feature matrix and the in-hospital
    mortality label
  * fits the LightGBM classifier and the Beta + Bayes calibrator
    on the training split
  * evaluates test AUROC and expected calibration error (ECE)
  * writes results.json next to expected_results.json
  * asserts every metric matches expected_results.json to the
    precision recorded there (test_auroc to 4 dp, expected_ece to 4 dp)
  * exits 0 on match, non-zero on drift

The --ci flag (used by GitHub Actions) enables a short subset that
completes inside the CI job cap.
"""

import sys


def main() -> int:
    raise NotImplementedError(
        "This reproducer is not yet implemented. See "
        "benchmarks/mimic_iv_w1_mortality/README.md for context."
    )


if __name__ == "__main__":
    sys.exit(main())

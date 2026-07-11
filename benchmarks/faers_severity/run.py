"""Reproducer entry point for the FAERS adverse-event severity benchmark.

Loads the report-level severity cohort, splits it **temporally** on receive
year, fits the leak-safe feature vocabulary on the training window only, scores
seriousness with a LightGBM classifier, and reports discrimination (AUROC,
AUPRC) and calibration (Brier, ECE) on the held-out later window.

The temporal split is the point of the design: feature selection and model
fitting see only reports from before the split year, so no later-era statistic
can inform the prediction — the same leak-safety principle behind the W1
Charlson fix, enforced here on the time axis.

Usage
-----
    python run.py                 # canonical demo run (seed 42, split year 2013)
    python run.py --ci            # same canonical run, for CI
    python run.py --split-year 2013 --seed 7

Determinism: a given (--seed, --split-year) reproduces byte-identically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src import (
    evaluate_predictions,
    fit_predict_proba,
    load_faers_cohort,
)
from src.features import build_features_apply, build_features_fit
from src.loader import LABEL_COL, YEAR_COL

_HERE = Path(__file__).resolve().parent
_EXPECTED = _HERE / "expected_results.json"

_CANONICAL_SEED = 42
_CANONICAL_SPLIT_YEAR = 2013
_METRIC_KEYS = ("auroc", "auprc", "brier", "ece")


def run_reproducer(
    data_path: str | None,
    seed: int = _CANONICAL_SEED,
    split_year: int = _CANONICAL_SPLIT_YEAR,
    cohort=None,
) -> dict:
    """Temporal-split, fit, and score the FAERS severity cohort.

    ``cohort`` may be supplied directly (for tests on synthetic reports);
    otherwise the demo slice at ``data_path`` is loaded.
    """
    if cohort is None:
        cohort = load_faers_cohort(data_path)
    frame = cohort.frame

    train_mask = frame[YEAR_COL] < split_year
    test_mask = ~train_mask
    train_frame = frame[train_mask].reset_index(drop=True)
    test_frame = frame[test_mask].reset_index(drop=True)
    if train_frame.empty or test_frame.empty:
        raise ValueError(
            f"temporal split on year {split_year} left an empty fold "
            f"(train={len(train_frame)}, test={len(test_frame)})"
        )

    y_train = train_frame[LABEL_COL].to_numpy(dtype=int)
    y_test = test_frame[LABEL_COL].to_numpy(dtype=int)

    X_train, state = build_features_fit(train_frame)
    X_test = build_features_apply(test_frame, state)

    proba = fit_predict_proba(X_train, y_train, X_test, random_state=seed)
    metrics = evaluate_predictions(y_test, proba)

    return {
        "dataset": "faers-openfda-demo-slice",
        "n_reports": int(cohort.n),
        "n_train": int(len(train_frame)),
        "n_test": int(len(test_frame)),
        "split_year": split_year,
        "prevalence_train": round(float(y_train.mean()), 6),
        "prevalence_test": round(float(y_test.mean()), 6),
        "n_features": int(X_train.shape[1]),
        "seed": seed,
        "metrics": metrics,
    }


def assert_within_tolerance(results: dict, expected: dict) -> list[str]:
    """Return breach messages comparing metrics to the demo targets."""
    breaches: list[str] = []
    targets = expected.get("reproducer_targets", {}).get("demo", {})
    observed = results.get("metrics", {})
    for metric, spec in targets.items():
        if metric not in observed:
            continue
        obs, tgt, tol = observed[metric], spec["target"], spec["tol"]
        if abs(obs - tgt) > tol:
            breaches.append(
                f"metrics.{metric}: observed {obs:.4f} vs target "
                f"{tgt:.4f} +/- {tol} (delta {obs - tgt:+.4f})"
            )
    return breaches


def _print_table(results: dict, expected: dict) -> None:
    targets = expected.get("reproducer_targets", {}).get("demo", {})
    m = results["metrics"]
    print(f"\nFAERS severity demo: {results['n_reports']} reports "
          f"(train {results['n_train']} / test {results['n_test']}), "
          f"split year {results['split_year']}, {results['n_features']} features")
    print(f"{'metric':<16}{'observed':>12}{'target':>12}{'tol':>8}{'ok':>6}")
    for metric in _METRIC_KEYS:
        obs = m[metric]
        spec = targets.get(metric)
        if spec is None:
            print(f"{metric:<16}{obs:>12.4f}{'-':>12}{'-':>8}{'-':>6}")
            continue
        ok = abs(obs - spec["target"]) <= spec["tol"]
        print(f"{metric:<16}{obs:>12.4f}{spec['target']:>12.4f}"
              f"{spec['tol']:>8}{'PASS' if ok else 'FAIL':>6}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=_CANONICAL_SEED)
    parser.add_argument("--split-year", type=int, default=_CANONICAL_SPLIT_YEAR)
    parser.add_argument("--data", type=str, default=None,
                        help="path to a FAERS demo slice (defaults to the committed one)")
    parser.add_argument("--output", type=str, default=str(_HERE / "results.json"))
    parser.add_argument("--ci", action="store_true",
                        help="run the fixed-seed canonical configuration used by CI")
    args = parser.parse_args(argv)

    seed = _CANONICAL_SEED if args.ci else args.seed
    split_year = _CANONICAL_SPLIT_YEAR if args.ci else args.split_year

    try:
        results = run_reproducer(args.data, seed=seed, split_year=split_year)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("Fetch the demo slice: python -m benchmarks.faers_severity.src.fetch",
              file=sys.stderr)
        return 1

    Path(args.output).write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"wrote {args.output}")

    if not _EXPECTED.is_file():
        print("no expected_results.json to assert against", file=sys.stderr)
        return 0
    expected = json.loads(_EXPECTED.read_text())
    _print_table(results, expected)

    is_canonical = seed == _CANONICAL_SEED and split_year == _CANONICAL_SPLIT_YEAR
    if not is_canonical:
        print("\n(informational only: tolerances apply to the canonical demo run)")
        return 0
    breaches = assert_within_tolerance(results, expected)
    if breaches:
        print("\nTOLERANCE BREACH:", file=sys.stderr)
        for b in breaches:
            print(f"  - {b}", file=sys.stderr)
        return 1
    print("\nall metrics within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())

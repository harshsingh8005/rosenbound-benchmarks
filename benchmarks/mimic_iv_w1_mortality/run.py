"""Reproducer entry point for the MIMIC-IV W1 in-hospital mortality benchmark.

The script loads the first-24h ICU-admission cohort, scores in-hospital
mortality with a LightGBM classifier whose probabilities are recalibrated by a
standalone Beta + Bayes calibrator, and reports discrimination and calibration.

Why cross-validation rather than one split
------------------------------------------
On the 100-patient open-access demo the cohort is ~130 ICU admissions with ~15
deaths, so a single train/test split gives a metric dominated by which three or
four deaths land in the test fold. The reproducer instead pools out-of-fold
predictions from a stratified K-fold cross-validation, so every admission is
scored exactly once by a model that never saw it, and repeats the whole
cross-validation over many seeds and averages — turning a high-variance number
into a stable, reproducible one. Within each training fold a further stratified
split is held out to fit the calibrator, so calibration never sees fold-test
labels.

Usage
-----
    python run.py                      # canonical demo run (seed 42)
    python run.py --ci                 # same canonical run, for CI
    python run.py --seed 7 --n-seeds 5
    python run.py --credentialed       # full corpus (MIMIC_IV_CREDENTIALED=1)

Determinism: a given (--seed, --n-seeds, --folds) reproduces byte-identically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src import (
    BetaBayesCalibrator,
    evaluate_predictions,
    fit_predict_proba,
    load_w1_cohort,
    reliability_curve,
)
from src.features import build_features_apply, build_features_fit
from src.loader import LABEL_COL

_HERE = Path(__file__).resolve().parent
_EXPECTED = _HERE / "expected_results.json"

# The reproduction contract is defined on this fixed configuration. Because the
# whole pipeline is deterministic, --ci re-runs exactly it. 40 seed repeats of a
# 5-fold cross-validation is enough to settle the demo metrics to their third
# decimal while finishing well inside the smoke wall-time budget.
_CANONICAL_SEED = 42
_CANONICAL_N_SEEDS = 40
_FOLDS = 5
_CALIBRATION_FRACTION = 0.25
_METRIC_KEYS = ("auroc", "average_precision", "brier", "ece")


def _one_cv(
    frame,
    y: np.ndarray,
    seed: int,
    folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled out-of-fold predictions for one seeded cross-validation.

    Returns ``(uncalibrated, calibrated)`` probability arrays aligned to the
    rows of ``frame``: each admission's entry is the prediction from the one
    fold in which it was held out.
    """
    oof_raw = np.zeros(len(y), dtype=np.float64)
    oof_cal = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for tr_idx, te_idx in skf.split(frame, y):
        tr_frame = frame.iloc[tr_idx]
        y_tr = y[tr_idx]
        # Hold out a stratified slice of the training fold for the calibrator.
        fit_pos, cal_pos = train_test_split(
            np.arange(len(tr_idx)),
            test_size=_CALIBRATION_FRACTION,
            random_state=seed,
            stratify=y_tr,
        )
        X_fit, state = build_features_fit(tr_frame.iloc[fit_pos])
        X_cal = build_features_apply(tr_frame.iloc[cal_pos], state)
        X_te = build_features_apply(frame.iloc[te_idx], state)

        cal_scores = fit_predict_proba(X_fit, y_tr[fit_pos], X_cal, random_state=seed)
        te_scores = fit_predict_proba(X_fit, y_tr[fit_pos], X_te, random_state=seed)

        calibrator = BetaBayesCalibrator().fit(cal_scores, y_tr[cal_pos])
        oof_raw[te_idx] = te_scores
        oof_cal[te_idx] = calibrator.predict(te_scores)

    return oof_raw, oof_cal


def run_reproducer(
    data_root: str | os.PathLike[str] | None,
    seed: int,
    n_seeds: int,
    folds: int = _FOLDS,
    verbose: bool = False,
) -> dict:
    """Load the cohort and aggregate cross-validated metrics over seed repeats.

    Metrics are computed on each seed's pooled out-of-fold predictions and then
    averaged; the reliability curve is built from predictions pooled across all
    seeds. Returns the results contract written to disk.
    """
    cohort = load_w1_cohort(data_root)
    frame = cohort.frame
    y = frame[LABEL_COL].to_numpy(dtype=int)
    features = frame.drop(columns=[LABEL_COL])

    raw_metrics: list[dict[str, float]] = []
    cal_metrics: list[dict[str, float]] = []
    all_cal: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for s in range(seed, seed + n_seeds):
        oof_raw, oof_cal = _one_cv(features, y, seed=s, folds=folds)
        raw_metrics.append(evaluate_predictions(y, oof_raw))
        cal_metrics.append(evaluate_predictions(y, oof_cal))
        all_cal.append(oof_cal)
        all_y.append(y)
        if verbose:
            print(f"  seed {s}: auroc={cal_metrics[-1]['auroc']:.4f} "
                  f"ece={cal_metrics[-1]['ece']:.4f}")

    def _avg(rows: list[dict[str, float]]) -> dict[str, float]:
        return {k: float(np.mean([r[k] for r in rows])) for k in _METRIC_KEYS}

    reliab = reliability_curve(np.concatenate(all_y), np.concatenate(all_cal))
    return {
        "dataset": "mimic-iv-clinical-database-demo-2.2",
        "n_admissions": cohort.n,
        "prevalence": round(cohort.prevalence, 6),
        "seed": seed,
        "n_seeds": n_seeds,
        "folds": folds,
        "calibrator": "beta_bayes",
        "uncalibrated": _avg(raw_metrics),
        "calibrated": _avg(cal_metrics),
        "reliability_curve": reliab,
    }


def assert_within_tolerance(results: dict, expected: dict) -> list[str]:
    """Return breach messages comparing calibrated metrics to the demo targets."""
    breaches: list[str] = []
    targets = expected.get("reproducer_targets", {}).get("demo", {})
    observed = results.get("calibrated", {})
    for metric, spec in targets.items():
        if metric not in observed:
            continue
        obs, tgt, tol = observed[metric], spec["target"], spec["tol"]
        if abs(obs - tgt) > tol:
            breaches.append(
                f"calibrated.{metric}: observed {obs:.4f} vs target "
                f"{tgt:.4f} +/- {tol} (delta {obs - tgt:+.4f})"
            )
    return breaches


def _print_table(results: dict, expected: dict) -> None:
    targets = expected.get("reproducer_targets", {}).get("demo", {})
    cal = results["calibrated"]
    print(f"\nMIMIC-IV W1 demo: {results['n_admissions']} admissions, "
          f"prevalence {results['prevalence']:.3f}, "
          f"{results['n_seeds']}x{results['folds']}-fold CV (seed {results['seed']})")
    print(f"{'metric':<22}{'observed':>12}{'target':>12}{'tol':>8}{'ok':>6}")
    for metric in _METRIC_KEYS:
        obs = cal[metric]
        spec = targets.get(metric)
        if spec is None:
            print(f"{'calibrated.' + metric:<22}{obs:>12.4f}{'-':>12}{'-':>8}{'-':>6}")
            continue
        ok = abs(obs - spec["target"]) <= spec["tol"]
        print(f"{'calibrated.' + metric:<22}{obs:>12.4f}{spec['target']:>12.4f}"
              f"{spec['tol']:>8}{'PASS' if ok else 'FAIL':>6}")


def _resolve_data_root(credentialed: bool) -> str | None:
    """Return the data root, honouring the credentialed full-corpus gate."""
    if not credentialed:
        return None
    if os.environ.get("MIMIC_IV_CREDENTIALED") != "1":
        raise SystemExit(
            "--credentialed requires MIMIC_IV_CREDENTIALED=1 and the full "
            "MIMIC-IV tables. See data/mimic_iv/README.md."
        )
    root = os.environ.get("MIMIC_IV_PATH")
    if not root:
        raise SystemExit("--credentialed requires MIMIC_IV_PATH to point at the corpus.")
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=_CANONICAL_SEED)
    parser.add_argument("--n-seeds", type=int, default=_CANONICAL_N_SEEDS)
    parser.add_argument("--folds", type=int, default=_FOLDS)
    parser.add_argument("--output", type=str, default=str(_HERE / "results.json"))
    parser.add_argument("--ci", action="store_true",
                        help="run the fixed-seed canonical sample used by CI")
    parser.add_argument("--credentialed", action="store_true",
                        help="use the full MIMIC-IV corpus (MIMIC_IV_CREDENTIALED=1)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.ci:
        seed, n_seeds, folds = _CANONICAL_SEED, _CANONICAL_N_SEEDS, _FOLDS
    else:
        seed, n_seeds, folds = args.seed, args.n_seeds, args.folds

    data_root = _resolve_data_root(args.credentialed)
    try:
        results = run_reproducer(data_root, seed, n_seeds, folds, verbose=args.verbose)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("Fetch the demo first: python data/mimic_iv/fetch.py", file=sys.stderr)
        return 1

    Path(args.output).write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"wrote {args.output}")

    if not _EXPECTED.is_file():
        print("no expected_results.json to assert against", file=sys.stderr)
        return 0
    expected = json.loads(_EXPECTED.read_text())
    _print_table(results, expected)

    # Tolerances describe the canonical demo configuration; only that run is
    # gated. Other configurations print the table for reference and pass.
    is_canonical = (
        not args.credentialed
        and seed == _CANONICAL_SEED
        and n_seeds == _CANONICAL_N_SEEDS
        and folds == _FOLDS
    )
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

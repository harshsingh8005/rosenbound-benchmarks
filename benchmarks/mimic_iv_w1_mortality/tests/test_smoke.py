"""End-to-end smoke tests for the MIMIC-IV W1 reproducer.

Two kinds of test live here:

- Data-free tests build a small synthetic cohort frame with the loader's column
  schema and a planted signal, then drive features -> model -> calibration ->
  evaluate. These always run and pin the estimator/calibrator contracts.
- Data-backed tests exercise the real load -> cross-validate pipeline on the
  demo cohort. They skip when the demo data is absent, so the suite is green on
  a checkout without the (un-committed) data.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import (
    BetaBayesCalibrator,
    evaluate_predictions,
    fit_predict_proba,
    reliability_curve,
)
from src.features import build_features_apply, build_features_fit
from src.loader import (
    CATEGORICAL_COLS,
    LAB_ITEMIDS,
    LABEL_COL,
    NUMERIC_BASE_COLS,
    VITAL_ITEMIDS,
    resolve_demo_root,
)
import run


def _synthetic_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """A cohort-schema frame whose label depends on a few features.

    Mirrors the columns of :class:`~src.loader.W1Cohort.frame` so the feature,
    model, calibration, and evaluation code can run without real data. Age and
    a couple of labs drive the mortality probability, so a competent classifier
    should score above chance.
    """
    rng = np.random.default_rng(seed)
    vitals = [v for v in VITAL_ITEMIDS if v not in ("temp_c", "temp_f")] + ["temperature"]
    frame = pd.DataFrame()
    frame["gender"] = rng.choice(["M", "F"], size=n)
    frame["admission_type"] = rng.choice(["EW EMER.", "URGENT", "ELECTIVE"], size=n)
    frame["admission_location"] = rng.choice(["EMERGENCY ROOM", "PHYSICIAN REFERRAL"], size=n)
    frame["insurance"] = rng.choice(["Medicare", "Medicaid", "Other"], size=n)
    age = rng.uniform(18, 95, size=n)
    frame["anchor_age"] = age
    frame["n_diagnoses"] = rng.poisson(10, size=n)
    for col in vitals:
        frame[col] = rng.normal(80, 15, size=n)
    creat = rng.uniform(0.5, 4.0, size=n)
    for col in LAB_ITEMIDS:
        frame[col] = rng.uniform(1, 10, size=n)
    frame["creatinine"] = creat
    logits = 0.04 * (age - 60) + 0.6 * (creat - 1.5) - 1.5
    prob = 1.0 / (1.0 + np.exp(-logits))
    frame[LABEL_COL] = (rng.uniform(size=n) < prob).astype(int)
    # Sprinkle missingness the imputer must handle.
    frame.loc[rng.uniform(size=n) < 0.1, "lactate"] = np.nan
    return frame


# --------------------------------------------------------------------------
# Data-free tests
# --------------------------------------------------------------------------

def test_feature_fit_apply_columns_align():
    frame = _synthetic_frame(seed=1)
    train, test = frame.iloc[:180], frame.iloc[180:]
    X_train, state = build_features_fit(train)
    X_test = build_features_apply(test, state)
    assert list(X_train.columns) == list(X_test.columns)
    assert X_train.notna().all().all() and X_test.notna().all().all()
    for col in CATEGORICAL_COLS:
        assert any(c.startswith(f"{col}=") for c in X_train.columns)
    for col in list(NUMERIC_BASE_COLS) + list(LAB_ITEMIDS):
        assert col in X_train.columns


def test_model_scores_above_chance_on_planted_signal():
    frame = _synthetic_frame(n=400, seed=2)
    train, test = frame.iloc[:300], frame.iloc[300:]
    X_train, state = build_features_fit(train)
    X_test = build_features_apply(test, state)
    proba = fit_predict_proba(X_train, train[LABEL_COL].to_numpy(), X_test, random_state=0)
    assert proba.shape == (len(test),)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
    metrics = evaluate_predictions(test[LABEL_COL].to_numpy(), proba)
    assert metrics["auroc"] > 0.55


def test_beta_bayes_calibrator_reduces_or_holds_ece():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=500)
    # Deliberately over-confident scores: push toward 0/1 away from the truth.
    base = np.where(y == 1, 0.6, 0.4)
    scores = np.clip(base + rng.normal(0, 0.2, size=500), 0.01, 0.99) ** 0.3
    before = evaluate_predictions(y, scores)["ece"]
    cal = BetaBayesCalibrator().fit_predict(scores, y)
    after = evaluate_predictions(y, cal)["ece"]
    assert ((cal >= 0.0) & (cal <= 1.0)).all()
    assert after <= before + 1e-6


def test_calibrator_identity_on_single_class_fold():
    scores = np.linspace(0.1, 0.9, 20)
    out = BetaBayesCalibrator().fit(scores, np.zeros(20, dtype=int)).predict(scores)
    assert np.allclose(out, scores)


def test_evaluate_requires_both_classes_and_curve_shapes():
    with pytest.raises(ValueError):
        evaluate_predictions(np.zeros(10, dtype=int), np.linspace(0, 1, 10))
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.15, 0.6])
    curve = reliability_curve(y, p, n_bins=5)
    assert len(curve["bin_mid"]) == len(curve["frac_pos"]) == len(curve["count"])
    assert sum(curve["count"]) == len(y)


# --------------------------------------------------------------------------
# Data-backed pipeline tests
# --------------------------------------------------------------------------

def _data_available() -> bool:
    try:
        resolve_demo_root()
        return True
    except FileNotFoundError:
        return False


requires_data = pytest.mark.skipif(
    not _data_available(),
    reason="MIMIC-IV demo not present; run python data/mimic_iv/fetch.py",
)


@requires_data
def test_cohort_loads_with_both_classes():
    from src import load_w1_cohort

    cohort = load_w1_cohort()
    assert cohort.n > 0
    assert set(np.unique(cohort.frame[LABEL_COL])) == {0, 1}
    assert 0.0 < cohort.prevalence < 1.0


@requires_data
def test_run_reproducer_sane_and_deterministic(tmp_path):
    results = run.run_reproducer(None, seed=42, n_seeds=2, folds=5)
    cal = results["calibrated"]
    assert 0.5 <= cal["auroc"] <= 1.0
    assert 0.0 <= cal["ece"] <= 1.0
    assert results["n_admissions"] == results["n_admissions"]
    again = run.run_reproducer(None, seed=42, n_seeds=2, folds=5)
    assert results == again


@requires_data
def test_main_exits_zero_and_writes_json(tmp_path):
    out = tmp_path / "results.json"
    code = run.main(["--seed", "42", "--n-seeds", "2", "--folds", "5", "--output", str(out)])
    assert code == 0
    payload = json.loads(out.read_text())
    assert "calibrated" in payload and "uncalibrated" in payload

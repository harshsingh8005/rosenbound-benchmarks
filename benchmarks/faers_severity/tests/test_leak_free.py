"""Leak-safety tests for the FAERS temporal split.

These are the checks that would have caught the W1 `n_diagnoses` leak had they
existed there: assert that no feature can carry information from after the
prediction point. For FAERS the prediction point is the temporal split, so the
feature vocabulary must be learned from the training window only, and no column
may be derived from the label.
"""

from __future__ import annotations

import numpy as np

from src import build_cohort
from src.features import build_features_apply, build_features_fit
from src.loader import LABEL_COL


def _report(rid: str, year: int, drug: str, reac: str, serious: str) -> dict:
    return {
        "safetyreportid": rid,
        "receivedate": f"{year}0601",
        "serious": serious,
        "sex": "1",
        "age": 50,
        "age_unit": "801",
        "reporter_qualification": "1",
        "reporter_country": "US",
        "reporttype": "1",
        "drugs": [{"name": drug, "char": "1", "indication": "PAIN"}],
        "reactions": [reac],
    }


def test_vocabulary_is_fit_on_training_window_only():
    # A drug and a reaction that appear ONLY in the test window (2013) must not
    # become feature columns: the vocabulary is fit on the training window.
    train = [_report(f"T{i}", 2012, "COMMONDRUG", "COMMONREAC", "1" if i % 2 else "2")
             for i in range(40)]
    test = [_report(f"E{i}", 2013, "TESTONLYDRUG", "TESTONLYREAC", "1" if i % 2 else "2")
            for i in range(20)]
    frame = build_cohort(train + test).frame

    X_train, state = build_features_fit(frame[frame["year"] == 2012])

    assert "COMMONDRUG" in state.top_drugs
    assert "TESTONLYDRUG" not in state.top_drugs
    assert "TESTONLYREAC" not in state.top_reacs
    assert not any("TESTONLYDRUG" in c for c in state.columns)
    assert not any("TESTONLYREAC" in c for c in state.columns)


def test_apply_never_introduces_test_derived_columns():
    train = [_report(f"T{i}", 2012, "COMMONDRUG", "COMMONREAC", "1" if i % 2 else "2")
             for i in range(40)]
    test = [_report(f"E{i}", 2013, "TESTONLYDRUG", "TESTONLYREAC", "1" if i % 2 else "2")
            for i in range(20)]
    frame = build_cohort(train + test).frame

    X_train, state = build_features_fit(frame[frame["year"] == 2012])
    X_test = build_features_apply(frame[frame["year"] == 2013], state)

    # The test matrix has exactly the training columns — a test-only token that
    # is unseen in training contributes an all-zero (already-known) column, not
    # a new feature.
    assert list(X_test.columns) == list(X_train.columns)


def test_label_is_not_a_feature():
    frame = build_cohort(
        [_report(f"T{i}", 2012, "COMMONDRUG", "COMMONREAC", "1" if i % 2 else "2")
         for i in range(40)]
    ).frame
    X, _ = build_features_fit(frame)
    assert LABEL_COL not in X.columns
    # No feature perfectly reproduces the label (a crude tautology guard).
    y = frame[LABEL_COL].to_numpy(dtype=float)
    for col in X.columns:
        assert not np.array_equal(X[col].to_numpy(dtype=float), y)

"""End-to-end smoke tests for the FAERS severity reproducer.

- Data-free tests build synthetic openFDA-shaped reports with a planted signal
  and drive loader -> features -> model -> evaluate, so they run anywhere.
- A data-backed test exercises the committed demo slice and pins the canonical
  demo metrics. The slice is committed (FAERS is public domain), so this test
  always runs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import build_cohort, evaluate_predictions
from src.features import build_features_apply, build_features_fit
from src.loader import DEMO_SLICE, LABEL_COL
import run


def _synthetic_reports(n: int = 800, seed: int = 0) -> list[dict]:
    """openFDA-shaped reports whose seriousness depends on a few features.

    A high-risk drug and a specific reaction raise the probability of a serious
    label, so a competent classifier scores above chance. Reports span two
    receive years (2012 train, 2013 test) for the temporal split.
    """
    rng = np.random.default_rng(seed)
    reactions_pool = ["NAUSEA", "HEADACHE", "DIZZINESS", "RASH", "FATIGUE"]
    benign_drugs = ["ACETAMINOPHEN", "IBUPROFEN", "LORATADINE", "OMEPRAZOLE"]
    reports = []
    for i in range(n):
        year = 2012 if i < int(n * 0.65) else 2013
        high_risk = rng.random() < 0.4
        drug_name = "WARFARIN SODIUM" if high_risk else rng.choice(benign_drugs)
        reacs = list(rng.choice(reactions_pool, size=rng.integers(1, 3), replace=False))
        bad_reac = rng.random() < 0.5
        if bad_reac:
            reacs.append("CARDIAC ARREST")
        logit = 1.4 * high_risk + 1.1 * bad_reac - 1.2
        serious = rng.random() < 1.0 / (1.0 + np.exp(-logit))
        reports.append({
            "safetyreportid": f"SR{i}",
            "receivedate": f"{year}0401",
            "serious": "1" if serious else "2",
            "sex": str(rng.integers(1, 3)),
            "age": int(rng.integers(20, 90)),
            "age_unit": "801",
            "reporter_qualification": str(rng.integers(1, 6)),
            "reporter_country": "US",
            "reporttype": "1",
            "drugs": [{"name": drug_name, "char": "1", "indication": "PAIN"}],
            "reactions": reacs,
        })
    return reports


# --------------------------------------------------------------------------
# Data-free tests
# --------------------------------------------------------------------------

def test_cohort_parses_reports_and_labels():
    cohort = build_cohort(_synthetic_reports(seed=1))
    frame = cohort.frame
    assert cohort.n == 800
    assert set(np.unique(frame[LABEL_COL])) == {0, 1}
    assert set(frame["year"].unique()) == {2012, 2013}


def test_feature_fit_apply_columns_align_and_impute():
    cohort = build_cohort(_synthetic_reports(seed=2))
    frame = cohort.frame
    train = frame[frame["year"] == 2012]
    test = frame[frame["year"] == 2013]
    X_train, state = build_features_fit(train)
    X_test = build_features_apply(test, state)
    assert list(X_train.columns) == list(X_test.columns)
    assert X_train.notna().all().all() and X_test.notna().all().all()
    assert LABEL_COL not in X_train.columns


def test_pipeline_scores_above_chance_on_planted_signal():
    cohort = build_cohort(_synthetic_reports(n=1200, seed=3))
    results = run.run_reproducer(None, seed=0, split_year=2013, cohort=cohort)
    assert results["metrics"]["auroc"] > 0.6


def test_evaluate_requires_both_classes():
    with pytest.raises(ValueError):
        evaluate_predictions(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


# --------------------------------------------------------------------------
# Data-backed test (committed demo slice)
# --------------------------------------------------------------------------

def test_demo_slice_present():
    assert DEMO_SLICE.is_file(), "committed FAERS demo slice is missing"


def test_run_reproducer_matches_pinned_targets(tmp_path):
    out = tmp_path / "results.json"
    code = run.main(["--ci", "--output", str(out)])
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["metrics"]["auroc"] > 0.5


def test_run_reproducer_deterministic():
    a = run.run_reproducer(None)
    b = run.run_reproducer(None)
    assert a == b

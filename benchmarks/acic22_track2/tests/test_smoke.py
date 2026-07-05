"""End-to-end smoke tests for the ACIC22 Track-2 reproducer.

Two kinds of test live here:

- Data-free tests build a synthetic dataset with a known effect and confirm the
  estimators recover it and that the sensitivity bound behaves monotonically.
  These always run.
- Data-backed tests exercise the full load -> estimate -> aggregate pipeline on
  a handful of real cohorts. They skip when the Track-2 data is not present, so
  the suite is green on a checkout without the (large, un-committed) data.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import (
    aipw_att,
    dr_att,
    enumerate_track2_cohorts,
    load_track2_cohort,
    rosenbaum_gamma_zero,
)
import run


def _synthetic(n: int = 1200, true_att: float = 4.0, seed: int = 0):
    """A confounded dataset whose treated-group effect is exactly ``true_att``.

    Treatment depends on the covariates (so a naive difference is biased), and
    the outcome is linear in the covariates plus the effect on the treated, so
    a correct doubly-robust estimator should recover ``true_att``.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    logits = 0.8 * X[:, 0] - 0.6 * X[:, 1]
    e = 1.0 / (1.0 + np.exp(-logits))
    T = (rng.uniform(size=n) < e).astype(int)
    Y = X @ np.array([1.5, -1.0, 0.5, 0.2]) + true_att * T + rng.normal(scale=1.0, size=n)
    return X, T, Y


# --------------------------------------------------------------------------
# Data-free estimator tests
# --------------------------------------------------------------------------

def test_aipw_recovers_known_effect():
    X, T, Y = _synthetic(true_att=4.0)
    res = aipw_att(X, T, Y, random_state=0)
    assert abs(res.att - 4.0) < 1.0
    assert res.ci_lo_95 < res.att < res.ci_hi_95
    assert res.se > 0.0


def test_dr_att_recovers_known_effect():
    X, T, Y = _synthetic(true_att=4.0)
    res = dr_att(X, T, Y, random_state=0)
    assert abs(res.att - 4.0) < 1.0
    assert res.n_trimmed >= 0


def test_rosenbaum_monotone_and_finite():
    strong = rosenbaum_gamma_zero(estimated_effect=10.0, se_effect=1.0)
    weak = rosenbaum_gamma_zero(estimated_effect=2.0, se_effect=1.0)
    # A stronger signal tolerates a stronger confounder before reaching zero.
    assert strong.gamma_zero > weak.gamma_zero
    assert np.isfinite(strong.gamma_zero)
    assert strong.note == "widening-bound simplification"
    # Not significant -> already sensitive at Gamma = 1.
    assert rosenbaum_gamma_zero(1.0, 5.0).gamma_zero == 1.0


# --------------------------------------------------------------------------
# Data-backed pipeline tests
# --------------------------------------------------------------------------

def _data_available() -> bool:
    try:
        return len(enumerate_track2_cohorts()) > 0
    except FileNotFoundError:
        return False


requires_data = pytest.mark.skipif(
    not _data_available(),
    reason="ACIC22 Track-2 data not present; see data/acic22/README.md",
)


@requires_data
def test_selection_is_deterministic_and_nested():
    five = run.select_cohorts(5, seed=42)
    ten = run.select_cohorts(10, seed=42)
    assert five == run.select_cohorts(5, seed=42)  # deterministic
    assert set(five).issubset(set(ten))            # nested


@requires_data
def test_single_cohort_reduction_shapes():
    cohort = load_track2_cohort(enumerate_track2_cohorts()[0])
    assert cohort.X.ndim == 2 and cohort.X.shape[0] == cohort.T.shape[0]
    assert cohort.Y.shape[0] == cohort.T.shape[0]
    assert set(np.unique(cohort.T)).issubset({0, 1})
    assert np.isfinite(cohort.true_ate)


@requires_data
def test_end_to_end_five_cohorts_sane(tmp_path):
    ids = run.select_cohorts(5, seed=42)
    results = run.run_reproducer(ids, ("aipw", "dr_att", "rosenbaum"), seed=42)

    assert results["n_cohorts"] == 5
    for method in ("aipw", "dr_att"):
        m = results[method]
        assert np.isfinite(m["bias_mean"]) and abs(m["bias_mean"]) < 100.0
        assert m["rmse"] > 0.0
        assert 0.0 <= m["coverage_95"] <= 1.0
        assert m["ci_width_mean"] > 0.0
    assert results["rosenbaum"]["gamma_zero_median"] >= 1.0

    # Determinism: a second identical run yields identical aggregates.
    again = run.run_reproducer(ids, ("aipw", "dr_att", "rosenbaum"), seed=42)
    assert results == again


@requires_data
def test_main_exits_zero_and_writes_json(tmp_path):
    out = tmp_path / "results.json"
    code = run.main(["--n-cohorts", "5", "--seed", "42", "--output", str(out)])
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["n_cohorts"] == 5
    assert "dr_att" in payload

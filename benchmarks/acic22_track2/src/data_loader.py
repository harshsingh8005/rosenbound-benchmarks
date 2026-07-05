"""Loader for ACIC22 Track-2 cohorts.

The Track-2 release is a panel: each cohort holds ~500 practices observed over
four years, with practice-level covariates, a practice-level treatment
indicator ``Z``, and a per-practice-year continuous outcome ``Y`` (a Medicare
expenditure measure). Treatment acts only in the post-intervention years,
flagged by ``post``. The challenge's ground truth is the sample average
treatment effect on the treated (SATT).

To bring this panel into the ``(X, T, Y)`` form the cross-sectional
doubly-robust estimators consume, each cohort is reduced with a practice-level
difference-in-differences change score:

- ``T`` is the practice treatment indicator ``Z``.
- ``Y`` is the change score ``Ybar_post - Ybar_pre``, the patient-weighted
  mean outcome in post years minus that in pre years. Differencing removes
  each practice's time-invariant level, so under parallel trends conditional
  on the covariates the effect of ``Z`` on this change score identifies the
  SATT.
- ``X`` stacks the practice covariates (categoricals one-hot encoded) with
  pre-period summaries: the pre-period outcome level, mean panel size, and
  the pre-period patient-mix averages. Conditioning on the pre-period level
  and mix is what makes the parallel-trends assumption plausible.

``true_ate`` is the cohort's overall SATT taken directly from the challenge
estimand-truth table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_N_COHORTS = 3400


@dataclass(frozen=True)
class Track2Cohort:
    """A single ACIC22 Track-2 cohort reduced to ``(X, T, Y)`` plus truth.

    Attributes
    ----------
    cohort_id
        Integer cohort identifier (1-based).
    X
        Practice-level covariate matrix, shape ``(n_practices, p)``.
    T
        Practice treatment indicator ``Z``, shape ``(n_practices,)``.
    Y
        Difference-in-differences change score, shape ``(n_practices,)``.
    true_ate
        Overall SATT for the cohort from the challenge truth table.
    feature_names
        Column labels for ``X``.
    """

    cohort_id: int
    X: NDArray[np.float64]
    T: NDArray[np.int_]
    Y: NDArray[np.float64]
    true_ate: float
    feature_names: tuple[str, ...]

    @property
    def n(self) -> int:
        return int(self.X.shape[0])


def _data_root() -> Path:
    """Resolve the directory holding ``practice/`` and ``practice_year/``.

    Honours the ``ACIC22_DATA_ROOT`` environment variable; otherwise defaults
    to ``data/acic22`` at the repository root. Accepts either the directory
    itself or a ``track2`` subdirectory beneath it.
    """
    env = os.environ.get("ACIC22_DATA_ROOT")
    base = Path(env) if env else Path(__file__).resolve().parents[3] / "data" / "acic22"
    for cand in (base / "track2", base):
        if (cand / "practice").is_dir() and (cand / "practice_year").is_dir():
            return cand
    raise FileNotFoundError(
        f"ACIC22 Track-2 data not found under {base}. Expected "
        f"'practice/' and 'practice_year/' subdirectories. "
        f"See data/acic22/README.md, or set ACIC22_DATA_ROOT."
    )


@lru_cache(maxsize=1)
def _truth_table() -> dict[int, float]:
    """Map cohort id -> overall SATT from the estimand-truth table (cached).

    The truth CSV lists many estimands per cohort; the overall SATT is the
    row whose ``variable`` is ``Overall``.
    """
    path = _data_root().parent / "ACIC_estimand_truths.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"estimand-truth table not found at {path}. It ships alongside "
            f"the Track-2 archive; see data/acic22/README.md."
        )
    df = pd.read_csv(path, usecols=["dataset.num", "variable", "SATT"])
    overall = df[df["variable"] == "Overall"]
    return {
        int(row["dataset.num"]): float(row["SATT"])
        for _, row in overall.iterrows()
    }


def _weighted_mean(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return float(values.mean())
    return float(np.dot(values, weights) / total)


def enumerate_track2_cohorts() -> list[int]:
    """Return the sorted list of Track-2 cohort ids present on disk."""
    practice_dir = _data_root() / "practice"
    ids = sorted(
        int(p.stem.split("_")[-1])
        for p in practice_dir.glob("acic_practice_*.csv")
    )
    return ids


def load_track2_cohort(cohort_id: int) -> Track2Cohort:
    """Load and reduce one Track-2 cohort to ``(X, T, Y, true_ate)``.

    Parameters
    ----------
    cohort_id
        1-based cohort identifier (e.g. ``1`` for ``acic_practice_0001``).

    Returns
    -------
    Track2Cohort
    """
    root = _data_root()
    tag = f"{cohort_id:04d}"
    practice_path = root / "practice" / f"acic_practice_{tag}.csv"
    panel_path = root / "practice_year" / f"acic_practice_year_{tag}.csv"
    if not practice_path.is_file() or not panel_path.is_file():
        raise FileNotFoundError(
            f"cohort {cohort_id} not found ({practice_path} / {panel_path})"
        )

    practice = pd.read_csv(practice_path)
    panel = pd.read_csv(panel_path)

    # One practice-level row: Z, and pre/post change score + pre summaries.
    mix_cols = [c for c in panel.columns if c.endswith("_avg")]
    records = []
    for pid, g in panel.groupby("id.practice", sort=True):
        pre = g[g["post"] == 0]
        post = g[g["post"] == 1]
        if len(pre) == 0 or len(post) == 0:
            continue
        w_pre = pre["n.patients"].to_numpy(dtype=np.float64)
        w_post = post["n.patients"].to_numpy(dtype=np.float64)
        y_pre = _weighted_mean(pre["Y"].to_numpy(dtype=np.float64), w_pre)
        y_post = _weighted_mean(post["Y"].to_numpy(dtype=np.float64), w_post)
        rec: dict[str, float] = {
            "id.practice": int(pid),
            "Z": int(g["Z"].iloc[0]),
            "dY": y_post - y_pre,
            "pre_Y": y_pre,
            "pre_n_patients": float(np.mean(w_pre)),
        }
        for c in mix_cols:
            rec[f"pre_{c}"] = _weighted_mean(
                pre[c].to_numpy(dtype=np.float64), w_pre
            )
        records.append(rec)

    reduced = pd.DataFrame.from_records(records)
    merged = reduced.merge(practice, on="id.practice", how="inner")

    T = merged["Z"].to_numpy(dtype=np.int_)
    Y = merged["dY"].to_numpy(dtype=np.float64)

    covariate_cols = [
        c for c in merged.columns
        if c not in ("id.practice", "Z", "dY")
    ]
    X_df = merged[covariate_cols]
    # Practice covariates X2, X4 (and any others) may be categorical letters;
    # one-hot encode every non-numeric column, leave numerics as-is.
    X_df = pd.get_dummies(X_df, drop_first=True, dtype=np.float64)
    X = X_df.to_numpy(dtype=np.float64)

    true_ate = _truth_table().get(cohort_id)
    if true_ate is None:
        raise KeyError(f"no truth-table SATT for cohort {cohort_id}")

    return Track2Cohort(
        cohort_id=cohort_id,
        X=X,
        T=T,
        Y=Y,
        true_ate=float(true_ate),
        feature_names=tuple(X_df.columns),
    )


__all__ = [
    "Track2Cohort",
    "enumerate_track2_cohorts",
    "load_track2_cohort",
]

"""Turn the raw W1 cohort frame into a numeric model matrix.

Feature engineering is split into a fit step and an apply step so that every
data-dependent transform — the one-hot category vocabulary and the per-column
imputation medians — is learned on the training fold only and replayed on
held-out folds. Fitting these on the whole dataset would leak test-fold
statistics into training and inflate the reported metrics; keeping the two
steps separate makes the pipeline leakage-free by construction.

The matrix has three feature blocks:

- categorical admission descriptors, one-hot encoded against the training
  vocabulary (unseen categories at apply time fall through to all-zero);
- numeric demographics and the diagnosis count;
- first-24h vital and lab means, median-imputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .loader import (
    CATEGORICAL_COLS,
    LAB_ITEMIDS,
    LABEL_COL,
    NUMERIC_BASE_COLS,
    VITAL_ITEMIDS,
)

# Numeric feature columns in canonical order: demographics + diagnosis count,
# then first-24h vitals, then first-24h labs.
_VITAL_COLS = [v for v in VITAL_ITEMIDS if v not in ("temp_c", "temp_f")] + ["temperature"]
_NUMERIC_COLS = list(NUMERIC_BASE_COLS) + _VITAL_COLS + list(LAB_ITEMIDS.keys())

# Human-readable description of the feature blocks, surfaced by the notebook.
FEATURE_SPEC = {
    "categorical": list(CATEGORICAL_COLS),
    "numeric": _NUMERIC_COLS,
}


@dataclass(frozen=True)
class FeatureState:
    """Fitted transform learned on a training fold and replayed elsewhere.

    Attributes
    ----------
    categories
        Ordered category levels per categorical column, learned from the
        training fold; drives one-hot encoding with a stable column set.
    medians
        Per-numeric-column training medians used to impute missing values.
    columns
        Final ordered column set of the encoded matrix.
    """

    categories: dict[str, list[str]]
    medians: dict[str, float]
    columns: list[str] = field(default_factory=list)


def _one_hot(frame: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """One-hot encode categoricals against a fixed vocabulary (drop-first)."""
    blocks: list[pd.DataFrame] = []
    for col, levels in categories.items():
        values = frame[col].astype("object").where(frame[col].notna(), other="__missing__")
        # Drop the first level to avoid collinearity; unseen values map to
        # an all-zero row across this column's dummies.
        for level in levels[1:]:
            blocks.append((values == level).astype(np.float64).rename(f"{col}={level}"))
    if not blocks:
        return pd.DataFrame(index=frame.index)
    return pd.concat(blocks, axis=1)


def build_features_fit(frame: pd.DataFrame) -> tuple[pd.DataFrame, FeatureState]:
    """Fit the encoder + imputer on a training fold and transform it.

    Parameters
    ----------
    frame
        Raw cohort rows for the training fold (may include ``LABEL_COL``,
        which is ignored here).

    Returns
    -------
    (X, state)
        ``X`` is the encoded training matrix; ``state`` replays the same
        transform on other folds via :func:`build_features_apply`.
    """
    categories: dict[str, list[str]] = {}
    for col in CATEGORICAL_COLS:
        values = frame[col].astype("object").where(frame[col].notna(), other="__missing__")
        categories[col] = sorted(str(v) for v in values.unique())

    medians = {
        col: float(frame[col].astype(float).median()) if frame[col].notna().any() else 0.0
        for col in _NUMERIC_COLS
    }

    state = FeatureState(categories=categories, medians=medians)
    X = _transform(frame, state)
    object.__setattr__(state, "columns", list(X.columns))
    return X, state


def build_features_apply(frame: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    """Transform a held-out fold with a previously fitted :class:`FeatureState`."""
    X = _transform(frame, state)
    return X.reindex(columns=state.columns, fill_value=0.0)


def _transform(frame: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    cat = _one_hot(frame, state.categories)
    num = pd.DataFrame(index=frame.index)
    for col in _NUMERIC_COLS:
        series = frame[col].astype(float) if col in frame.columns else pd.Series(
            np.nan, index=frame.index
        )
        num[col] = series.fillna(state.medians[col])
    return pd.concat([num, cat], axis=1).astype(np.float64)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit-and-transform in one call (whole-frame convenience, e.g. for EDA).

    This fits the imputer and encoder on the same ``frame`` it transforms, so it
    must not be used to build train/test matrices for scoring — use the
    :func:`build_features_fit` / :func:`build_features_apply` pair there.
    """
    X, _ = build_features_fit(frame)
    return X


__all__ = [
    "FEATURE_SPEC",
    "FeatureState",
    "build_features",
    "build_features_fit",
    "build_features_apply",
]

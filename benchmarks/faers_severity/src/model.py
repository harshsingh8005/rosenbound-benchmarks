"""LightGBM severity classifier for the FAERS reproducer.

A classical gradient-boosted-trees classifier (LightGBM; Ke et al., 2017) over
the symbolic FAERS feature matrix — no learned text embeddings anywhere in the
path, matching the interpretable public-method baseline. The parameters suit a
few-thousand-row slice with a wide sparse top-K bag: shallow trees, column
subsampling, and L2 leaf regularisation. All randomness is pinned to
``random_state`` and the fit is single-threaded, so a fold reproduces
bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_lambda": 1.0,
    "n_jobs": 1,
    "verbose": -1,
}


def fit_predict_proba(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    params: dict | None = None,
    random_state: int = 0,
) -> np.ndarray:
    """Fit LightGBM on the training window and return ``P(serious=1)`` on the test.

    ``X_train`` and ``X_test`` must share columns. A degenerate single-class
    training window falls back to the training base rate.
    """
    from lightgbm import LGBMClassifier

    merged = {**DEFAULT_PARAMS, **(params or {}), "random_state": random_state}
    model = LGBMClassifier(**merged)
    model.fit(X_train, np.asarray(y_train).astype(int))

    classes = list(model.classes_)
    if 1 not in classes:
        return np.full(len(X_test), float(np.mean(y_train)))
    idx = classes.index(1)
    return model.predict_proba(X_test)[:, idx]


__all__ = ["DEFAULT_PARAMS", "fit_predict_proba"]

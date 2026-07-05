"""Gradient-boosted-trees classifier for W1 in-hospital mortality.

The model is LightGBM (Ke et al., 2017). Its defaults are tuned for large
datasets; the parameters here are set for the small, class-imbalanced W1
cohort — shallow trees, few leaves, a floor on leaf occupancy, and L2 leaf
regularisation — so the fit does not memorise a hundred-row training fold. All
randomness is pinned to ``random_state`` and the fit is single-threaded, so a
given fold reproduces bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Small-data defaults. LightGBM's stock settings (num_leaves=31, unbounded
# depth, min_child_samples=20) overfit or degenerate on ~100-row folds; these
# keep the trees shallow and the leaves populated.
DEFAULT_PARAMS: dict = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 5,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
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
    """Fit LightGBM on the training fold and return ``P(y=1)`` on the test fold.

    Parameters
    ----------
    X_train, X_test
        Encoded numeric feature matrices with identical columns.
    y_train
        Binary training labels, shape ``(n_train,)``.
    params
        Optional overrides merged over :data:`DEFAULT_PARAMS`.
    random_state
        Seed for LightGBM's bagging and feature sampling.

    Returns
    -------
    numpy.ndarray
        Predicted positive-class probabilities on ``X_test``, shape
        ``(n_test,)``, each in ``[0, 1]``.
    """
    from lightgbm import LGBMClassifier

    merged = {**DEFAULT_PARAMS, **(params or {}), "random_state": random_state}
    model = LGBMClassifier(**merged)
    model.fit(X_train, np.asarray(y_train).astype(int))

    classes = list(model.classes_)
    if 1 not in classes:
        # Degenerate training fold with a single class: predict its base rate.
        return np.full(len(X_test), float(np.mean(y_train)))
    idx = classes.index(1)
    return model.predict_proba(X_test)[:, idx]


__all__ = ["DEFAULT_PARAMS", "fit_predict_proba"]

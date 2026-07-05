"""Nuisance-model factories shared by the doubly-robust estimators.

Both AIPW and DR-ATT require a propensity model ``e(X) = P(T=1 | X)`` and an
outcome regression. Centralising the model construction here keeps the two
estimators consistent (same ``ml_model`` string selects the same learner in
both) and keeps the estimator modules focused on the causal arithmetic.

The default learner is ``logit_ridge``: a standardized, L2-regularized
logistic propensity model paired with a ridge outcome regression. On the
ACIC22 Track-2 cohorts -- roughly 500 practices per cohort, with a noisy
difference-in-differences change-score outcome -- the regularized linear
nuisances generalize markedly better than gradient boosting, whose flexible
propensity fit drives inverse-propensity weights to their clip boundaries and
inflates both bias and variance. Gradient boosting and random forests remain
selectable for cohorts or datasets where a more flexible nuisance fit is
warranted. All learners are CPU-only, so the reproducer has no GPU dependency.
"""

from __future__ import annotations

from typing import Any

from sklearn.pipeline import Pipeline

_SUPPORTED = ("logit_ridge", "hist_gbm", "lgbm", "rf")


def _check(ml_model: str) -> None:
    if ml_model not in _SUPPORTED:
        raise ValueError(
            f"ml_model {ml_model!r} not in {_SUPPORTED}"
        )


def make_classifier(ml_model: str, random_state: int) -> Any:
    """Return an unfitted propensity classifier for ``ml_model``.

    The classifier must expose ``predict_proba``; the estimators use the
    probability of the treated class as the propensity score.
    """
    _check(ml_model)
    if ml_model == "logit_ridge":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=1000)),
        ])
    if ml_model == "hist_gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.1, max_iter=200,
            random_state=random_state,
        )
    if ml_model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1,
            random_state=random_state,
        )
    from lightgbm import LGBMClassifier  # ml_model == "lgbm"

    return LGBMClassifier(
        n_estimators=300, num_leaves=15, learning_rate=0.05,
        random_state=random_state, verbose=-1,
    )


def make_regressor(ml_model: str, random_state: int) -> Any:
    """Return an unfitted outcome regressor for ``ml_model``."""
    _check(ml_model)
    if ml_model == "logit_ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=10.0, random_state=random_state)
    if ml_model == "hist_gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.1, max_iter=200,
            random_state=random_state,
        )
    if ml_model == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1,
            random_state=random_state,
        )
    from lightgbm import LGBMRegressor  # ml_model == "lgbm"

    return LGBMRegressor(
        n_estimators=300, num_leaves=15, learning_rate=0.05,
        random_state=random_state, verbose=-1,
    )


__all__ = ["make_classifier", "make_regressor"]

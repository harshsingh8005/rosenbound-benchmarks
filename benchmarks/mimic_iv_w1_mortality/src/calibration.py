"""Beta calibration with a Bayesian (L2-MAP) smoothing prior.

Beta calibration (Kull, Silva Filho & Flach, 2017, *JMLR*) recalibrates a
classifier's scores ``s`` with a three-parameter family:

    logit(p_cal) = a * ln(s) - b * ln(1 - s) + c

The three parameters are fit by logistic regression of the labels on the two
features ``ln(s)`` and ``-ln(1 - s)``. Unlike Platt scaling it can bend the
reliability curve in both directions, which fits the sigmoid-shaped
miscalibration a boosted-trees model typically shows.

On the small W1 validation folds an unregularised fit is unstable, so the
parameters are estimated by maximum a posteriori under a zero-mean Gaussian
prior — equivalent to L2-regularised logistic regression. The prior is the
"Bayes" smoothing: it shrinks the calibrator toward the identity map when the
fold carries little information, and vanishes as the fold grows. A fold with a
single label class carries no calibration signal at all and the calibrator
falls back to the identity transform.
"""

from __future__ import annotations

import numpy as np

# Scores are clipped away from {0, 1} before the log-odds transform so ln(s)
# and ln(1 - s) stay finite.
_EPS = 1e-6


class BetaBayesCalibrator:
    """Fit-once, apply-many Beta calibrator with Gaussian-prior smoothing.

    Parameters
    ----------
    prior_strength
        Precision of the zero-mean Gaussian prior on ``(a, b, c)``. Larger
        values shrink harder toward the identity map; maps to
        ``C = 1 / prior_strength`` in the underlying logistic regression.

    Lifecycle
    ---------
    Construct, call :meth:`fit` on validation-fold ``(scores, y)``, then
    :meth:`predict` on any scores. An unfitted or degenerate calibrator
    applies the identity transform.
    """

    def __init__(self, prior_strength: float = 1.0) -> None:
        self._prior_strength = float(prior_strength)
        self._model = None  # fitted sklearn LogisticRegression, or None (identity)

    @staticmethod
    def _design(scores: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(scores, dtype=np.float64), _EPS, 1.0 - _EPS)
        return np.column_stack([np.log(s), -np.log1p(-s)])

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "BetaBayesCalibrator":
        """Fit the calibrator on validation-fold scores and labels.

        A fold with fewer than two label classes leaves the calibrator as the
        identity transform. Returns ``self``.
        """
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            self._model = None
            return self
        X = self._design(scores)
        # Default L2 penalty is the Gaussian-prior MAP; C = 1 / prior precision.
        model = LogisticRegression(
            C=1.0 / self._prior_strength,
            solver="lbfgs",
            max_iter=1000,
        )
        model.fit(X, y)
        self._model = model
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Return calibrated positive-class probabilities for ``scores``."""
        if self._model is None:
            return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
        X = self._design(scores)
        return self._model.predict_proba(X)[:, 1]

    def fit_predict(self, scores: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Convenience: fit on ``(scores, y)`` then calibrate the same scores."""
        return self.fit(scores, y).predict(scores)


__all__ = ["BetaBayesCalibrator"]

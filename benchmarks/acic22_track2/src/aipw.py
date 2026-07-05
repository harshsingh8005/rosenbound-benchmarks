"""Cross-fitted AIPW estimator for the average effect on the treated.

The augmented inverse-probability-weighting (AIPW) estimator combines an
outcome regression with a propensity model so that the effect estimate stays
consistent when *either* nuisance model is correctly specified (the
double-robustness property). Cross-fitting the two nuisance models on
held-out folds removes the own-observation bias that plugging in an
over-fit learner would otherwise introduce, which is what licenses the
root-n normal inference used for the standard error and interval below.

The estimand is the effect on the treated (ATT). The ACIC22 Track-2
challenge scores the sample average treatment effect on the treated
(SATT), so the treated-side estimand is the one that matches the
challenge's ground truth; the outcome model is therefore fit on control
units and used to impute each treated unit's untreated counterfactual.

References
----------
Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of
regression coefficients when some regressors are not always observed.
Journal of the American Statistical Association, 89(427), 846-866.

Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing
data and causal inference models. Biometrics, 61(4), 962-973.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W., & Robins, J. (2018). Double/debiased machine learning for
treatment and structural parameters. The Econometrics Journal, 21(1),
C1-C68.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from ._nuisance import make_classifier, make_regressor

# Clip cross-fitted propensities away from 0/1 so the control-side inverse
# weights stay bounded. This is a stabilization on the score, not an overlap
# trim -- no units are dropped (contrast dr_att, which trims).
_CLIP = (0.02, 0.98)


@dataclass(frozen=True)
class AipwResult:
    """Result of a cross-fitted AIPW-ATT estimate.

    Attributes
    ----------
    att
        Point estimate of the effect on the treated.
    se
        Standard error from the efficient influence function.
    ci_lo_95, ci_hi_95
        Bounds of the 95% Wald confidence interval.
    n_effective
        Number of treated units the estimand averages over.
    propensity_range
        ``(min, max)`` of the cross-fitted propensity scores, a quick
        overlap diagnostic.
    """

    att: float
    se: float
    ci_lo_95: float
    ci_hi_95: float
    n_effective: int
    propensity_range: tuple[float, float]


def _cross_fit_nuisances(
    X: NDArray[np.float64],
    T: NDArray[np.int_],
    Y: NDArray[np.float64],
    k_folds: int,
    ml_model: str,
    random_state: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return out-of-fold propensity ``e_hat`` and control-outcome ``mu0_hat``.

    For each fold the nuisance models are fit on the other folds and applied
    to the held-out fold, so every prediction is out-of-sample. ``mu0`` is
    the control-outcome regression E[Y | X, T=0].
    """
    n = X.shape[0]
    e_hat = np.empty(n, dtype=np.float64)
    mu0_hat = np.empty(n, dtype=np.float64)

    skf = StratifiedKFold(
        n_splits=k_folds, shuffle=True, random_state=random_state
    )
    for train_idx, test_idx in skf.split(X, T):
        clf = make_classifier(ml_model, random_state)
        clf.fit(X[train_idx], T[train_idx])
        e_hat[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

        ctrl = train_idx[T[train_idx] == 0]
        reg = make_regressor(ml_model, random_state)
        reg.fit(X[ctrl], Y[ctrl])
        mu0_hat[test_idx] = reg.predict(X[test_idx])

    np.clip(e_hat, _CLIP[0], _CLIP[1], out=e_hat)
    return e_hat, mu0_hat


def aipw_att(
    X: NDArray[np.float64],
    T: NDArray[np.int_],
    Y: NDArray[np.float64],
    k_folds: int = 5,
    ml_model: str = "logit_ridge",
    random_state: int = 0,
) -> AipwResult:
    """Estimate the effect on the treated by cross-fitted AIPW.

    Parameters
    ----------
    X
        Covariate matrix, shape ``(n, p)``.
    T
        Binary treatment indicator, shape ``(n,)``.
    Y
        Outcome, shape ``(n,)``.
    k_folds
        Number of cross-fitting folds.
    ml_model
        Nuisance learner: ``"logit_ridge"`` (default), ``"hist_gbm"``,
        ``"lgbm"``, or ``"rf"``.
    random_state
        Seed for the fold split and the nuisance learners.

    Returns
    -------
    AipwResult
        Point estimate, influence-function standard error, 95% interval,
        the treated count, and the propensity range.

    Notes
    -----
    With out-of-fold nuisances ``e(X)`` and ``mu0(X)``, the point estimate
    is the treated-side augmented contrast with self-normalized (Hajek)
    control weights ``w_i = e_i / (1 - e_i)``:

        att = mean_{T=1}(Y - mu0)
              - sum_{T=0} w_i (Y_i - mu0_i) / sum_{T=0} w_i

    Self-normalizing the control term bounds the inverse-propensity
    contribution, which keeps the estimate stable under the limited overlap
    typical of these cohorts. The standard error uses the efficient
    influence function for the ATT,

        phi_i = (1/p) [ T_i (Y_i - mu0_i) - (1-T_i) w_i (Y_i - mu0_i) ]
                - (T_i / p) att,

    with ``p = P(T=1)``, reported as ``sd(phi) / sqrt(n)``.
    """
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T).astype(np.int_)
    Y = np.asarray(Y, dtype=np.float64)

    n = X.shape[0]
    n_treated = int((T == 1).sum())
    if n_treated < k_folds or (T == 0).sum() < k_folds:
        raise ValueError(
            f"each arm must have >= k_folds units: "
            f"treated={n_treated}, control={int((T == 0).sum())}, "
            f"k_folds={k_folds}"
        )

    e_hat, mu0_hat = _cross_fit_nuisances(
        X, T, Y, k_folds, ml_model, random_state
    )

    p_hat = n_treated / n
    resid = Y - mu0_hat
    w = e_hat / (1.0 - e_hat)  # control inverse-propensity weight

    is_ctrl = T == 0
    treated_term = float(resid[T == 1].mean())
    control_term = float(
        np.dot(w[is_ctrl], resid[is_ctrl]) / w[is_ctrl].sum()
    )
    att = treated_term - control_term

    # Efficient influence function for the ATT (self-normalized control
    # weights coincide asymptotically with the plug-in efficient form).
    infl = (
        np.where(T == 1, resid, -w * resid) / p_hat
        - (T / p_hat) * att
    )
    se = float(infl.std(ddof=1) / np.sqrt(n))

    return AipwResult(
        att=att,
        se=se,
        ci_lo_95=att - 1.959963984540054 * se,
        ci_hi_95=att + 1.959963984540054 * se,
        n_effective=n_treated,
        propensity_range=(float(e_hat.min()), float(e_hat.max())),
    )


__all__ = ["AipwResult", "aipw_att"]

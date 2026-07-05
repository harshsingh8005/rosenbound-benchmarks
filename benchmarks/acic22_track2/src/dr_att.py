"""Doubly-robust ATT with a Crump overlap trim.

This estimator targets the effect on the treated over the region of covariate
overlap. It differs from the plain AIPW estimator in one respect: before
forming the doubly-robust contrast it discards units whose propensity score
falls outside ``[trim_alpha, 1 - trim_alpha]``. Trimming the non-overlapping
tails is what keeps the inverse-propensity weights bounded, so the estimator
stays stable when treated and control covariate distributions only partly
overlap -- the common regime in the ACIC22 Track-2 cohorts.

The trimmed estimand is the effect on the treated units that survive the
overlap restriction. On cohorts with good overlap it coincides with the full
ATT; on cohorts with thin overlap it is the honestly reportable quantity.

References
----------
Hahn, J. (1998). On the role of the propensity score in efficient
semiparametric estimation of average treatment effects. Econometrica,
66(2), 315-331.

Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009).
Dealing with limited overlap in estimation of average treatment effects.
Biometrika, 96(1), 187-199.

Farrell, M. H. (2015). Robust inference on average treatment effects with
possibly more covariates than observations. Journal of Econometrics,
189(1), 1-23.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from ._nuisance import make_classifier, make_regressor

_CLIP = (0.01, 0.99)
_Z95 = 1.959963984540054


@dataclass(frozen=True)
class DrAttResult:
    """Result of a doubly-robust ATT estimate with an overlap trim.

    Attributes
    ----------
    att
        Effect on the treated over the overlap region.
    se
        Standard error from the efficient influence function on the
        trimmed sample.
    ci_lo_95, ci_hi_95
        Bounds of the 95% Wald confidence interval.
    n_trimmed
        Number of units removed by the overlap trim.
    propensity_range
        ``(min, max)`` of the cross-fitted propensity scores before
        trimming.
    """

    att: float
    se: float
    ci_lo_95: float
    ci_hi_95: float
    n_trimmed: int
    propensity_range: tuple[float, float]


def _cross_fit_nuisances(
    X: NDArray[np.float64],
    T: NDArray[np.int_],
    Y: NDArray[np.float64],
    k_folds: int,
    ml_model: str,
    random_state: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Out-of-fold propensity and control-outcome regression (see aipw)."""
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


def dr_att(
    X: NDArray[np.float64],
    T: NDArray[np.int_],
    Y: NDArray[np.float64],
    k_folds: int = 5,
    trim_alpha: float = 0.10,
    ml_model: str = "logit_ridge",
    random_state: int = 0,
) -> DrAttResult:
    """Estimate the effect on the treated with a Crump overlap trim.

    Parameters
    ----------
    X, T, Y
        Covariates ``(n, p)``, binary treatment ``(n,)``, outcome ``(n,)``.
    k_folds
        Cross-fitting folds for the nuisance models.
    trim_alpha
        Crump-2009 overlap threshold; units with propensity outside
        ``[trim_alpha, 1 - trim_alpha]`` are dropped.
    ml_model
        Nuisance learner: ``"logit_ridge"`` (default), ``"hist_gbm"``,
        ``"lgbm"``, or ``"rf"``.
    random_state
        Seed for the fold split and the nuisance learners.

    Returns
    -------
    DrAttResult

    Raises
    ------
    ValueError
        If either arm is smaller than ``k_folds``, or the overlap trim
        leaves too few units in either arm to form the contrast.
    """
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T).astype(np.int_)
    Y = np.asarray(Y, dtype=np.float64)

    n = X.shape[0]
    if (T == 1).sum() < k_folds or (T == 0).sum() < k_folds:
        raise ValueError(
            f"each arm must have >= k_folds units: "
            f"treated={int((T == 1).sum())}, "
            f"control={int((T == 0).sum())}, k_folds={k_folds}"
        )

    e_hat, mu0_hat = _cross_fit_nuisances(
        X, T, Y, k_folds, ml_model, random_state
    )
    prop_range = (float(e_hat.min()), float(e_hat.max()))

    keep = (e_hat >= trim_alpha) & (e_hat <= 1.0 - trim_alpha)
    n_trimmed = int((~keep).sum())

    Tt, Yt = T[keep], Y[keep]
    et, mu0t = e_hat[keep], mu0_hat[keep]
    n_treated = int((Tt == 1).sum())
    if n_treated < 5 or (Tt == 0).sum() < 5:
        raise ValueError(
            f"overlap trim at alpha={trim_alpha} left too few units: "
            f"treated={n_treated}, control={int((Tt == 0).sum())}"
        )

    m = Tt.shape[0]
    p_hat = n_treated / m
    resid = Yt - mu0t
    w = et / (1.0 - et)  # control inverse-propensity weight

    is_ctrl = Tt == 0
    treated_term = float(resid[Tt == 1].mean())
    control_term = float(
        np.dot(w[is_ctrl], resid[is_ctrl]) / w[is_ctrl].sum()
    )
    att = treated_term - control_term

    infl = (
        np.where(Tt == 1, resid, -w * resid) / p_hat
        - (Tt / p_hat) * att
    )
    se = float(infl.std(ddof=1) / np.sqrt(m))

    return DrAttResult(
        att=att,
        se=se,
        ci_lo_95=att - _Z95 * se,
        ci_hi_95=att + _Z95 * se,
        n_trimmed=n_trimmed,
        propensity_range=prop_range,
    )


__all__ = ["DrAttResult", "dr_att"]

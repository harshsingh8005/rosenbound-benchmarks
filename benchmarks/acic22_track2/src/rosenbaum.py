"""Rosenbaum Gamma-sensitivity bound (widening-bound simplification).

A Rosenbaum sensitivity analysis asks how strong an unmeasured confounder
would have to be to overturn an estimated effect. The confounder's strength
is parameterised by Gamma >= 1, the maximum odds ratio by which two units
with identical measured covariates could differ in their odds of treatment.
At Gamma = 1 treatment is as-good-as-random given the covariates; larger
Gamma admits progressively stronger hidden bias.

This module implements the *widening-bound simplification*: rather than
inverting a paired signed-rank statistic, it treats Gamma as inflating the
worst-case bias in the estimate linearly, so the one-sided confidence bound
on the effect widens as

    lower(Gamma) = |effect| - z_alpha * se * Gamma.

``gamma_zero`` is the smallest Gamma at which this bound reaches zero -- the
sensitivity at which a confounder of that strength could explain the effect
away. A larger ``gamma_zero`` means a more robust finding. This normal-
approximation surrogate is monotone in Gamma and closed-form; it is a
deliberate simplification of the exact design-sensitivity calculation and is
reported as such.

Reference
---------
Rosenbaum, P. R. (2002). Observational Studies (2nd ed.), Chapter 4
("Sensitivity to Hidden Bias"). Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

_NOTE = "widening-bound simplification"


@dataclass(frozen=True)
class RosenbaumResult:
    """Result of a widening-bound Rosenbaum sensitivity analysis.

    Attributes
    ----------
    gamma_zero
        Smallest confounder strength Gamma at which the one-sided
        confidence bound on the effect reaches zero. ``1.0`` means the
        effect is already non-significant at Gamma = 1; ``inf`` only in the
        degenerate zero-standard-error case.
    gamma_grid_evaluated
        The Gamma grid over which the widening bound curve was evaluated
        (for plotting); it does not cap ``gamma_zero``.
    lower_bound_grid
        The one-sided lower confidence bound at each grid point, for
        plotting the widening.
    note
        Records that this is the widening-bound simplification.
    """

    gamma_zero: float
    gamma_grid_evaluated: NDArray[np.float64]
    lower_bound_grid: NDArray[np.float64]
    note: str = field(default=_NOTE)


def rosenbaum_gamma_zero(
    estimated_effect: float,
    se_effect: float,
    alpha: float = 0.05,
    gamma_grid: NDArray[np.float64] | None = None,
) -> RosenbaumResult:
    """Compute the widening-bound Gamma at which the effect reaches zero.

    Parameters
    ----------
    estimated_effect
        The point estimate whose robustness is being probed.
    se_effect
        Its standard error.
    alpha
        One-sided significance level for the widening bound (default 0.05,
        i.e. a one-sided 95% bound).
    gamma_grid
        Grid of Gamma values to evaluate. Defaults to
        ``linspace(1.0, 3.0, 41)``.

    Returns
    -------
    RosenbaumResult

    Notes
    -----
    The bound crosses zero at ``Gamma = |effect| / (z_alpha * se)``. When
    the effect is not significant at Gamma = 1 this crossing is below 1 and
    ``gamma_zero`` is clamped to 1.0. The grid only controls the resolution
    of the returned bound curve; it does not cap ``gamma_zero``.
    """
    if gamma_grid is None:
        gamma_grid = np.linspace(1.0, 3.0, 41)
    gamma_grid = np.asarray(gamma_grid, dtype=np.float64)

    z_alpha = float(norm.ppf(1.0 - alpha))
    effect = abs(float(estimated_effect))
    se = float(se_effect)

    lower_bound_grid = effect - z_alpha * se * gamma_grid

    if se <= 0.0:
        # Degenerate SE: the bound never widens; the effect is either
        # exactly zero or infinitely robust.
        gamma_zero = 1.0 if effect == 0.0 else float("inf")
        return RosenbaumResult(
            gamma_zero=gamma_zero,
            gamma_grid_evaluated=gamma_grid,
            lower_bound_grid=lower_bound_grid,
            note=_NOTE,
        )

    crossing = effect / (z_alpha * se)
    gamma_zero = 1.0 if crossing <= 1.0 else float(crossing)

    return RosenbaumResult(
        gamma_zero=gamma_zero,
        gamma_grid_evaluated=gamma_grid,
        lower_bound_grid=lower_bound_grid,
        note=_NOTE,
    )


__all__ = ["RosenbaumResult", "rosenbaum_gamma_zero"]

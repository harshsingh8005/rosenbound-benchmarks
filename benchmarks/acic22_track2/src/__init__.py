"""Public doubly-robust estimators and the ACIC22 Track-2 loader.

This package implements three published, non-proprietary causal-inference
methods and a loader that reduces an ACIC22 Track-2 cohort to the
``(covariates, treatment, outcome)`` form the estimators consume:

- :mod:`.aipw`        — cross-fitted augmented inverse-probability weighting.
- :mod:`.dr_att`      — doubly-robust ATT with an overlap trim.
- :mod:`.rosenbaum`   — Rosenbaum Gamma-sensitivity bound.
- :mod:`.data_loader` — ACIC22 Track-2 cohort reader.
"""

from .aipw import AipwResult, aipw_att
from .dr_att import DrAttResult, dr_att
from .rosenbaum import RosenbaumResult, rosenbaum_gamma_zero
from .data_loader import (
    Track2Cohort,
    enumerate_track2_cohorts,
    load_track2_cohort,
)

__all__ = [
    "AipwResult",
    "aipw_att",
    "DrAttResult",
    "dr_att",
    "RosenbaumResult",
    "rosenbaum_gamma_zero",
    "Track2Cohort",
    "enumerate_track2_cohorts",
    "load_track2_cohort",
]

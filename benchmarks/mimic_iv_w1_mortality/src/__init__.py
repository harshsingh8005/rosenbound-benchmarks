"""Public building blocks for the MIMIC-IV W1 in-hospital mortality reproducer.

The reproducer predicts in-hospital mortality for ICU admissions in MIMIC-IV
using only published, non-proprietary methods:

- :mod:`.loader`      — assemble the first-24h ICU-admission cohort from the
                        MIMIC-IV hospital and ICU tables.
- :mod:`.features`    — turn the raw cohort frame into a typed model matrix.
- :mod:`.model`       — a gradient-boosted-trees classifier (LightGBM).
- :mod:`.calibration` — standalone Beta calibration with a Bayesian
                        (L2-MAP) smoothing prior.
- :mod:`.evaluate`    — discrimination and calibration metrics.
"""

from .loader import W1Cohort, load_w1_cohort, resolve_demo_root
from .features import FEATURE_SPEC, build_features
from .model import fit_predict_proba
from .calibration import BetaBayesCalibrator
from .evaluate import evaluate_predictions, reliability_curve

__all__ = [
    "W1Cohort",
    "load_w1_cohort",
    "resolve_demo_root",
    "FEATURE_SPEC",
    "build_features",
    "fit_predict_proba",
    "BetaBayesCalibrator",
    "evaluate_predictions",
    "reliability_curve",
]

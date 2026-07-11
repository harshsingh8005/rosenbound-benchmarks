"""FAERS adverse-event severity reproducer (public-method baseline)."""

from .evaluate import evaluate_predictions, expected_calibration_error
from .features import build_features_apply, build_features_fit
from .loader import FaersCohort, LABEL_COL, YEAR_COL, build_cohort, load_faers_cohort
from .model import fit_predict_proba

__all__ = [
    "FaersCohort",
    "LABEL_COL",
    "YEAR_COL",
    "build_cohort",
    "load_faers_cohort",
    "build_features_fit",
    "build_features_apply",
    "fit_predict_proba",
    "evaluate_predictions",
    "expected_calibration_error",
]

"""Discrimination and calibration metrics for the FAERS reproducer.

Discrimination is the area under the ROC curve (AUROC) and average precision
(AUPRC); calibration is the Brier score and the expected calibration error
(ECE, the standard equal-width 10-bin occupancy-weighted estimate; Naeini et
al., 2015). The severity classes are imbalanced, so AUPRC is reported alongside
AUROC as the more sensitive discrimination measure.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DEFAULT_ECE_BINS = 10


def expected_calibration_error(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = DEFAULT_ECE_BINS
) -> float:
    """Equal-width binned ECE over ``[0, 1]`` (occupancy-weighted)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    n = len(prob)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - prob[mask].mean())
    return float(ece)


def evaluate_predictions(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = DEFAULT_ECE_BINS
) -> dict[str, float]:
    """Compute AUROC, AUPRC (average precision), Brier score, and ECE.

    ``y_true`` must contain both classes; AUROC and AUPRC are undefined on a
    single-class ground truth and this raises there.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        raise ValueError("evaluate_predictions requires both classes in y_true")
    return {
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "ece": expected_calibration_error(y_true, prob, n_bins=n_bins),
    }


__all__ = ["DEFAULT_ECE_BINS", "evaluate_predictions", "expected_calibration_error"]

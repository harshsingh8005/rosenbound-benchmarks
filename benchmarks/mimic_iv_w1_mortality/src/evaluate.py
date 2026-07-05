"""Discrimination and calibration metrics for the W1 reproducer.

Discrimination is the area under the ROC curve (AUROC) and average precision;
calibration is the Brier score and the expected calibration error (ECE). ECE
is the standard equal-width 10-bin estimate: the average over bins of the gap
between mean predicted probability and observed positive rate, weighted by bin
occupancy (Naeini et al., 2015; Guo et al., 2017).
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
    # Right-closed bins with the top edge inclusive so p == 1.0 lands in-range.
    idx = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    n = len(prob)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_curve(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = DEFAULT_ECE_BINS
) -> dict[str, list[float]]:
    """Per-bin reliability data for plotting and for the results contract.

    Returns a dict of equal-length lists over occupied bins: ``bin_mid`` (bin
    centre), ``mean_pred`` (mean predicted probability), ``frac_pos`` (observed
    positive rate), and ``count`` (number of samples in the bin).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, n_bins - 1)
    mids, mean_pred, frac_pos, count = [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        mids.append(float((edges[b] + edges[b + 1]) / 2))
        mean_pred.append(float(prob[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
        count.append(int(mask.sum()))
    return {
        "bin_mid": mids,
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
        "count": count,
    }


def evaluate_predictions(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = DEFAULT_ECE_BINS
) -> dict[str, float]:
    """Compute AUROC, average precision, Brier score, and ECE.

    ``y_true`` must contain both classes; AUROC and average precision are
    undefined on a single-class ground truth and this raises there.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        raise ValueError("evaluate_predictions requires both classes in y_true")
    return {
        "auroc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "ece": expected_calibration_error(y_true, prob, n_bins=n_bins),
    }


__all__ = [
    "DEFAULT_ECE_BINS",
    "evaluate_predictions",
    "expected_calibration_error",
    "reliability_curve",
]

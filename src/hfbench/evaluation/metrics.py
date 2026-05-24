"""Core evaluation metrics for HF-Bench."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from hfbench.evaluation.calibration import expected_calibration_error

logger = logging.getLogger(__name__)


def threshold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Sensitivity, specificity, PPV, NPV at a fixed probability threshold."""
    y_bin = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "threshold": threshold,
    }


def find_threshold_at_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_specificity: float = 0.80,
) -> float:
    """Find the lowest threshold that achieves at least *target_specificity*."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    specificities = 1.0 - fpr
    # Find thresholds where specificity >= target
    mask = specificities >= target_specificity
    if not mask.any():
        logger.warning(
            "No threshold achieves target specificity %.2f; "
            "falling back to most conservative threshold.",
            target_specificity,
        )
        # roc_curve thresholds are descending; [0] is the most conservative
        return float(thresholds[0])
    # Among those, return the one with highest sensitivity (lowest threshold)
    idx = np.where(mask)[0]
    return float(thresholds[idx[np.argmax(tpr[idx])]])


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_fixed: float = 0.5,
    target_specificity: float = 0.80,
    ece_n_bins: int = 10,
) -> Dict[str, float]:
    """Compute the full HF-Bench metric suite.

    Parameters
    ----------
    y_true: binary ground-truth labels.
    y_pred: predicted probabilities in [0,1].
    threshold_fixed: fixed threshold for threshold metrics.
    target_specificity: operating point for second threshold set.
    ece_n_bins: number of equal-width bins for ECE.

    Returns
    -------
    Flat dict of metric name -> value.
    """
    metrics: Dict[str, float] = {}

    if len(np.unique(y_true)) < 2:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, y_pred))
        metrics["auprc"] = float(average_precision_score(y_true, y_pred))

    metrics["brier_score"] = float(brier_score_loss(y_true, y_pred))
    metrics["ece"] = float(expected_calibration_error(y_true, y_pred, n_bins=ece_n_bins))
    metrics["prevalence"] = float(y_true.mean())
    metrics["n"] = int(len(y_true))

    # Threshold @ 0.5
    t05 = threshold_metrics(y_true, y_pred, threshold=threshold_fixed)
    for k, v in t05.items():
        metrics[f"t0.5_{k}"] = v

    # Threshold @ target specificity
    thr_sp = find_threshold_at_specificity(y_true, y_pred, target_specificity)
    tsp = threshold_metrics(y_true, y_pred, threshold=thr_sp)
    for k, v in tsp.items():
        metrics[f"t{int(target_specificity*100)}sp_{k}"] = v

    return metrics

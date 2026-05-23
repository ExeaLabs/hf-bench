"""Calibration metrics for HF-Bench."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Uses equal-width bins from 0 to 1.

    Parameters
    ----------
    y_true: binary labels.
    y_pred: predicted probabilities.
    n_bins: number of equal-width bins.

    Returns
    -------
    ECE in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        # Include right edge in last bin
        if i == n_bins - 1:
            mask = (y_pred >= bins[i]) & (y_pred <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_frac = mask.sum() / n
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        ece += bin_frac * abs(bin_acc - bin_conf)

    return float(ece)

"""Tests for evaluation metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from hfbench.evaluation.calibration import expected_calibration_error
from hfbench.evaluation.metrics import compute_all_metrics, threshold_metrics


def test_ece_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.uniform(0, 1, size=200)
    ece = expected_calibration_error(y_true, y_pred)
    assert 0.0 <= ece <= 1.0


def test_ece_perfect_calibration():
    # If predictions == labels, ECE should be near 0
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.0, 0.0, 1.0, 1.0])
    ece = expected_calibration_error(y_true, y_pred)
    assert ece < 0.01


def test_ece_worst_calibration():
    # Completely wrong calibration: predict 0 when true=1 and vice versa
    y_true = np.array([1] * 50 + [0] * 50)
    y_pred = np.array([0.0] * 50 + [1.0] * 50)
    ece = expected_calibration_error(y_true, y_pred)
    assert ece > 0.5


def test_threshold_metrics_shape():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=100)
    y_pred = rng.uniform(0, 1, size=100)
    result = threshold_metrics(y_true, y_pred, 0.5)
    for key in ["sensitivity", "specificity", "ppv", "npv"]:
        assert key in result
        assert 0.0 <= result[key] <= 1.0


def test_compute_all_metrics_keys():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=150)
    y_pred = rng.uniform(0, 1, size=150)
    metrics = compute_all_metrics(y_true, y_pred)
    for key in ["auroc", "auprc", "brier_score", "ece", "prevalence"]:
        assert key in metrics
        assert not np.isnan(metrics[key])

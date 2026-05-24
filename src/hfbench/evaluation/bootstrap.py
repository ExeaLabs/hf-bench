"""Bootstrap CI and multi-seed aggregation utilities."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

NON_METRIC_COLS = {"seed", "n", "threshold", "t0.5_threshold", "t80sp_threshold"}


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute a metric with 95% bootstrap CI (percentile method).

    Returns
    -------
    Dict with keys 'mean', 'ci_lower', 'ci_upper'.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            v = metric_fn(y_true[idx], y_pred[idx])
            boot_vals.append(v)
        except Exception:
            continue
    boot_arr = np.array(boot_vals)
    return {
        "mean": float(point),
        "ci_lower": float(np.percentile(boot_arr, 2.5)),
        "ci_upper": float(np.percentile(boot_arr, 97.5)),
    }


def aggregate_seed_results(records: List[Dict]) -> pd.DataFrame:
    """Aggregate metric dicts across seeds using mean +/- 95% CI.

    Parameters
    ----------
    records: list of metric dicts, one per seed run.

    Returns
    -------
    DataFrame with columns: metric, mean, ci_lower, ci_upper, std, n_seeds.
    """
    df = pd.DataFrame(records)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    rows = []
    for col in numeric_cols:
        if col in NON_METRIC_COLS:
            continue
        vals = df[col].dropna().values
        n = len(vals)
        if n == 0:
            continue
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else 0.0
        half_ci = 1.96 * std / np.sqrt(n)
        rows.append({
            "metric": col,
            "mean": mean,
            "ci_lower": mean - half_ci,
            "ci_upper": mean + half_ci,
            "std": std,
            "n_seeds": n,
        })

    return pd.DataFrame(rows)

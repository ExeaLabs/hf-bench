"""Subgroup fairness metrics for HF-Bench."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from hfbench.evaluation.calibration import expected_calibration_error
from hfbench.evaluation.metrics import threshold_metrics

logger = logging.getLogger(__name__)


def _safe_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_pred))


def _safe_auprc(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_pred))


def subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
    ece_n_bins: int = 10,
) -> pd.DataFrame:
    """Compute per-group metrics for a single subgroup column.

    Returns a DataFrame with one row per group.
    """
    rows = []
    unique_groups = np.unique(groups[~pd.isnull(groups)])

    for g in unique_groups:
        mask = groups == g
        yt = y_true[mask]
        yp = y_pred[mask]
        row: Dict = {"group": g, "n": int(mask.sum()), "prevalence": float(yt.mean())}
        row["auroc"] = _safe_auroc(yt, yp)
        row["auprc"] = _safe_auprc(yt, yp)
        row["brier"] = float(brier_score_loss(yt, yp))
        row["ece"] = float(expected_calibration_error(yt, yp, n_bins=ece_n_bins))
        tm = threshold_metrics(yt, yp, threshold)
        row["tpr"] = tm["sensitivity"]
        row["fpr"] = 1.0 - tm["specificity"]
        rows.append(row)

    return pd.DataFrame(rows)


def compute_fairness_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_df: pd.DataFrame,
) -> Dict:
    """Compute aggregate fairness metrics from per-group table.

    Parameters
    ----------
    y_true, y_pred: overall labels and predictions.
    subgroup_df: output of subgroup_metrics().

    Returns
    -------
    Dict with worst_subgroup_auroc, demographic_parity_gap,
    equalized_odds_tpr_gap, equalized_odds_fpr_gap.
    """
    aurocs = subgroup_df["auroc"].dropna().values
    tprs = subgroup_df["tpr"].values
    fprs = subgroup_df["fpr"].values
    prev = subgroup_df["prevalence"].values

    return {
        "worst_subgroup_auroc": float(aurocs.min()) if len(aurocs) > 0 else float("nan"),
        "mean_subgroup_auroc": float(aurocs.mean()) if len(aurocs) > 0 else float("nan"),
        "demographic_parity_gap": float(prev.max() - prev.min()),
        "equalized_odds_tpr_gap": float(tprs.max() - tprs.min()),
        "equalized_odds_fpr_gap": float(fprs.max() - fprs.min()),
    }


def run_fairness_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df_meta: pd.DataFrame,
    subgroup_cols: List[str],
    threshold: float = 0.5,
    ece_n_bins: int = 10,
) -> Dict:
    """Run full fairness evaluation across all subgroup columns.

    Parameters
    ----------
    df_meta: DataFrame with subgroup columns aligned with y_true/y_pred.

    Returns
    -------
    Dict mapping subgroup_col -> {per_group_df, summary_dict}.
    """
    results = {}
    for col in subgroup_cols:
        if col not in df_meta.columns:
            logger.warning("Subgroup column '%s' not found, skipping.", col)
            continue
        groups = df_meta[col].values
        sg_df = subgroup_metrics(y_true, y_pred, groups, threshold=threshold, ece_n_bins=ece_n_bins)
        summary = compute_fairness_summary(y_true, y_pred, sg_df)
        results[col] = {"per_group": sg_df, "summary": summary}
    return results

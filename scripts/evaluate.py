#!/usr/bin/env python
"""Evaluate saved prediction files and write metric JSONs.

Usage:
    python scripts/evaluate.py [--predictions-dir results/predictions]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from hfbench.constants import SUBGROUP_COLS
from hfbench.evaluation.fairness import run_fairness_evaluation
from hfbench.evaluation.metrics import compute_all_metrics
from hfbench.utils.io import load_predictions, save_json
from hfbench.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate HF-Bench prediction files.")
    p.add_argument("--predictions-dir", type=Path, default=Path("results/predictions"))
    p.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--target-specificity", type=float, default=0.80)
    p.add_argument("--ece-bins", type=int, default=10)
    return p.parse_args()


def evaluate_prediction_file(
    pred_path: Path,
    metrics_dir: Path,
    threshold: float,
    target_specificity: float,
    ece_bins: int,
) -> None:
    df = load_predictions(pred_path)
    y_true = df["y_true"].values.astype(int)
    y_pred = df["y_pred"].values.astype(float)

    metrics = compute_all_metrics(
        y_true, y_pred,
        threshold_fixed=threshold,
        target_specificity=target_specificity,
        ece_n_bins=ece_bins,
    )

    # Subgroup fairness
    subgroup_cols = [c for c in SUBGROUP_COLS + ["age_group"] if c in df.columns]
    fairness = run_fairness_evaluation(
        y_true, y_pred, df, subgroup_cols,
        threshold=threshold, ece_n_bins=ece_bins,
    )

    # Flatten fairness summary into metrics
    for col, result in fairness.items():
        for k, v in result["summary"].items():
            metrics[f"fairness_{col}_{k}"] = v

    stem = pred_path.stem  # e.g. "xgboost_seed42"
    out_path = metrics_dir / f"{stem}_metrics.json"
    save_json(metrics, out_path)
    logger.info("Metrics for %s: AUROC=%.4f  [SYNTHETIC DATA]", stem, metrics.get("auroc", float("nan")))

    # Save per-group fairness tables
    for col, result in fairness.items():
        sg_path = metrics_dir / f"{stem}_fairness_{col}.csv"
        result["per_group"].to_csv(sg_path, index=False)


def main():
    args = parse_args()
    setup_logging()
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(args.predictions_dir.glob("*.parquet"))
    if not pred_files:
        logger.warning("No .parquet prediction files found in %s", args.predictions_dir)
        return

    for pred_path in pred_files:
        try:
            evaluate_prediction_file(
                pred_path, args.metrics_dir,
                args.threshold, args.target_specificity, args.ece_bins,
            )
        except Exception as exc:
            logger.error("Error evaluating %s: %s", pred_path.name, exc, exc_info=True)

    logger.info("Evaluation complete. Metrics saved to %s", args.metrics_dir)


if __name__ == "__main__":
    main()

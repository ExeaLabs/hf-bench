#!/usr/bin/env python
"""Aggregate metrics across seeds and write summary table.

Usage:
    python scripts/aggregate_results.py [--metrics-dir results/metrics]
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from hfbench.evaluation.bootstrap import aggregate_seed_results
from hfbench.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate HF-Bench metrics across seeds.")
    p.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    p.add_argument("--output", type=Path, default=None, help="Output CSV (default: summary in metrics-dir)")
    return p.parse_args()


def load_metric_files(metrics_dir: Path):
    """Load all *_metrics.json files and return grouped by model."""
    pattern = re.compile(r"^(.+)_seed(\d+)_metrics\.json$")
    model_records = {}

    for f in sorted(metrics_dir.glob("*_metrics.json")):
        m = pattern.match(f.name)
        if not m:
            # Try quick metrics
            continue
        model_name = m.group(1)
        seed = int(m.group(2))
        with open(f) as fh:
            rec = json.load(fh)
        rec["seed"] = seed
        model_records.setdefault(model_name, []).append(rec)

    return model_records


def main():
    args = parse_args()
    setup_logging()

    model_records = load_metric_files(args.metrics_dir)

    if not model_records:
        logger.warning("No *_seed*_metrics.json files found in %s", args.metrics_dir)
        return

    all_rows = []
    for model_name, records in model_records.items():
        agg = aggregate_seed_results(records)
        agg.insert(0, "model", model_name)
        all_rows.append(agg)

    summary = pd.concat(all_rows, ignore_index=True)

    out_path = args.output or (args.metrics_dir / "summary_mean_ci.csv")
    summary.to_csv(out_path, index=False)
    logger.info("Summary saved to %s [SYNTHETIC DATA]", out_path)

    # Pretty print key metrics
    key_metrics = ["auroc", "auprc", "brier_score", "ece"]
    try:
        pivot = summary[summary["metric"].isin(key_metrics)].pivot(
            index="model", columns="metric", values="mean"
        )
        logger.info("\n=== Mean metrics across seeds (SYNTHETIC DATA) ===\n%s", pivot.to_string())
    except Exception:
        logger.info("Full summary:\n%s", summary.to_string())


if __name__ == "__main__":
    main()

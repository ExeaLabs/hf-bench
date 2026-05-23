#!/usr/bin/env python
"""Generate synthetic HF readmission demo dataset.

Usage:
    python scripts/make_demo_data.py [--n 5000] [--seed 42] [--output data/raw/demo_hf.csv]
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hfbench.data.demo import make_demo_dataset
from hfbench.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic HF demo dataset.")
    p.add_argument("--n", type=int, default=5000, help="Number of rows")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/demo_hf.csv"),
        help="Output CSV path",
    )
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger.info("Generating synthetic demo dataset (n=%d, seed=%d)...", args.n, args.seed)

    df = make_demo_dataset(n=args.n, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    pos_rate = df["readmit_30d"].mean()
    logger.info(
        "Saved %d rows to %s (readmit rate=%.1f%%) [SYNTHETIC DATA — NOT REAL PATIENTS]",
        len(df), args.output, 100 * pos_rate,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Preprocess raw HF cohort data: split and fit preprocessing pipeline.

Usage:
    python scripts/preprocess.py [--data-path data/raw/demo_hf.csv]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from hfbench.constants import LABEL_COL, SUBGROUP_COLS, TIME_COL
from hfbench.data.preprocess import fit_preprocessor, save_preprocessor, transform
from hfbench.data.split import save_split_indices, temporal_split
from hfbench.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess HF cohort data.")
    p.add_argument(
        "--data-path", type=Path, default=Path("data/raw/demo_hf.csv"),
        help="Path to raw cohort CSV",
    )
    p.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--no-smote", action="store_true", help="Skip SMOTE")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()

    logger.info("Loading data from %s", args.data_path)
    df = pd.read_csv(args.data_path, parse_dates=["admittime", "dischtime"])
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])

    # Derive age group for subgroup analysis
    if "age" in df.columns and "age_group" not in df.columns:
        df["age_group"] = (df["age"] >= 65).map({True: ">=65", False: "<65"})

    train, val, test = temporal_split(df)

    # Save split indices
    args.interim_dir.mkdir(parents=True, exist_ok=True)
    save_split_indices(train, val, test, args.interim_dir)

    # Fit preprocessor on training data only
    preprocessor = fit_preprocessor(train)
    save_preprocessor(preprocessor, args.interim_dir / "preprocessor.pkl")

    # Save processed splits as parquet
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        X = transform(preprocessor, split_df)
        y = split_df[LABEL_COL].values

        # Save feature matrix
        import numpy as np
        np.save(args.processed_dir / f"X_{split_name}.npy", X)
        np.save(args.processed_dir / f"y_{split_name}.npy", y)

        # Save metadata (ids + subgroup cols) for evaluation
        meta_cols = ["subject_id", "hadm_id", LABEL_COL] + [
            c for c in SUBGROUP_COLS + ["age_group"] if c in split_df.columns
        ]
        split_df[meta_cols].to_parquet(args.processed_dir / f"meta_{split_name}.parquet", index=False)
        logger.info("Saved %s split: %d rows, %d features", split_name, len(split_df), X.shape[1])

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()

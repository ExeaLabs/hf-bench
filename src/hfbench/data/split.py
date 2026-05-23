"""Temporal train/validation/test split for HF-Bench."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from hfbench.constants import TIME_COL, TRAIN_FRAC, VAL_FRAC

logger = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/val/test by time order.

    Sorts by *time_col* and takes the first *train_frac* as train,
    next *val_frac* as val, remainder as test. No data leakage.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df_sorted.iloc[:n_train].copy()
    val = df_sorted.iloc[n_train : n_train + n_val].copy()
    test = df_sorted.iloc[n_train + n_val :].copy()

    logger.info(
        "Temporal split: train=%d (%.1f%%) val=%d (%.1f%%) test=%d (%.1f%%)",
        len(train), 100 * len(train) / n,
        len(val), 100 * len(val) / n,
        len(test), 100 * len(test) / n,
    )
    return train, val, test


def save_split_indices(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
    id_col: str = "hadm_id",
) -> None:
    """Persist hadm_id lists so splits can be reproduced from raw data."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df[[id_col]].to_parquet(out_dir / f"{name}_ids.parquet", index=False)
    logger.info("Split indices saved to %s", out_dir)


def load_split_from_indices(
    df: pd.DataFrame,
    interim_dir: Path,
    id_col: str = "hadm_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reload splits using saved hadm_id index files."""
    interim_dir = Path(interim_dir)
    splits = {}
    for name in ("train", "val", "test"):
        ids = pd.read_parquet(interim_dir / f"{name}_ids.parquet")[id_col]
        splits[name] = df[df[id_col].isin(ids)].copy()
    return splits["train"], splits["val"], splits["test"]

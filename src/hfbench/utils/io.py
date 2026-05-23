"""I/O helpers: parquet, JSON, CSV."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    logger.debug("Saved JSON to %s", path)


def load_json(path: Path) -> Dict[str, Any]:
    with open(Path(path)) as f:
        return json.load(f)


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df_meta: pd.DataFrame,
    path: Path,
) -> None:
    """Save predictions as parquet with metadata columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df_meta.copy().reset_index(drop=True)
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out.to_parquet(path, index=False)
    logger.debug("Saved predictions to %s", path)


def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))

"""Final model training after hyperparameter tuning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import optuna

from hfbench.models.base import BaseModel

logger = logging.getLogger(__name__)


def train_best_model(
    model_cls,
    best_params: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    seed: int = 42,
) -> BaseModel:
    """Instantiate model with best params and train on train+val."""
    logger.info("Training final model with params: %s", best_params)
    model: BaseModel = model_cls(seed=seed, **best_params)

    # For deep learning models use val for early stopping; for others combine
    if X_val is not None and hasattr(model, "patience"):
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    elif X_val is not None:
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train)

    return model

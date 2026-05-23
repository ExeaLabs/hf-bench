"""Abstract base class for all HF-Bench models."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional

import numpy as np


class BaseModel(abc.ABC):
    """All model wrappers must implement this interface."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._model = None

    @abc.abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        """Fit model; return self."""

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities, shape (n,)."""

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Persist model to *path*."""

    @abc.abstractmethod
    def load(self, path: Path) -> "BaseModel":
        """Load model from *path*; return self."""

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

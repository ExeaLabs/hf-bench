"""Logistic Regression model wrapper."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from hfbench.models.base import BaseModel


class LogisticRegressionModel(BaseModel):
    """Scikit-learn Logistic Regression with class balancing."""

    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(seed=seed)
        self._init_kwargs = kwargs

    def _build(self, **kwargs) -> LogisticRegression:
        penalty = kwargs.get("penalty", "l2")
        C = kwargs.get("C", 1.0)

        solver = "saga" if penalty == "l1" else "lbfgs"
        return LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            class_weight="balanced",
            max_iter=2000,
            random_state=self.seed,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "LogisticRegressionModel":
        params = {**self._init_kwargs, **kwargs}
        self._model = self._build(**params)
        self._model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: Path) -> "LogisticRegressionModel":
        with open(Path(path), "rb") as f:
            self._model = pickle.load(f)
        return self

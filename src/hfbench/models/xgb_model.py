"""XGBoost model wrapper."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from hfbench.models.base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost classifier wrapper."""

    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(seed=seed)
        self._init_kwargs = kwargs

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "XGBoostModel":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is required. Install with: pip install xgboost") from exc

        params = {**self._init_kwargs, **kwargs}
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        self._model = XGBClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            min_child_weight=params.get("min_child_weight", 5),
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=self.seed,
            n_jobs=-1,
        )

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: Path) -> "XGBoostModel":
        with open(Path(path), "rb") as f:
            self._model = pickle.load(f)
        return self

"""TabNet model wrapper with graceful fallback if pytorch-tabnet is missing."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from hfbench.models.base import BaseModel

logger = logging.getLogger(__name__)


def _check_tabnet():
    try:
        import pytorch_tabnet  # noqa: F401
        return True
    except ImportError:
        return False


class TabNetModel(BaseModel):
    """TabNet classifier wrapper using pytorch-tabnet."""

    def __init__(
        self,
        seed: int = 42,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 5,
        gamma: float = 1.3,
        lambda_sparse: float = 1e-4,
        lr: float = 2e-2,
        batch_size: int = 16384,
        max_epochs: int = 200,
        patience: int = 15,
        **kwargs,
    ):
        super().__init__(seed=seed)
        if not _check_tabnet():
            raise ImportError(
                "pytorch-tabnet is required for TabNet. "
                "Install with: pip install pytorch-tabnet"
            )
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "TabNetModel":
        from pytorch_tabnet.tab_model import TabNetClassifier

        self._model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            optimizer_params={"lr": self.lr},
            seed=self.seed,
            verbose=0,
        )

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._model.fit(
            X_train.astype(np.float32),
            y_train,
            eval_set=eval_set,
            eval_metric=["auc"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            drop_last=False,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X.astype(np.float32))[:, 1]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # TabNet has its own save mechanism
        self._model.save_model(str(path).replace(".pkl", ""))

    def load(self, path: Path) -> "TabNetModel":
        from pytorch_tabnet.tab_model import TabNetClassifier

        self._model = TabNetClassifier()
        self._model.load_model(str(Path(path)).replace(".pkl", "") + ".zip")
        return self

"""PyTorch MLP model for binary classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from hfbench.models.base import BaseModel

logger = logging.getLogger(__name__)


class _MLPNet:
    """Internal PyTorch MLP — imported lazily so torch is not required at import time."""

    def __new__(cls, *args, **kwargs):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, in_features: int, hidden_dim: int, num_layers: int, dropout: float):
                super().__init__()
                layers = []
                dim_in = in_features
                for _ in range(num_layers):
                    layers += [
                        nn.Linear(dim_in, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                    dim_in = hidden_dim
                layers.append(nn.Linear(hidden_dim, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x).squeeze(-1)

        return Net(*args, **kwargs)


class MLPModel(BaseModel):
    """PyTorch MLP with BatchNorm, Dropout, early stopping."""

    def __init__(
        self,
        seed: int = 42,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 16384,
        max_epochs: int = 200,
        patience: int = 15,
        **kwargs,
    ):
        super().__init__(seed=seed)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self._net = None
        self._in_features = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "MLPModel":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import roc_auc_score

        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._in_features = X_train.shape[1]
        net = _MLPNet(self._in_features, self.hidden_dim, self.num_layers, self.dropout)
        net = net.to(device)

        # Compute pos_weight for class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        X_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
        y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        best_val_auc = -1.0
        best_state = None
        wait = 0

        for epoch in range(self.max_epochs):
            net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()

            if X_val is not None:
                net.eval()
                with torch.no_grad():
                    Xv = torch.from_numpy(X_val.astype(np.float32)).to(device)
                    logits = net(Xv).cpu().numpy()
                    probs = 1.0 / (1.0 + np.exp(-logits))
                try:
                    val_auc = roc_auc_score(y_val, probs)
                except Exception:
                    val_auc = 0.5

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        logger.info("MLP early stop at epoch %d, best val AUC=%.4f", epoch, best_val_auc)
                        break

        if best_state is not None:
            net.load_state_dict(best_state)

        self._net = net.cpu()
        self._device = "cpu"
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        self._net.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(X.astype(np.float32))
            logits = self._net(Xt).numpy()
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, path: Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._net.state_dict(),
            "in_features": self._in_features,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "seed": self.seed,
        }, path)

    def load(self, path: Path) -> "MLPModel":
        import torch

        ckpt = torch.load(Path(path), map_location="cpu")
        self._in_features = ckpt["in_features"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.num_layers = ckpt["num_layers"]
        self.dropout = ckpt["dropout"]
        net = _MLPNet(self._in_features, self.hidden_dim, self.num_layers, self.dropout)
        net.load_state_dict(ckpt["state_dict"])
        self._net = net
        return self

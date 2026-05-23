"""FT-Transformer: Feature Tokenizer + Transformer for tabular data.

Compact PyTorch implementation treating each feature as a token via a
learned linear projection, then encoding with standard TransformerEncoder
layers before pooling a CLS token to binary output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from hfbench.models.base import BaseModel

logger = logging.getLogger(__name__)


class _FTTransformerNet:
    """Lazy factory — returns a torch.nn.Module instance."""

    def __new__(
        cls,
        in_features: int,
        d_token: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        import torch
        import torch.nn as nn
        import math

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # Project each feature to d_token; CLS token
                self.feat_proj = nn.Linear(in_features, d_token)
                self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
                nn.init.trunc_normal_(self.cls_token, std=0.02)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_token,
                    nhead=n_heads,
                    dim_feedforward=d_token * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.head = nn.Linear(d_token, 1)

            def forward(self, x):
                # x: (B, F)  ->  (B, 1, d_token)  via linear projection
                tokens = self.feat_proj(x).unsqueeze(1)  # (B, 1, d)
                cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d)
                seq = torch.cat([cls, tokens], dim=1)           # (B, 2, d)
                out = self.transformer(seq)                      # (B, 2, d)
                return self.head(out[:, 0, :]).squeeze(-1)       # (B,)

        return Net()


class FTTransformerModel(BaseModel):
    """FT-Transformer for binary tabular classification."""

    def __init__(
        self,
        seed: int = 42,
        d_token: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 15,
        **kwargs,
    ):
        super().__init__(seed=seed)
        self.d_token = d_token
        self.n_layers = n_layers
        self.n_heads = n_heads
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
    ) -> "FTTransformerModel":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import roc_auc_score

        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._in_features = X_train.shape[1]

        # n_heads must divide d_token
        n_heads = self.n_heads
        while self.d_token % n_heads != 0 and n_heads > 1:
            n_heads //= 2

        net = _FTTransformerNet(
            in_features=self._in_features,
            d_token=self.d_token,
            n_layers=self.n_layers,
            n_heads=n_heads,
            dropout=self.dropout,
        ).to(device)

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        X_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
        y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        best_val_auc = -1.0
        best_state = None
        wait = 0

        for epoch in range(self.max_epochs):
            net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(net(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

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
                        logger.info(
                            "FT-Transformer early stop epoch %d, best val AUC=%.4f",
                            epoch, best_val_auc,
                        )
                        break

        if best_state is not None:
            net.load_state_dict(best_state)

        self._net = net.cpu()
        self._n_heads_used = n_heads
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
            "d_token": self.d_token,
            "n_layers": self.n_layers,
            "n_heads": getattr(self, "_n_heads_used", self.n_heads),
            "dropout": self.dropout,
            "seed": self.seed,
        }, path)

    def load(self, path: Path) -> "FTTransformerModel":
        import torch

        ckpt = torch.load(Path(path), map_location="cpu")
        self._in_features = ckpt["in_features"]
        self.d_token = ckpt["d_token"]
        self.n_layers = ckpt["n_layers"]
        self.dropout = ckpt["dropout"]
        n_heads = ckpt["n_heads"]
        net = _FTTransformerNet(
            in_features=self._in_features,
            d_token=self.d_token,
            n_layers=self.n_layers,
            n_heads=n_heads,
            dropout=self.dropout,
        )
        net.load_state_dict(ckpt["state_dict"])
        self._net = net
        return self

"""Preprocessing pipeline: imputation, scaling, encoding, SMOTE."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hfbench.constants import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    LABEL_COL,
    NUMERIC_FEATURES,
)

logger = logging.getLogger(__name__)


def _numeric_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])


def _categorical_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer for the feature set."""
    return ColumnTransformer(
        transformers=[
            ("num", _numeric_pipeline(), numeric_cols),
            ("cat", _categorical_pipeline(), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def fit_preprocessor(
    train_df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Fit preprocessor on training data only."""
    if numeric_cols is None:
        numeric_cols = [c for c in NUMERIC_FEATURES if c in train_df.columns]
    if categorical_cols is None:
        categorical_cols = [c for c in CATEGORICAL_FEATURES if c in train_df.columns]

    # Binary features treated as numeric (already 0/1, impute median)
    binary_present = [c for c in BINARY_FEATURES if c in train_df.columns]
    all_numeric = numeric_cols + binary_present

    preprocessor = build_preprocessor(all_numeric, categorical_cols)
    preprocessor.fit(train_df)
    logger.info(
        "Preprocessor fit on %d rows; %d numeric, %d categorical features.",
        len(train_df), len(all_numeric), len(categorical_cols),
    )
    return preprocessor


def transform(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
) -> np.ndarray:
    """Apply a fitted preprocessor; returns dense numpy array."""
    return preprocessor.transform(df)


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to (X_train, y_train) only."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. "
            "Install it with: pip install imbalanced-learn"
        ) from exc

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(
        "SMOTE: %d -> %d samples (positive rate %.2f%% -> %.2f%%)",
        len(y), len(y_res),
        100 * y.mean(), 100 * y_res.mean(),
    )
    return X_res, y_res


def save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("Preprocessor saved to %s", path)


def load_preprocessor(path: Path) -> ColumnTransformer:
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def prepare_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = LABEL_COL,
    smote: bool = True,
    smote_seed: int = 42,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple:
    """Full preprocessing workflow: fit on train, transform all splits.

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test, preprocessor)
    """
    preprocessor = fit_preprocessor(train_df, numeric_cols, categorical_cols)

    X_train = transform(preprocessor, train_df)
    y_train = train_df[label_col].values.astype(int)

    X_val = transform(preprocessor, val_df)
    y_val = val_df[label_col].values.astype(int)

    X_test = transform(preprocessor, test_df)
    y_test = test_df[label_col].values.astype(int)

    if smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state=smote_seed)

    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor

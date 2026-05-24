"""End-to-end demo pipeline test (tiny n=200, no Optuna tuning)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from hfbench.constants import LABEL_COL
from hfbench.data.demo import make_demo_dataset
from hfbench.data.preprocess import apply_smote, fit_preprocessor, transform
from hfbench.data.split import temporal_split
from hfbench.evaluation.metrics import compute_all_metrics
from hfbench.models.sklearn_models import LogisticRegressionModel
from hfbench.models.xgb_model import XGBoostModel
from hfbench.models.lgbm_model import LightGBMModel


@pytest.fixture(scope="module")
def tiny_data():
    df = make_demo_dataset(n=200, seed=0)
    if "age_group" not in df.columns:
        df["age_group"] = (df["age"] >= 65).map({True: ">=65", False: "<65"})
    return df


@pytest.fixture(scope="module")
def tiny_splits(tiny_data):
    train, val, test = temporal_split(tiny_data)
    preprocessor = fit_preprocessor(train)
    X_train = transform(preprocessor, train)
    y_train = train[LABEL_COL].values.astype(int)
    X_val = transform(preprocessor, val)
    y_val = val[LABEL_COL].values.astype(int)
    X_test = transform(preprocessor, test)
    y_test = test[LABEL_COL].values.astype(int)
    return X_train, y_train, X_val, y_val, X_test, y_test


def test_demo_data_shape(tiny_data):
    assert len(tiny_data) == 200
    assert LABEL_COL in tiny_data.columns
    assert tiny_data[LABEL_COL].isin([0, 1]).all()


def test_demo_positive_rate(tiny_data):
    rate = tiny_data[LABEL_COL].mean()
    assert 0.05 <= rate <= 0.50, f"Unexpected positive rate: {rate:.2f}"


def test_preprocessing_no_nan(tiny_splits):
    X_train, y_train, X_val, y_val, X_test, y_test = tiny_splits
    for X in [X_train, X_val, X_test]:
        assert not np.isnan(X).any(), "Preprocessing left NaNs in feature matrix"


def test_smote_increases_minority(tiny_splits):
    X_train, y_train, _, _, _, _ = tiny_splits
    X_res, y_res = apply_smote(X_train, y_train, random_state=42)
    assert len(y_res) >= len(y_train)
    assert y_res.sum() >= y_train.sum()


def test_logistic_regression_pipeline(tiny_splits):
    X_train, y_train, X_val, y_val, X_test, y_test = tiny_splits
    model = LogisticRegressionModel(seed=42)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    probs = model.predict_proba(X_test)
    assert probs.shape == (len(y_test),)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_xgboost_pipeline(tiny_splits):
    pytest.importorskip("xgboost")
    X_train, y_train, X_val, y_val, X_test, y_test = tiny_splits
    model = XGBoostModel(seed=42, n_estimators=10, max_depth=3)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    probs = model.predict_proba(X_test)
    assert probs.shape == (len(y_test),)


def test_lightgbm_pipeline(tiny_splits):
    pytest.importorskip("lightgbm")
    X_train, y_train, X_val, y_val, X_test, y_test = tiny_splits
    model = LightGBMModel(seed=42, n_estimators=10, num_leaves=15)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    probs = model.predict_proba(X_test)
    assert probs.shape == (len(y_test),)


def test_ft_transformer_pipeline(tiny_splits):
    pytest.importorskip("torch")
    from hfbench.models.ft_transformer import FTTransformerModel
    X_train, y_train, X_val, y_val, X_test, y_test = tiny_splits
    model = FTTransformerModel(seed=42, d_token=32, n_layers=1, n_heads=4, max_epochs=3, patience=2)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    probs = model.predict_proba(X_test)
    assert probs.shape == (len(y_test),)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_full_metrics(tiny_splits):
    X_train, y_train, X_val, y_val, X_test, y_test = tiny_splits
    model = LogisticRegressionModel(seed=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    metrics = compute_all_metrics(y_test, probs)
    assert "auroc" in metrics
    if not np.isnan(metrics["auroc"]):
        assert 0.0 <= metrics["auroc"] <= 1.0

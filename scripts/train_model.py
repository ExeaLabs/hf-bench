#!/usr/bin/env python
"""Train a single HF-Bench model with Optuna hyperparameter tuning.

Usage:
    python scripts/train_model.py --model xgboost --seed 42 --n-trials 50
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from hfbench.constants import LABEL_COL, SEEDS, SUBGROUP_COLS, TIME_COL
from hfbench.data.preprocess import apply_smote, fit_preprocessor, transform
from hfbench.data.split import temporal_split
from hfbench.models import MODEL_REGISTRY
from hfbench.training.seed import set_global_seed
from hfbench.training.train import train_best_model
from hfbench.training.tune import run_study
from hfbench.utils.config import (
    get_search_space,
    load_default_config,
    load_model_config,
)
from hfbench.utils.io import save_json, save_predictions
from hfbench.utils.logging import setup_logging

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="Train an HF-Bench model.")
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to train",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials", type=int, default=100, help="Optuna trials")
    p.add_argument("--timeout", type=float, default=None, help="Optuna timeout (seconds)")
    p.add_argument("--config", type=Path, default=None, help="Override default config path")
    p.add_argument(
        "--data-path", type=Path, default=Path("data/raw/demo_hf.csv"),
        help="Raw cohort CSV path",
    )
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--no-smote", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    set_global_seed(args.seed)

    cfg = load_default_config(REPO_ROOT)
    model_cfg = load_model_config(args.model, REPO_ROOT)
    search_space = get_search_space(model_cfg)

    logger.info(
        "[DEMO/SYNTHETIC DATA] Training model=%s seed=%d n_trials=%d",
        args.model, args.seed, args.n_trials,
    )

    # Load data
    df = pd.read_csv(args.data_path, parse_dates=["admittime", "dischtime"])
    if "age_group" not in df.columns and "age" in df.columns:
        df["age_group"] = (df["age"] >= 65).map({True: ">=65", False: "<65"})

    train, val, test = temporal_split(df)

    # Preprocess
    preprocessor = fit_preprocessor(train)
    X_train = transform(preprocessor, train)
    y_train = train[LABEL_COL].values.astype(int)
    X_val = transform(preprocessor, val)
    y_val = val[LABEL_COL].values.astype(int)
    X_test = transform(preprocessor, test)
    y_test = test[LABEL_COL].values.astype(int)

    if not args.no_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state=args.seed)

    studies_dir = args.output_dir / "studies"
    model_cls = MODEL_REGISTRY[args.model]

    # Hyperparameter tuning
    if args.n_trials > 0 and search_space:
        study = run_study(
            model_name=args.model,
            model_cls=model_cls,
            search_space=search_space,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            seed=args.seed,
            n_trials=args.n_trials,
            timeout=args.timeout,
            studies_dir=studies_dir,
        )
        best_params = study.best_params
    else:
        best_params = {}

    # Train final model
    model = train_best_model(
        model_cls=model_cls,
        best_params=best_params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seed=args.seed,
    )

    # Save model
    models_dir = args.output_dir / "models"
    model_path = models_dir / f"{args.model}_seed{args.seed}.pkl"
    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    # Save predictions
    y_pred_test = model.predict_proba(X_test)
    meta_cols = ["subject_id", "hadm_id"] + [
        c for c in dict.fromkeys(SUBGROUP_COLS) if c in test.columns
    ]
    test_meta = test[meta_cols].reset_index(drop=True)

    pred_path = args.output_dir / "predictions" / f"{args.model}_seed{args.seed}.parquet"
    save_predictions(y_test, y_pred_test, test_meta, pred_path)
    logger.info("Predictions saved to %s", pred_path)

    # Quick metrics log
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_test, y_pred_test)
        logger.info("Test AUROC=%.4f [SYNTHETIC DATA]", auc)
        save_json(
            {"model": args.model, "seed": args.seed, "test_auroc": auc, "best_params": best_params},
            args.output_dir / "metrics" / f"{args.model}_seed{args.seed}_quick.json",
        )
    except Exception as exc:
        logger.warning("Could not compute AUROC: %s", exc)


if __name__ == "__main__":
    main()

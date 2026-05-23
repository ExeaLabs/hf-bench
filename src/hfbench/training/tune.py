"""Optuna hyperparameter tuning for HF-Bench models."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from hfbench.models.base import BaseModel

logger = logging.getLogger(__name__)

# Silence optuna's per-trial logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a config search_space dict into Optuna suggestions."""
    params = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def make_objective(
    model_cls,
    search_space: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Callable[[optuna.Trial], float]:
    """Return an Optuna objective function for the given model class."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, search_space)
        model: BaseModel = model_cls(seed=seed, **params)
        try:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            probs = model.predict_proba(X_val)
            if len(np.unique(y_val)) < 2:
                return 0.5
            return float(roc_auc_score(y_val, probs))
        except Exception as exc:
            logger.warning("Trial failed: %s", exc)
            raise optuna.exceptions.TrialPruned() from exc

    return objective


def run_study(
    model_name: str,
    model_cls,
    search_space: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    n_trials: int = 100,
    timeout: Optional[float] = None,
    studies_dir: Path = Path("results/studies"),
) -> optuna.Study:
    """Run Optuna study and return best params."""
    studies_dir = Path(studies_dir)
    studies_dir.mkdir(parents=True, exist_ok=True)

    study_name = f"{model_name}_seed{seed}"
    storage_path = studies_dir / f"{study_name}.db"
    storage_url = f"sqlite:///{storage_path}"

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = make_objective(model_cls, search_space, X_train, y_train, X_val, y_val, seed)

    logger.info(
        "Starting Optuna study '%s' (%d trials)...", study_name, n_trials
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    logger.info(
        "Best trial: val AUROC=%.4f params=%s",
        study.best_value,
        study.best_params,
    )
    return study

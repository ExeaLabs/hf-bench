"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return as dict."""
    with open(Path(path)) as f:
        return yaml.safe_load(f) or {}


def load_default_config(repo_root: Path) -> Dict[str, Any]:
    return load_yaml(repo_root / "configs" / "default.yaml")


def load_model_config(model_name: str, repo_root: Path) -> Dict[str, Any]:
    path = repo_root / "configs" / "models" / f"{model_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config found for model '{model_name}' at {path}")
    return load_yaml(path)


def get_search_space(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the hyperparameter_search block from a model config."""
    return model_config.get("hyperparameter_search", {})

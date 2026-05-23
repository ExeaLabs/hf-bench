"""Model registry for HF-Bench."""

from hfbench.models.sklearn_models import LogisticRegressionModel
from hfbench.models.xgb_model import XGBoostModel
from hfbench.models.lgbm_model import LightGBMModel
from hfbench.models.mlp import MLPModel
from hfbench.models.tabnet_model import TabNetModel
from hfbench.models.ft_transformer import FTTransformerModel

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "mlp": MLPModel,
    "tabnet": TabNetModel,
    "ft_transformer": FTTransformerModel,
}


def get_model(name: str, **kwargs):
    """Instantiate a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)

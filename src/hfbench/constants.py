"""Project-wide constants."""

from pathlib import Path

# Repository root (two levels up from this file: src/hfbench/ -> src/ -> root)
REPO_ROOT = Path(__file__).parent.parent.parent

LABEL_COL = "readmit_30d"
TIME_COL = "admittime"
ID_COLS = ["subject_id", "hadm_id"]

SUBGROUP_COLS = ["sex", "race", "age_group"]

SEEDS = [42, 123, 7, 0, 256]

MODEL_NAMES = [
    "logistic_regression",
    "xgboost",
    "lightgbm",
    "mlp",
    "tabnet",
    "ft_transformer",
]

NUMERIC_FEATURES = [
    "age",
    "length_of_stay",
    "prior_admissions_6mo",
    "days_since_last_admit",
    "charlson_index",
    "bnp",
    "creatinine",
    "sodium",
    "hemoglobin",
    "egfr",
    "discharge_weight",
    "systolic_bp",
    "heart_rate",
    "o2_saturation",
    "loop_diuretic_dose",
    "med_count",
]

BINARY_FEATURES = [
    "diabetes",
    "ckd",
    "copd",
    "afib",
    "hypertension",
    "ace_arb_arni",
    "beta_blocker",
    "loop_diuretic",
]

CATEGORICAL_FEATURES = [
    "sex",
    "race",
    "insurance",
    "admission_type",
    "admission_location",
]

ALL_FEATURE_COLS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 0.15 (remainder)

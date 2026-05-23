"""Column schema and dtype definitions for HF-Bench datasets."""

from typing import Dict, List

DTYPE_MAP: Dict[str, str] = {
    "subject_id": "int64",
    "hadm_id": "int64",
    "admittime": "datetime64[ns]",
    "dischtime": "datetime64[ns]",
    "age": "float32",
    "sex": "category",
    "race": "category",
    "insurance": "category",
    "admission_type": "category",
    "admission_location": "category",
    "length_of_stay": "float32",
    "prior_admissions_6mo": "float32",
    "days_since_last_admit": "float32",
    "diabetes": "int8",
    "ckd": "int8",
    "copd": "int8",
    "afib": "int8",
    "hypertension": "int8",
    "charlson_index": "float32",
    "bnp": "float32",
    "creatinine": "float32",
    "sodium": "float32",
    "hemoglobin": "float32",
    "egfr": "float32",
    "discharge_weight": "float32",
    "systolic_bp": "float32",
    "heart_rate": "float32",
    "o2_saturation": "float32",
    "loop_diuretic_dose": "float32",
    "ace_arb_arni": "int8",
    "beta_blocker": "int8",
    "loop_diuretic": "int8",
    "med_count": "float32",
    "readmit_30d": "int8",
}

REQUIRED_COLUMNS: List[str] = list(DTYPE_MAP.keys())

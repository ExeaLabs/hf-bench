"""Generate a synthetic heart failure readmission demo dataset.

All data is entirely synthetic. It is NOT derived from real patients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def make_demo_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic HF readmission cohort.

    Parameters
    ----------
    n:
        Number of rows (patients/admissions).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns matching the MIMIC-IV cohort schema.
    """
    rng = np.random.default_rng(seed)

    subject_ids = np.arange(1, n + 1)
    hadm_ids = np.arange(100_000, 100_000 + n)

    # Admission times spanning ~4 years for temporal splitting
    base_date = pd.Timestamp("2018-01-01")
    admit_offsets_days = rng.uniform(0, 365 * 4, size=n)
    admit_offsets_days = np.sort(admit_offsets_days)  # temporal ordering
    admittimes = [base_date + pd.Timedelta(days=float(d)) for d in admit_offsets_days]

    los = rng.gamma(shape=2.0, scale=3.5, size=n).clip(0.5, 60.0)
    dischtimes = [a + pd.Timedelta(days=float(d)) for a, d in zip(admittimes, los)]

    age = rng.normal(loc=72, scale=12, size=n).clip(18, 100)

    sex = rng.choice(["M", "F"], size=n, p=[0.52, 0.48])

    race_choices = ["WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO", "ASIAN", "OTHER/UNKNOWN"]
    race_probs = [0.60, 0.18, 0.10, 0.05, 0.07]
    race = rng.choice(race_choices, size=n, p=race_probs)

    insurance_choices = ["Medicare", "Medicaid", "Private", "Self Pay", "Other"]
    insurance_probs = [0.55, 0.18, 0.18, 0.05, 0.04]
    insurance = rng.choice(insurance_choices, size=n, p=insurance_probs)

    admission_type_choices = ["EMERGENCY", "ELECTIVE", "URGENT", "OBSERVATION ADMIT"]
    admission_type_probs = [0.65, 0.10, 0.15, 0.10]
    admission_type = rng.choice(admission_type_choices, size=n, p=admission_type_probs)

    admission_location_choices = [
        "EMERGENCY ROOM", "PHYSICIAN REFERRAL", "TRANSFER FROM HOSPITAL",
        "WALK-IN/SELF REFERRAL", "OTHER"
    ]
    admission_location_probs = [0.55, 0.20, 0.15, 0.05, 0.05]
    admission_location = rng.choice(admission_location_choices, size=n, p=admission_location_probs)

    prior_admissions_6mo = rng.poisson(lam=0.8, size=n).clip(0, 10).astype(float)
    days_since_last_admit = np.where(
        prior_admissions_6mo > 0,
        rng.exponential(scale=45, size=n).clip(1, 180),
        np.nan,
    )

    # Comorbidities — correlated with age
    age_norm = (age - 18) / 82.0
    diabetes = (rng.random(n) < (0.25 + 0.15 * age_norm)).astype(int)
    ckd = (rng.random(n) < (0.20 + 0.20 * age_norm)).astype(int)
    copd = (rng.random(n) < (0.15 + 0.10 * age_norm)).astype(int)
    afib = (rng.random(n) < (0.25 + 0.25 * age_norm)).astype(int)
    hypertension = (rng.random(n) < (0.55 + 0.20 * age_norm)).astype(int)

    charlson_index = (
        diabetes * 1.0
        + ckd * 2.0
        + copd * 1.0
        + afib * 0.5
        + hypertension * 0.5
        + rng.poisson(lam=0.5, size=n).astype(float)
    ).clip(0, 15)

    # Labs — with realistic missingness
    # BNP: ~40% missing
    bnp_true = rng.lognormal(mean=np.log(500 + 300 * ckd + 200 * age_norm), sigma=1.0, size=n)
    bnp = np.where(rng.random(n) < 0.40, np.nan, bnp_true)

    creatinine = np.where(
        rng.random(n) < 0.08,
        np.nan,
        rng.gamma(shape=3, scale=0.5, size=n).clip(0.4, 12.0) + ckd * 0.8,
    )

    sodium = np.where(
        rng.random(n) < 0.08,
        np.nan,
        rng.normal(loc=138, scale=4, size=n).clip(120, 155),
    )

    hemoglobin = np.where(
        rng.random(n) < 0.10,
        np.nan,
        rng.normal(loc=11.5 - 1.5 * (sex == "F").astype(float), scale=1.8, size=n).clip(5, 18),
    )

    # eGFR: ~20% missing, inversely related to creatinine
    egfr_true = (90.0 / np.where(creatinine > 0, creatinine, 1.0)).clip(5, 120)
    egfr = np.where(
        rng.random(n) < 0.20,
        np.nan,
        egfr_true + rng.normal(0, 5, n),
    )

    discharge_weight = np.where(
        rng.random(n) < 0.25,
        np.nan,
        rng.normal(loc=90, scale=20, size=n).clip(35, 250),
    )

    systolic_bp = np.where(
        rng.random(n) < 0.15,
        np.nan,
        rng.normal(loc=122, scale=18, size=n).clip(70, 200),
    )

    heart_rate = np.where(
        rng.random(n) < 0.12,
        np.nan,
        rng.normal(loc=80, scale=16, size=n).clip(40, 160),
    )

    o2_saturation = np.where(
        rng.random(n) < 0.15,
        np.nan,
        rng.normal(loc=96, scale=3, size=n).clip(70, 100),
    )

    loop_diuretic_dose = np.where(
        rng.random(n) < 0.30,
        np.nan,
        rng.choice([20, 40, 80, 120, 160, 200], size=n, p=[0.05, 0.30, 0.30, 0.20, 0.10, 0.05]).astype(float),
    )

    ace_arb_arni = (rng.random(n) < 0.55).astype(int)
    beta_blocker = (rng.random(n) < 0.65).astype(int)
    loop_diuretic = (rng.random(n) < 0.75).astype(int)
    med_count = rng.poisson(lam=8, size=n).clip(1, 25).astype(float)

    # Label: logistic function of clinical predictors
    # Fills NaN with median for label generation only
    creat_fill = np.where(np.isnan(creatinine), np.nanmedian(creatinine), creatinine)
    sodium_fill = np.where(np.isnan(sodium), np.nanmedian(sodium), sodium)
    bnp_fill = np.where(np.isnan(bnp), np.nanmedian(bnp[~np.isnan(bnp)]), bnp)

    logit = (
        -2.8
        + 0.03 * (age - 70)
        + 0.35 * prior_admissions_6mo
        + 0.50 * ckd
        + 0.25 * diabetes
        + 0.30 * copd
        + 0.00003 * bnp_fill
        + 0.20 * creat_fill
        - 0.04 * (sodium_fill - 135)
        + 0.06 * los
        + 0.04 * med_count
        - 0.20 * ace_arb_arni
        - 0.15 * beta_blocker
        + rng.normal(0, 0.8, n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    readmit_30d = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "subject_id": subject_ids,
        "hadm_id": hadm_ids,
        "admittime": admittimes,
        "dischtime": dischtimes,
        "age": age.astype("float32"),
        "sex": sex,
        "race": race,
        "insurance": insurance,
        "admission_type": admission_type,
        "admission_location": admission_location,
        "length_of_stay": los.astype("float32"),
        "prior_admissions_6mo": prior_admissions_6mo.astype("float32"),
        "days_since_last_admit": days_since_last_admit.astype("float32"),
        "diabetes": diabetes.astype("int8"),
        "ckd": ckd.astype("int8"),
        "copd": copd.astype("int8"),
        "afib": afib.astype("int8"),
        "hypertension": hypertension.astype("int8"),
        "charlson_index": charlson_index.astype("float32"),
        "bnp": bnp.astype("float32"),
        "creatinine": creatinine.astype("float32"),
        "sodium": sodium.astype("float32"),
        "hemoglobin": hemoglobin.astype("float32"),
        "egfr": egfr.astype("float32"),
        "discharge_weight": discharge_weight.astype("float32"),
        "systolic_bp": systolic_bp.astype("float32"),
        "heart_rate": heart_rate.astype("float32"),
        "o2_saturation": o2_saturation.astype("float32"),
        "loop_diuretic_dose": loop_diuretic_dose.astype("float32"),
        "ace_arb_arni": ace_arb_arni.astype("int8"),
        "beta_blocker": beta_blocker.astype("int8"),
        "loop_diuretic": loop_diuretic.astype("int8"),
        "med_count": med_count.astype("float32"),
        "readmit_30d": readmit_30d.astype("int8"),
    })

    return df

-- =============================================================================
-- HF-Bench: MIMIC-IV Heart Failure Cohort Extraction
-- =============================================================================
-- Purpose : Extract 30-day readmission cohort after index HF admissions.
-- Database: PostgreSQL with MIMIC-IV v2.0+ loaded in schema 'mimiciv_hosp'
--           and 'mimiciv_derived' (if using MIMIC-IV derived tables).
-- NOTE    : Demo mode does NOT require running this SQL.
--           Use scripts/make_demo_data.py for a fully synthetic dataset.
-- =============================================================================

-- Confirm schema names match your MIMIC-IV installation.
-- Adjust 'mimiciv_hosp', 'mimiciv_icu', 'mimiciv_derived' as needed.

-- -----------------------------------------------------------------------------
-- Step 1: Heart failure index admissions
-- Adults (>=18), ICD-10 I50.*, discharged alive
-- -----------------------------------------------------------------------------
WITH hf_diagnoses AS (
    SELECT DISTINCT d.hadm_id
    FROM mimiciv_hosp.diagnoses_icd d
    WHERE d.icd_version = 10
      AND d.icd_code LIKE 'I50%'
),

admissions_base AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.discharge_location,
        a.admission_type,
        a.admission_location,
        a.insurance,
        -- Exclude deaths at discharge
        CASE
            WHEN UPPER(a.discharge_location) LIKE '%DEAD%'
              OR UPPER(a.discharge_location) LIKE '%EXPIRED%'
              OR UPPER(a.discharge_location) LIKE '%HOSPICE%'
            THEN 1
            ELSE 0
        END AS discharge_death_flag,
        EXTRACT(EPOCH FROM (a.dischtime - a.admittime)) / 86400.0 AS length_of_stay_days,
        p.anchor_age                                                AS age_at_anchor,
        -- Approximate age at admission
        p.anchor_age + EXTRACT(YEAR FROM a.admittime)
                     - p.anchor_year                               AS age,
        p.gender                                                   AS sex,
        a.race
    FROM mimiciv_hosp.admissions a
    JOIN mimiciv_hosp.patients p ON a.subject_id = p.subject_id
    JOIN hf_diagnoses hf ON a.hadm_id = hf.hadm_id
    WHERE p.anchor_age >= 18
),

index_admissions AS (
    SELECT *
    FROM admissions_base
    WHERE discharge_death_flag = 0
      AND age >= 18
),

-- -----------------------------------------------------------------------------
-- Step 2: 30-day readmission label
-- Any subsequent admission within 30 days after discharge
-- -----------------------------------------------------------------------------
readmissions AS (
    SELECT
        ia.hadm_id                                                 AS index_hadm_id,
        MIN(a_next.admittime)                                      AS next_admit_time,
        CASE
            WHEN MIN(a_next.admittime) IS NOT NULL
             AND EXTRACT(EPOCH FROM (MIN(a_next.admittime) - ia.dischtime))
                 / 86400.0 <= 30
            THEN 1
            ELSE 0
        END AS readmit_30d
    FROM index_admissions ia
    LEFT JOIN mimiciv_hosp.admissions a_next
        ON ia.subject_id = a_next.subject_id
       AND a_next.admittime > ia.dischtime
       AND EXTRACT(EPOCH FROM (a_next.admittime - ia.dischtime)) / 86400.0 <= 30
    GROUP BY ia.hadm_id, ia.dischtime
),

-- -----------------------------------------------------------------------------
-- Step 3: Prior admissions in previous 6 months
-- -----------------------------------------------------------------------------
prior_admissions AS (
    SELECT
        ia.hadm_id,
        COUNT(a_prior.hadm_id)                                     AS prior_admissions_6mo,
        MIN(EXTRACT(EPOCH FROM (ia.admittime - a_prior.dischtime))
            / 86400.0)                                             AS days_since_last_admit
    FROM index_admissions ia
    LEFT JOIN mimiciv_hosp.admissions a_prior
        ON ia.subject_id = a_prior.subject_id
       AND a_prior.dischtime < ia.admittime
       AND EXTRACT(EPOCH FROM (ia.admittime - a_prior.dischtime)) / 86400.0 <= 180
    GROUP BY ia.hadm_id, ia.admittime
),

-- -----------------------------------------------------------------------------
-- Step 4: Comorbidities from ICD-10 codes on index admission
-- -----------------------------------------------------------------------------
comorbidities AS (
    SELECT
        hadm_id,
        MAX(CASE
            WHEN icd_version = 10 AND (
                icd_code LIKE 'E10%' OR icd_code LIKE 'E11%' OR
                icd_code LIKE 'E12%' OR icd_code LIKE 'E13%'
            ) THEN 1 ELSE 0
        END) AS diabetes,
        MAX(CASE
            WHEN icd_version = 10 AND (
                icd_code LIKE 'N18%' OR icd_code LIKE 'N19%'
            ) THEN 1 ELSE 0
        END) AS ckd,
        MAX(CASE
            WHEN icd_version = 10 AND (
                icd_code LIKE 'J44%' OR icd_code LIKE 'J43%'
            ) THEN 1 ELSE 0
        END) AS copd,
        MAX(CASE
            WHEN icd_version = 10 AND (
                icd_code LIKE 'I48%'
            ) THEN 1 ELSE 0
        END) AS afib,
        MAX(CASE
            WHEN icd_version = 10 AND (
                icd_code LIKE 'I10%' OR icd_code LIKE 'I11%' OR
                icd_code LIKE 'I12%' OR icd_code LIKE 'I13%'
            ) THEN 1 ELSE 0
        END) AS hypertension
    FROM mimiciv_hosp.diagnoses_icd
    GROUP BY hadm_id
),

-- -----------------------------------------------------------------------------
-- Step 5: Lab values — most recent before discharge
-- TODO: Confirm itemids against your MIMIC-IV labevents table.
-- Common MIMIC-IV itemids (may vary by version):
--   BNP:        item_id 53955 (or NT-proBNP: 50963)
--   Creatinine: 50912
--   Sodium:     50983
--   Hemoglobin: 51222
--   eGFR:       see note — often derived, not directly stored
-- -----------------------------------------------------------------------------
labs AS (
    SELECT DISTINCT ON (le.hadm_id, le.itemid)
        le.hadm_id,
        le.itemid,
        le.valuenum
    FROM mimiciv_hosp.labevents le
    JOIN index_admissions ia ON le.hadm_id = ia.hadm_id
    WHERE le.itemid IN (
        53955,  -- BNP (pg/mL) — verify itemid
        50963,  -- NT-proBNP — verify itemid
        50912,  -- Creatinine (mg/dL)
        50983,  -- Sodium (mEq/L)
        51222,  -- Hemoglobin (g/dL)
        50806   -- Chloride — placeholder; replace with eGFR source
    )
      AND le.valuenum IS NOT NULL
      AND le.valuenum > 0
      AND le.charttime <= ia.dischtime
    ORDER BY le.hadm_id, le.itemid, le.charttime DESC
),

labs_pivot AS (
    SELECT
        hadm_id,
        MAX(CASE WHEN itemid IN (53955, 50963) THEN valuenum END) AS bnp,
        MAX(CASE WHEN itemid = 50912 THEN valuenum END)            AS creatinine,
        MAX(CASE WHEN itemid = 50983 THEN valuenum END)            AS sodium,
        MAX(CASE WHEN itemid = 51222 THEN valuenum END)            AS hemoglobin
        -- eGFR: derive from creatinine using CKD-EPI formula, or join derived tables
    FROM labs
    GROUP BY hadm_id
),

-- -----------------------------------------------------------------------------
-- Step 6: Medication flags from prescriptions table
-- -----------------------------------------------------------------------------
meds AS (
    SELECT
        hadm_id,
        MAX(CASE
            WHEN LOWER(drug) LIKE '%furosemide%'
              OR LOWER(drug) LIKE '%torsemide%'
              OR LOWER(drug) LIKE '%bumetanide%'
              OR LOWER(drug) LIKE '%ethacrynic%'
            THEN 1 ELSE 0
        END) AS loop_diuretic,
        MAX(CASE
            WHEN LOWER(drug) LIKE '%lisinopril%'
              OR LOWER(drug) LIKE '%enalapril%'
              OR LOWER(drug) LIKE '%captopril%'
              OR LOWER(drug) LIKE '%ramipril%'
              OR LOWER(drug) LIKE '%losartan%'
              OR LOWER(drug) LIKE '%valsartan%'
              OR LOWER(drug) LIKE '%irbesartan%'
              OR LOWER(drug) LIKE '%sacubitril%'
            THEN 1 ELSE 0
        END) AS ace_arb_arni,
        MAX(CASE
            WHEN LOWER(drug) LIKE '%metoprolol%'
              OR LOWER(drug) LIKE '%carvedilol%'
              OR LOWER(drug) LIKE '%bisoprolol%'
              OR LOWER(drug) LIKE '%atenolol%'
            THEN 1 ELSE 0
        END) AS beta_blocker,
        COUNT(DISTINCT drug)                                       AS med_count
    FROM mimiciv_hosp.prescriptions
    GROUP BY hadm_id
),

-- -----------------------------------------------------------------------------
-- Step 7: Charlson Comorbidity Index — simplified version
-- For full CCI, consider the MIMIC-IV derived tables (comorbidity_scores)
-- -----------------------------------------------------------------------------
cci AS (
    SELECT
        hadm_id,
        -- Minimal CCI proxy from comorbidity flags (full version in derived tables)
        (CASE WHEN diabetes = 1 THEN 1 ELSE 0 END
         + CASE WHEN ckd = 1 THEN 2 ELSE 0 END
         + CASE WHEN copd = 1 THEN 1 ELSE 0 END) AS charlson_index_approx
    FROM comorbidities
)

-- -----------------------------------------------------------------------------
-- Final cohort assembly
-- -----------------------------------------------------------------------------
SELECT
    ia.subject_id,
    ia.hadm_id,
    ia.admittime,
    ia.dischtime,
    ia.age,
    ia.sex,
    ia.race,
    ia.insurance,
    ia.admission_type,
    ia.admission_location,
    ia.length_of_stay_days                             AS length_of_stay,
    COALESCE(pa.prior_admissions_6mo, 0)               AS prior_admissions_6mo,
    pa.days_since_last_admit,
    COALESCE(cm.diabetes, 0)                           AS diabetes,
    COALESCE(cm.ckd, 0)                                AS ckd,
    COALESCE(cm.copd, 0)                               AS copd,
    COALESCE(cm.afib, 0)                               AS afib,
    COALESCE(cm.hypertension, 0)                       AS hypertension,
    COALESCE(cci.charlson_index_approx, 0)             AS charlson_index,
    lp.bnp,
    lp.creatinine,
    lp.sodium,
    lp.hemoglobin,
    -- eGFR: join mimiciv_derived.kdigo_stages or compute from creatinine
    NULL::FLOAT                                        AS egfr,  -- TODO: derive
    NULL::FLOAT                                        AS discharge_weight,  -- from chartevents
    NULL::FLOAT                                        AS systolic_bp,       -- from chartevents
    NULL::FLOAT                                        AS heart_rate,        -- from chartevents
    NULL::FLOAT                                        AS o2_saturation,     -- from chartevents
    NULL::FLOAT                                        AS loop_diuretic_dose,-- from prescriptions
    COALESCE(m.ace_arb_arni, 0)                        AS ace_arb_arni,
    COALESCE(m.beta_blocker, 0)                        AS beta_blocker,
    COALESCE(m.loop_diuretic, 0)                       AS loop_diuretic,
    COALESCE(m.med_count, 0)                           AS med_count,
    COALESCE(r.readmit_30d, 0)                         AS readmit_30d

FROM index_admissions ia
LEFT JOIN readmissions r       ON ia.hadm_id = r.index_hadm_id
LEFT JOIN prior_admissions pa  ON ia.hadm_id = pa.hadm_id
LEFT JOIN comorbidities cm     ON ia.hadm_id = cm.hadm_id
LEFT JOIN labs_pivot lp        ON ia.hadm_id = lp.hadm_id
LEFT JOIN meds m               ON ia.hadm_id = m.hadm_id
LEFT JOIN cci                  ON ia.hadm_id = cci.hadm_id

ORDER BY ia.admittime;

-- =============================================================================
-- Notes:
-- 1. eGFR: Use mimiciv_derived.chemistry or compute CKD-EPI from creatinine/age/sex.
-- 2. Vitals (BP, HR, SpO2, weight): Extract from chartevents using itemids:
--    HR=220045, SBP=220179, DBP=220180, SpO2=220277, weight=226512/226531
-- 3. Loop diuretic dose: Aggregate from prescriptions.dose_val_rx.
-- 4. Charlson index: Use mimiciv_derived.charlson for full CCI.
-- 5. This query may be slow on large MIMIC; add indexes on hadm_id, subject_id.
-- =============================================================================

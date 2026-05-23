# HF-Bench: Tabular Deep Learning vs Gradient-Boosted Trees for 30-Day Heart Failure Readmission

> **⚠️ RESEARCH CODE — NOT FOR CLINICAL USE**
> This benchmark is intended solely for research and educational purposes. Outputs are generated from synthetic demo data. Do not use for clinical decision-making, patient care, or deployment.

---

## Overview

HF-Bench is a NeurIPS-style reproducible benchmark comparing tabular deep learning models against gradient-boosted trees for **30-day hospital readmission prediction after heart failure admission**. The benchmark evaluates six model families with:

- Rigorous temporal train/val/test splitting
- Hyperparameter tuning via Optuna
- Comprehensive evaluation: AUROC, AUPRC, Brier score, ECE, threshold-specific metrics
- Subgroup fairness analysis by sex, race/ethnicity, and age group
- Multi-seed reproducibility with 95% confidence intervals

The pipeline ships with a **synthetic demo dataset** so the full benchmark can run immediately without PhysioNet access. A MIMIC-IV SQL cohort extraction script is included for research use once data access is obtained.

---

## Quickstart (Demo Mode)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Generate synthetic demo data (~5000 patients)
python scripts/make_demo_data.py

# Run smoke test (2 trials, 1 seed, fast)
python scripts/run_all.py --smoke-test
```

## Full Benchmark

```bash
python scripts/run_all.py --n-trials 100
```

This runs all 6 models x 5 seeds = 30 training runs with 100 Optuna trials each.

## Individual Model Training

```bash
python scripts/train_model.py --model xgboost --seed 42 --n-trials 50
python scripts/train_model.py --model ft_transformer --seed 42 --n-trials 20
```

## Evaluate and Aggregate

```bash
python scripts/evaluate.py --predictions-dir results/predictions
python scripts/aggregate_results.py
```

---

## MIMIC-IV (Real Data)

The SQL cohort definition is in `sql/mimic_iv_hf_cohort.sql`. It requires:
- MIMIC-IV v2.0+ access via PhysioNet (https://physionet.org/content/mimiciv/)
- PostgreSQL database with MIMIC-IV loaded

Once data is extracted:
```bash
# Place cohort CSV at data/raw/mimic_hf_cohort.csv
python scripts/preprocess.py --data-path data/raw/mimic_hf_cohort.csv
python scripts/run_all.py --data-path data/raw/mimic_hf_cohort.csv --n-trials 100
```

The demo dataset does NOT require running the SQL file.

---

## Project Structure

```
HF-Bench/
├── configs/                  # Model and pipeline configuration
│   ├── default.yaml
│   └── models/               # Per-model hyperparameter search spaces
├── sql/
│   └── mimic_iv_hf_cohort.sql   # MIMIC-IV cohort extraction (PostgreSQL)
├── data/
│   ├── raw/                  # Input data (demo CSV or MIMIC export)
│   ├── interim/              # Split labels
│   └── processed/            # Preprocessed parquet files
├── scripts/                  # CLI entry points
├── src/hfbench/              # Core library
│   ├── data/                 # Dataset generation, splitting, preprocessing
│   ├── models/               # Model wrappers (LR, XGB, LGBM, MLP, TabNet, FT-T)
│   ├── training/             # Optuna tuning and training logic
│   ├── evaluation/           # Metrics, calibration, fairness, bootstrap CI
│   └── utils/                # I/O, logging, config loading
├── notebooks/
├── results/
└── tests/
```

---

## Models

| Model | Library | Notes |
|---|---|---|
| Logistic Regression | scikit-learn | L1/L2, class_weight='balanced' |
| XGBoost | xgboost | Full hyperparameter search |
| LightGBM | lightgbm | Full hyperparameter search |
| MLP | PyTorch | BatchNorm, Dropout, early stopping |
| TabNet | pytorch-tabnet | Attentive feature selection |
| FT-Transformer | PyTorch (custom) | Feature tokenizer + Transformer |

---

## Evaluation Metrics

**Discrimination:** AUROC, AUPRC (average precision)

**Calibration:** Brier score, Expected Calibration Error (ECE, 10 bins)

**Threshold-specific (@ 0.5 and @ 80% specificity operating point):**
Sensitivity, Specificity, PPV, NPV

**Fairness (subgroups: sex, race/ethnicity, age group <65 / >=65):**
- Subgroup AUROC, worst-case AUROC
- Equalized odds gap (TPR gap, FPR gap)
- Demographic parity gap
- Calibration by subgroup

---

## Reproducibility

Five random seeds: **42, 123, 7, 0, 256**

Results aggregated as **mean +/- 95% CI** = 1.96 * std / sqrt(n_seeds).

Temporal splits are deterministic (sort by admittime, 70/15/15).

---

## Expected Outputs (Demo/Synthetic)

All demo outputs are based on **synthetic data** and do not represent real clinical performance.

---

## Citation

```bibtex
@misc{hfbench2024,
  title  = {HF-Bench: A Benchmark for 30-Day Heart Failure Readmission Prediction},
  author = {[Authors]},
  year   = {2024},
  note   = {Preprint}
}
```

## License

MIT License.

## Disclaimer

Research purposes only. Not validated for clinical use.

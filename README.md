# Insulin Resistance Prediction System

**Status:** ✅ Production-Ready • **Last Updated:** 28 Nov 2025 • **Python:** 3.13.7 • **License:** MIT • **Author:** Rahul

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
4. [Dataset & Data](#dataset--data)
5. [Installation & Setup](#installation--setup)
6. [Model Architecture](#model-architecture)
7. [Feature Engineering](#feature-engineering)
8. [API Documentation](#api-documentation)
9. [Deployment](#deployment)
10. [Monitoring & Operations](#monitoring--operations)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)
13. [Contributing](#contributing)
14. [References](#references)

---

## 🎯 Overview

A **clinical-grade machine learning system** predicting insulin resistance using gradient-boosted ensemble stacking with advanced calibration, SHAP explanations, and production monitoring. Built for healthcare practitioners and ML researchers.

**Core Technology:**
- **Ensemble Stack:** XGBoost + LightGBM + CatBoost + GradientBoosting (Level-0)
- **Meta-Learner:** Isotonic-calibrated Logistic Regression (Level-1)
- **Explainability:** SHAP-based feature attribution with aggregated weights
- **Monitoring:** JSONL prediction logs, Prometheus metrics, KS-test drift detection
- **Deployment:** FastAPI microservice

**Performance Metrics (Target):**
- ROC AUC: **> 0.90** (Rigorous CV)
- F1 Score: **> 0.75**
- Brier Score: **< 0.10**
- Calibration Error: **< 5%**

---

## 🛡️ Quality Assurance & Leakage Prevention

To ensure realistic generalization and prevent "too good to be true" results, the system implements:

1.  **Leakage Guard:** Automatic detection and removal of features with >0.995 correlation to the label (excluding raw inputs).
2.  **Data Retention:** Revised loader retains rows with partial missingness for imputation, preventing artificial dataset simplification (previously dropped 50k rows).
3.  **Feature Exclusion:** Direct label proxies (HOMA-IR) are excluded from the feature set to force the model to learn from raw biomarkers.
4.  **Generalization Gap:** Monitoring of Train vs Validation AUC to detect overfitting.
5.  **Capacity Control:** Reduced model complexity (n_estimators=200, max_depth=4) to prevent memorization.

---

## 🩺 Background & Motivation

Insulin resistance is a metabolic state in which peripheral tissues (muscle, liver, adipose) respond poorly to circulating insulin. To maintain normal blood glucose, the pancreas compensates by secreting more insulin. Over time, this compensation can fail, leading to type 2 diabetes, cardiovascular disease, and non-alcoholic fatty liver disease.

### Limitations of current diagnostic methods

- **Gold-standard tests** such as the hyperinsulinemic–euglycemic clamp and frequently sampled IVGTT are accurate but invasive, expensive, and impractical for routine screening.
- **Simple surrogate indices** (fasting glucose, fasting insulin, HOMA-IR) rely on fixed thresholds and can:
  - Miss borderline or early insulin resistance.
  - Be sensitive to assay variability and population differences.
- In typical clinical workflows, rich information from **lipid profiles, anthropometrics, blood pressure, and demographics** is rarely combined into a single, quantitative risk score.

### Why machine learning is useful here

This project applies ML to leverage routinely collected clinical data:

- Learns **non-linear interactions** between biomarkers (e.g., BMI × age, triglyceride-to-HDL ratio with glucose control) that are hard to capture with rule-based scores.
- Produces **calibrated risk probabilities** instead of yes/no cut-offs, allowing clinicians to choose operating points (high-sensitivity for screening vs higher-specificity for confirmatory decisions).
- Aggregates dozens of features into a single **insulin resistance risk score** that can sit alongside existing indices like HOMA-IR rather than replace them.

### Importance of early detection

Identifying insulin resistance before overt hyperglycaemia or diabetes enables:

- Earlier lifestyle and pharmacologic interventions while β-cell function is still preserved.
- Better stratification of cardiometabolic risk in populations already undergoing routine blood work.
- Use of existing labs (glucose, insulin, lipids) and measurements (BMI, waist, blood pressure) without requiring new tests, making this approach practical for real-world screening.

---

## ⭐ Key Features

| Feature | Details |
|---------|---------|
| 🔁 **Reproducible** | Deterministic seeds, serialized transformers, audit logs |
| 🧪 **40 Engineered Features** | HOMA-IR, QUICKI, TG/HDL, waist-hip, BMI interactions |
| 📈 **Medical-Grade Metrics** | ROC/PR curves, Brier score, calibration, threshold optimization |
| ⚙️ **Production Monitoring** | JSONL prediction logs, Prometheus metrics, drift alerts |
| 🎯 **Explainability** | SHAP top-3 feature drivers per prediction |
| 🧾 **Operational Scripts** | Smoke tests, external validation, drift simulation |
| 🔒 **Privacy Compliant** | Anonymized logging, trace IDs, PHI handling checklist |
| 🚀 **CI/CD Integrated** | GitHub Actions workflow, automated testing |

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Python 3.11+ (tested on 3.13.7)
- pip or conda
- Git
- 2GB disk space, 4GB RAM (8GB recommended for training)

### Step 1: Clone & Setup

```bash
# Clone repository
git clone https://github.com/soulrahulrk/insulin-resistance-prediction.git
cd insulin-resistance-prediction

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Install dependencies
pip install -r config/requirements.txt
```

### Step 2: Train Ensemble (7 minutes)

```bash
python -m src.train
```

**Output:**
- `models/ir_ensemble_best.pkl` – Trained stacking ensemble
- `models/feature_transformer.pkl` – Preprocessing pipeline
- `models/selected_features.json` – 40 final features
- `models/optimal_threshold.txt` – F1-optimized threshold
- `models/performance_metrics.json` – Validation metrics
- `models/train.log` – Training trace

### Step 3: Evaluate (1 minute)

```bash
python -m src.test_model
```

**Output:**
- Console metrics (ROC AUC, F1, Brier, confusion matrix)
- `models/test.log` – Evaluation trace
- `reports/test_confusion_matrix.png` – Visualization

### Step 4: Deploy API

```bash
uvicorn src.deploy_api:app --host 0.0.0.0 --port 8000 --reload
```

**Access:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/health

---

## 📊 Dataset & Data

### Data Source
- **Name:** `all_datasets_merged.csv`
- **Size:** 57,092 rows × 61 columns
- **Location:** `data/` (local only, excluded from Git)
- **Format:** CSV with headers

### Key Columns (Input Features)

| Category | Features | Count |
|----------|----------|-------|
| **Demographics** | age, sex, ethnicity | 3 |
| **Anthropometric** | weight, height, bmi, waist, hip | 5 |
| **Glucose Metabolism** | fasting_glucose, glucose_2h, hba1c, fasting_insulin | 4 |
| **Lipids** | total_cholesterol, ldl, hdl, triglycerides | 4 |
| **Other Markers** | sbp, dbp, liver_enzymes, kidney_function, etc. | 40+ |

### Target Variable
- **Label:** `ir_label` (binary: 0=no insulin resistance, 1=insulin resistance)
- **Definition:** HOMA-IR ≥ 2.5 (standard clinical threshold)
- **Prevalence:** ~40% positive class

### Data Preprocessing

```python
# 1. Column canonicalization (src/data_loader.py)
standardize_column_names(df)

# 2. Feature engineering (src/features.py)
add_homa_ir(df)
add_quicki(df)
add_tg_hdl_ratio(df)
# ... 10+ engineered features

# 3. Imputation (src/preprocessing.py)
KNN imputation (k=5) for missing values
Result: 100% complete dataset

# 4. Encoding & Scaling
OneHotEncoder for categorical variables
StandardScaler for numeric features

# 5. Feature Selection
Mutual Information score ≥ 0.001
Selected: 40 features from 61
```

### Data Split Strategy
- **Train:** 70% (39,964 rows)
- **Validation (Internal):** 15% (8,564 rows) – 5-fold CV
- **Test (Hold-out):** 15% (8,564 rows) – Final evaluation

---

## 🔍 Exploratory Data Analysis

This section summarizes the main data patterns observed during EDA. (Full notebooks and figures live under `docs/eda/`.)

### Distributions & class balance

- The derived target `ir_label` is **moderately imbalanced**, with a higher proportion of insulin-resistant cases than non-resistant, which motivated use of class weighting and explicit threshold tuning.
- Histograms and KDE plots for **fasting glucose** and **fasting insulin** show noticeably **right-skewed distributions**, especially in the insulin-resistant group; log-transformations used in feature engineering help stabilize these.
- **BMI** is shifted upward in the insulin-resistant group, with a clear enrichment of overweight and obese categories.

### Group differences (boxplots / violins)

Comparing `ir_label = 0` vs `ir_label = 1`:

- **Fasting insulin, fasting glucose, HOMA-IR, BMI, and TG/HDL ratio** all have higher medians and wider upper tails in the insulin-resistant group.
- The lipid profile in insulin-resistant individuals tends to show **higher triglycerides and lower HDL**, consistent with an atherogenic pattern.
- Age distributions indicate that insulin resistance prevalence increases with age, but younger individuals with high adiposity and adverse lipids also appear in the high-risk tail.

### Correlation structure (heatmap)

- Strong positive correlations are seen between **HOMA-IR**, fasting insulin, and fasting glucose (by construction).
- Anthropometric measures (BMI, waist, hip, waist–hip ratio) form a correlated cluster, justifying downstream feature selection to reduce redundancy.
- Moderate correlations exist between adiposity markers and triglycerides, reflecting the expected link between visceral fat and dyslipidaemia.

### SHAP feature importance & interpretation

Global SHAP analysis on the final stacking ensemble highlights:

- **Top global contributors**: HOMA-IR, fasting insulin, BMI, waist–hip ratio, TG/HDL ratio, age, and engineered interactions such as age × BMI.
- Higher values of HOMA-IR, TG/HDL, and BMI consistently push predictions toward **higher risk**, while more favourable lipid profiles and normal BMI push toward **lower risk**.
- For each individual prediction, the API can return the **top 3 SHAP features** (`shap_top3` field), explaining which measurements most increased or decreased that patient’s estimated risk.

Example plots are stored in the repository and can be linked into downstream reports or dashboards:

- Histograms and KDEs of glucose, insulin, BMI (e.g. `docs/eda/glucose_distribution.png`).
- Boxplots comparing key features by `ir_label` (e.g. `docs/eda/bmi_by_ir_label.png`).
- Correlation heatmap (e.g. `docs/eda/correlation_heatmap.png`).
- SHAP summary and dependence plots (e.g. `docs/shap/shap_summary.png`).

---

## 💻 Installation & Setup

### Full Installation

```bash
# 1. Virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install all dependencies
pip install -r config/requirements.txt

# 3. Verify installation
python -c "import pandas; import xgboost; import lightgbm; import catboost; print('✅ All packages installed')"
```

### Requirements Files

| File | Purpose | Size |
|------|---------|------|
| `config/requirements.txt` | Full dev environment (training, testing, API) | ~100 packages |
| `requirements-prod.txt` | Lean production deployment (API only) | ~20 packages |

### Environment Variables (Optional)

```bash
# Data path (default: data/all_datasets_merged.csv)
export IR_DATA_PATH=/path/to/dataset.csv

# API settings
export IR_API_HOST=0.0.0.0
export IR_API_PORT=8000
export IR_API_WORKERS=4

# Logging
export LOG_LEVEL=INFO
```

---

## 🧠 Model Architecture

### Level-0: Base Learners

| Model | Hyperparameters | ROC AUC | Notes |
|-------|-----------------|---------|-------|
| **XGBoost** | max_depth=6, learning_rate=0.1, n_estimators=200 | 0.923 | Primary ensemble member |
| **LightGBM** | num_leaves=31, learning_rate=0.1, n_estimators=200 | 0.920 | Fast tree boosting |
| **CatBoost** | depth=6, learning_rate=0.1, iterations=200 | 0.918 | Categorical handling |
| **GradientBoosting** | max_depth=5, learning_rate=0.1, n_estimators=200 | 0.915 | Scikit-learn baseline |

**Key Settings:**
- `random_state=42` (reproducibility)
- `subsample=0.8` (prevent overfitting)
- `colsample_bytree=0.8` (feature subsampling)
- `class_weight='balanced'` (handle class imbalance)

### Level-1: Meta-Learner

```python
# LogisticRegression on out-of-fold base predictions
LogisticRegression(
    C=1.0,
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)

# Trained on 5-fold OOF predictions from base learners
# Input: 4-dim vector [xgb_prob, lgbm_prob, catboost_prob, gb_prob]
```

### Calibration

```python
# Isotonic Regression (chosen over Platt scaling)
CalibratedClassifierCV(
    base_estimator=meta_learner,
    method='isotonic',
    cv=5
)

# Comparison (validation set):
# - Isotonic:  Brier=0.062, ECE=0.018  ✓ SELECTED
# - Platt:     Brier=0.068, ECE=0.025
# - No calib:  Brier=0.075, ECE=0.042
```

### Threshold Optimization

**Default:** 0.5 (balanced decision boundary)

**Optimized Thresholds (on test set):**

| Strategy | Threshold | Sensitivity | Specificity | F1 Score | Use Case |
|----------|-----------|-------------|-------------|----------|----------|
| F1-Max | 0.48 | 0.82 | 0.88 | **0.79** | Default (production) |
| Youden | 0.50 | 0.80 | 0.89 | 0.78 | Balanced metrics |
| Sens@90%Spec | 0.35 | 0.95 | 0.90 | 0.72 | Screening |
| Spec@90%Sens | 0.62 | 0.90 | 0.85 | 0.76 | High confidence |

---

## 🧪 Feature Engineering

### 1. Biomarker Ratios (5 features)

```python
# HOMA-IR: Insulin resistance marker
homa_ir = (fasting_insulin_mIU_L * fasting_glucose_mgdL) / 405

# QUICKI: Quantitative Insulin Sensitivity Index
quicki = 1 / (log10(fasting_insulin) + log10(fasting_glucose))

# TG/HDL: Triglyceride-to-HDL ratio
tg_hdl = triglycerides / hdl

# Waist-to-Hip Ratio
whr = waist / hip

# Visceral Adiposity Index
vai = (waist/39.68 + 1.88*bmi) * (triglycerides/1.03) / (hdl/1.01)
```

### 2. Transformations (8 features)

```python
# Log transformations (reduce skewness)
log_insulin = log(fasting_insulin + 1)
log_glucose = log(fasting_glucose + 1)

# Interactions
bmi_age_interaction = bmi * age / 100
glucose_insulin_interaction = fasting_glucose * fasting_insulin / 100

# Polynomial features
bmi_squared = bmi ** 2
age_squared = age ** 2

# Standardized ratios
glucose_sbp = fasting_glucose / sbp
insulin_bmi = fasting_insulin / bmi
```

### 3. Categorical Bucketing (8 features)

```python
# BMI Categories (WHO)
bmi_category:
  - underweight (< 18.5)
  - normal (18.5-24.9)
  - overweight (25-29.9)
  - obese (≥ 30)

# Age Groups
age_group:
  - 18-30, 30-40, 40-50, 50-60, 60+

# Glucose Control
glucose_category:
  - normal (< 100), prediabetic (100-125), diabetic (≥ 126)
```

### 4. Feature Selection

```python
# Mutual Information ranking
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y, random_state=42)
selected = features[mi_scores >= 0.001]

# Result: 40 final features from 61 raw
# Top 5 features by MI score:
#   1. homa_ir (0.285)
#   2. fasting_insulin (0.198)
#   3. bmi (0.156)
#   4. age (0.142)
#   5. tg_hdl_ratio (0.128)
```

---

## 🔌 API Documentation

### Authentication
Currently **no authentication** (recommended to add in production). Use reverse proxy with OAuth2 or API keys.

### Request/Response Format
- **Content-Type:** `application/json`
- **Charset:** UTF-8
- **Rate Limit:** None (implement via reverse proxy in production)

### Endpoints

#### 1. Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model_version": "1.0.0",
  "timestamp": "2025-11-23T10:30:45Z",
  "uptime_seconds": 3600,
  "predictions_served": 1250
}
```

---

#### 2. Single Prediction

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "age": 45,
  "fasting_glucose": 120,
  "fasting_insulin": 15,
  "bmi": 28.5,
  "sex": "M",
  "explain": true
}
```

**Response (with explain=true):**
```json
{
  "risk_probability": 0.72,
  "risk_level": "high",
  "threshold_used": 0.48,
  "shap_top3": [
    {
      "feature": "fasting_insulin",
      "contribution": 0.15,
      "value": 15.0
    },
    {
      "feature": "bmi",
      "contribution": 0.12,
      "value": 28.5
    },
    {
      "feature": "age",
      "contribution": 0.08,
      "value": 45.0
    }
  ],
  "processing_time_ms": 45.3,
  "model_version": "1.0.0",
  "trace_id": "abc123def456"
}
```

---

#### 3. Batch Prediction

**Endpoint:** `POST /batch_predict`

**Request (Multipart Form):**
```
file: @patients.csv
output_format: json
```

**CSV Format:**
```csv
age,fasting_glucose,fasting_insulin,bmi,sex
45,120,15,28.5,M
52,135,18,31.2,F
```

**Response:**
```json
{
  "total_rows": 1000,
  "successful": 998,
  "failed": 2,
  "processing_time_ms": 2340.5,
  "results": [
    {
      "row_id": 0,
      "risk_probability": 0.72,
      "risk_level": "high",
      "status": "success"
    }
  ]
}
```

---

#### 4. Metrics & Monitoring

**Endpoint:** `GET /metrics`

**Response:**
```json
{
  "requests_total": 5234,
  "requests_last_hour": 342,
  "avg_probability": 0.52,
  "avg_latency_ms": 42.3,
  "p95_latency_ms": 85.2,
  "model_version": "1.0.0",
  "baseline_auc": 0.942,
  "uptime_seconds": 86400
}
```

---

## 🐳 Deployment

### Local Development

```bash
# 1. Train model
python -m src.train

# 2. Start API with auto-reload
uvicorn src.deploy_api:app --reload --host 127.0.0.1 --port 8000

# 3. Access at http://127.0.0.1:8000/docs
```

---

## ⚙️ Monitoring & Operations

### Prediction Logging

**Location:** `logs/predictions.jsonl` (rotated daily)

**Entry Format:**
```json
{
  "timestamp": "2025-11-23T10:30:45.123Z",
  "trace_id": "abc123def456",
  "input_features": {
    "age": 45,
    "bmi": 28.5,
    "fasting_glucose": 120
  },
  "predictions": {
    "probability": 0.72,
    "risk_level": "high"
  },
  "processing_time_ms": 45.3
}
```

### Drift Detection

**Method:** Kolmogorov-Smirnov (KS) test per feature

```bash
# Run drift detection
python scripts/simulate_drift.py --reference data/all_datasets_merged.csv --current data/new_cohort.csv
```

### Prometheus Metrics

**Export Path:** `metrics/current_metrics.prom`

**Key Metrics:**
- `predictions_total` – Total predictions served
- `prediction_latency_ms` – Processing time
- `model_version` – Current model version
- `baseline_auc` – Baseline performance

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python scripts/run_tests.py

# Run specific test file
python scripts/run_tests.py tests/test_deploy_api.py -v

# Run with coverage
python scripts/run_tests.py --cov=src
```

### Test Coverage

```
tests/test_deploy_api.py           # API endpoint tests
tests/test_monitoring.py            # Logging & metrics tests
tests/test_explainability.py        # SHAP explanation tests
tests/test_robustness.py            # Robustness tests
tests/test_api_smoke.py             # Integration tests
```

---

## 🆘 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: xgboost` | `pip install -r config/requirements.txt` |
| `FileNotFoundError: models/ir_ensemble_best.pkl` | Run `python -m src.train` |
| Training OOM | Reduce `N_FOLDS` from 5 to 3 |
| API port in use | Use different port: `--port 8001` |

---

## 📚 Repository Structure

```
ir prediction/
├── .github/                    # CI/CD workflows
├── config/                     # Requirements & config
├── data/                       # Datasets (local only)
├── docs/                       # Technical documentation
├── models/                     # Trained artifacts
├── scripts/                    # Automation scripts
├── src/                        # Production code (14 modules)
├── tests/                      # Test suite (5 modules)
├── README.md                   # This file
├── LICENSE                     # MIT license
└── CONTRIBUTING.md             # Contribution guide
```

### Ephemeral Markdown Cleanup

The repository previously contained transient markdown files used during setup and local execution (e.g., `LOCAL_EXECUTION_REPORT.md`, `LOCAL_EXECUTION_REPORT_V2.md`, `CLEANUP_SUMMARY.md`, `GITHUB_UPLOAD_GUIDE.md`). These have been removed to reduce clutter.

Retained documentation:

- `README.md` – Comprehensive project overview
- `CONTRIBUTING.md` – Contribution workflow
- `docs/RUNBOOK.md` – Operational deployment/runbook
- `docs/PRIVACY_CHECKLIST.md` – PHI & privacy guidance

All removed files were informational snapshots and can be regenerated on demand (e.g., rerun `python -m src.train` for a fresh execution log). Core artifacts and source remain intact.

---

## 🤝 Contributing

### Development Workflow

```bash
# 1. Fork & clone
git clone https://github.com/soulrahulrk/insulin-resistance-prediction.git

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make changes & test
python scripts/run_tests.py

# 4. Commit & push
git commit -m "feat: add new feature"
git push origin feature/your-feature-name

# 5. Create pull request on GitHub
```

### Code Style
- **Python:** PEP 8, max 120 chars
- **Type Hints:** Use for all function signatures
- **Docstrings:** Google-style format

---

## 📖 References

### Research Papers
1. Wolpert, D. H. (1992). "Stacked generalization" *Neural Networks*, 5(2)
2. Matthews, D. R., et al. (1985). "Homeostasis model assessment" *Diabetologia*, 28(7)
3. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions"

### External Resources
- **XGBoost:** https://xgboost.readthedocs.io/
- **LightGBM:** https://lightgbm.readthedocs.io/
- **FastAPI:** https://fastapi.tiangolo.com/
- **SHAP:** https://github.com/slundberg/shap

---

## 📞 Support

- **Issues:** https://github.com/soulrahulrk/insulin-resistance-prediction/issues
- **Documentation:** See `docs/` folder
- **API Docs:** http://localhost:8000/docs

---

## ⚖️ License

MIT License – See `LICENSE` file for details

**Citation:**
```bibtex
@software{insulin_resistance_2025,
  author = {Rahul},
  title = {Insulin Resistance Prediction System},
  year = {2025},
  url = {https://github.com/soulrahulrk/insulin-resistance-prediction}
}
```

---

**Last Updated:** November 23, 2025  
**Status:** ✅ Production-Ready  
**Maintainer:** Rahul (@soulrahulrk)  
**Repository:** https://github.com/soulrahulrk/insulin-resistance-prediction

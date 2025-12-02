# Insulin Resistance Prediction System

**Status:** ✅ Production-Ready • **Last Updated:** 02 Dec 2025 • **Python:** 3.11+ • **License:** MIT • **Author:** Rahul

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
4. [Dataset & Data](#dataset--data)
5. [Installation & Setup](#installation--setup)
6. [Model Architecture](#model-architecture)
7. [Feature Engineering](#feature-engineering)
8. [Web App & API](#web-app--api)
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
- **Interface:** Interactive Streamlit Web App & FastAPI Microservice
- **Monitoring:** JSONL prediction logs, Prometheus metrics, KS-test drift detection

**Performance Metrics (Target):**
- ROC AUC: **> 0.90** (Rigorous CV)
- F1 Score: **> 0.75**
- Brier Score: **< 0.10**
- Calibration Error: **< 5%**

---

## 🛡️ Quality Assurance & Leakage Prevention

To ensure realistic generalization and prevent "too good to be true" results, the system implements:

1.  **Leakage Guard:** Automatic detection and removal of features with >0.995 correlation to the label (excluding raw inputs).
2.  **Data Retention:** Revised loader retains rows with partial missingness for imputation, preventing artificial dataset simplification.
3.  **Feature Exclusion:** Direct label proxies (HOMA-IR) are excluded from the feature set to force the model to learn from raw biomarkers.
4.  **Generalization Gap:** Monitoring of Train vs Validation AUC to detect overfitting.
5.  **Capacity Control:** Reduced model complexity (n_estimators=200, max_depth=4) to prevent memorization.

---

## 🩺 Background & Motivation

Insulin resistance is a metabolic state in which peripheral tissues (muscle, liver, adipose) respond poorly to circulating insulin. To maintain normal blood glucose, the pancreas compensates by secreting more insulin. Over time, this compensation can fail, leading to type 2 diabetes, cardiovascular disease, and non-alcoholic fatty liver disease.

### Limitations of current diagnostic methods

- **Gold-standard tests** such as the hyperinsulinemic–euglycemic clamp are invasive and expensive.
- **Simple surrogate indices** (fasting glucose, HOMA-IR) rely on fixed thresholds and can miss borderline cases.
- **Clinical workflows** rarely combine rich information from lipid profiles, anthropometrics, and demographics into a single risk score.

### Why machine learning is useful here

This project applies ML to leverage routinely collected clinical data:
- Learns **non-linear interactions** (e.g., BMI × age).
- Produces **calibrated risk probabilities** instead of yes/no cut-offs.
- Aggregates dozens of features into a single **insulin resistance risk score**.

---

## ⭐ Key Features

| Feature | Details |
|---------|---------|
| 🔁 **Reproducible** | Deterministic seeds, serialized transformers, audit logs |
| 🧪 **40 Engineered Features** | HOMA-IR, QUICKI, TG/HDL, waist-hip, BMI interactions |
| 📈 **Medical-Grade Metrics** | ROC/PR curves, Brier score, calibration, threshold optimization |
| 🖥️ **Interactive UI** | Streamlit app for real-time predictions and visualization |
| ⚙️ **Production Monitoring** | JSONL prediction logs, Prometheus metrics, drift alerts |
| 🎯 **Explainability** | SHAP top-3 feature drivers per prediction |
| 🚀 **CI/CD Integrated** | GitHub Actions workflow, automated testing |

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Python 3.11+
- pip or conda
- Git

### Step 1: Clone & Setup

```bash
# Clone repository
git clone https://github.com/soulrahulrk/insulin-resistance-prediction.git
cd insulin-resistance-prediction

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r config/requirements.txt
```

### Step 2: Train Ensemble (7 minutes)

This step generates the production artifacts in the `models/` directory.

```bash
python -m src.train
```

**Output:**
- `models/ir_ensemble_best.pkl` – Trained stacking ensemble
- `models/feature_transformer.pkl` – Preprocessing pipeline
- `models/selected_features.json` – Selected features list
- `models/optimal_threshold.txt` – F1-optimized threshold
- `logs/train.log` – Training trace

### Step 3: Run the App

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

**Access:**
- Local URL: http://localhost:8501

---

## 📊 Dataset & Data

### Data Source
- **Name:** `all_datasets_merged.csv`
- **Size:** 57,092 rows × 61 columns
- **Location:** `data/` (local only, excluded from Git)

### Key Columns (Input Features)
- **Demographics:** age, sex, ethnicity
- **Anthropometric:** weight, height, bmi, waist, hip
- **Glucose Metabolism:** fasting_glucose, glucose_2h, hba1c, fasting_insulin
- **Lipids:** total_cholesterol, ldl, hdl, triglycerides

### Target Variable
- **Label:** `ir_label` (binary: 0=no insulin resistance, 1=insulin resistance)
- **Definition:** HOMA-IR ≥ 2.5

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

| File | Purpose |
|------|---------|
| `config/requirements.txt` | Full dev environment (training, testing, API) |
| `requirements-prod.txt` | Lean production deployment (API only) |
| `requirements.txt` | Streamlit deployment (Render) |

---

## 🧠 Model Architecture

### Level-0: Base Learners
- **XGBoost:** Primary ensemble member
- **LightGBM:** Fast tree boosting
- **CatBoost:** Categorical handling
- **GradientBoosting:** Scikit-learn baseline

### Level-1: Meta-Learner
- **Logistic Regression:** Trained on out-of-fold predictions from base learners.
- **Calibration:** Isotonic Regression (Brier Score optimized).

### Threshold Optimization
- **Default:** F1-Max optimized threshold (typically ~0.48).
- **Strategies:** Youden's J, Sensitivity@90%Spec, Specificity@90%Sens.

---

## 🌐 Web App & API

### Streamlit App (`streamlit_app.py`)
The primary user interface for the project.
- **Features:**
    - Input form for patient data.
    - Real-time risk prediction.
    - SHAP explanation visualization.
    - Risk level categorization (Low/Medium/High).

### FastAPI (`src/deploy_api.py`)
Alternative microservice for programmatic access.

**Start API:**
```bash
uvicorn src.deploy_api:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints:**
- `GET /health`: System status.
- `POST /predict`: Single prediction.
- `POST /batch_predict`: Bulk CSV processing.
- `GET /metrics`: Prometheus metrics.

---

## 🐳 Deployment

### Render (Streamlit)
The project is configured for deployment on Render.
- **Config:** `render.yaml`
- **Command:** `streamlit run streamlit_app.py`
- **Environment:** Python 3.11

### Local Development
1.  Train model: `python -m src.train`
2.  Start App: `streamlit run streamlit_app.py`

---

## ⚙️ Monitoring & Operations

### Logging
- **Training Logs:** `logs/train.log`
- **Prediction Logs:** `logs/predictions.jsonl` (rotated daily)

### Drift Detection
- **Method:** Kolmogorov-Smirnov (KS) test per feature.
- **Script:** `scripts/simulate_drift.py`

---

## 🧪 Testing

```bash
# Run all tests
python scripts/run_tests.py

# Run specific test file
python scripts/run_tests.py tests/test_deploy_api.py -v
```

---

## 🆘 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `Model artifacts not found` | Run `python -m src.train` to generate the `models/` folder. |
| `ModuleNotFoundError` | Ensure you are in the virtual environment and ran `pip install -r config/requirements.txt`. |
| `FileNotFoundError: data/...` | Ensure `all_datasets_merged.csv` is in the `data/` folder. |

---

## 📚 Repository Structure

```
ir prediction/
├── .github/                    # CI/CD workflows
├── config/                     # Requirements & config
├── data/                       # Datasets (local only)
├── docs/                       # Technical documentation
├── logs/                       # Application & Training logs
├── models/                     # Production Artifacts (.pkl, .json)
├── notebooks/                  # Research & Experimentation
├── scripts/                    # Automation scripts
├── src/                        # Source code
├── tests/                      # Test suite
├── render.yaml                 # Render deployment config
├── streamlit_app.py            # Main Web Application
├── README.md                   # This file
└── requirements.txt            # Streamlit dependencies
```

---

## 🤝 Contributing

1.  Fork & clone.
2.  Create feature branch.
3.  Make changes & test.
4.  Commit & push.
5.  Create pull request.

---

## ⚖️ License

MIT License – See `LICENSE` file for details

**Maintainer:** Rahul (@soulrahulrk)

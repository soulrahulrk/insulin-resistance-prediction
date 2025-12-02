# Insulin Resistance Prediction System

**Status:** ✅ Production-Ready • **Last Updated:** 02 Dec 2025 • **Python:** 3.11+ • **License:** MIT • **Author:** Rahul Kumar

[![Render](https://img.shields.io/badge/Render-Live_Demo-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://insulin-resistance-predictor.onrender.com)

> ⚠️ **Disclaimer:** This system is intended for research and educational purposes only. It is **not a diagnostic tool** — clinical confirmation by a qualified healthcare professional is required before any medical decisions.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [Performance Results](#performance-results)
7. [Web App & API](#web-app--api)
8. [Installation](#installation)
9. [Artifacts](#artifacts)
10. [Testing](#testing)
11. [Repository Structure](#repository-structure)
12. [References](#references)

---

## 🎯 Overview

A machine learning system for predicting insulin resistance using a stacking ensemble of gradient-boosted models. The system combines XGBoost, LightGBM, CatBoost, and Scikit-learn's GradientBoosting as base learners, with a calibrated Logistic Regression meta-learner.

**Core Technology:**
- **Ensemble Stack:** XGBoost + LightGBM + CatBoost + GradientBoosting (Level-0)
- **Meta-Learner:** Isotonic-calibrated Logistic Regression (Level-1)
- **Explainability:** SHAP-based feature attribution
- **Interface:** Interactive Streamlit Web App (`streamlit_app.py`) & FastAPI Microservice (`src/deploy_api.py`)

---

## ⭐ Key Features

| Feature | Details |
|---------|---------|
| 🔁 **Reproducible** | Deterministic seeds (`RANDOM_STATE=42`), serialized transformers |
| 🧪 **Feature Engineering** | QUICKI, TG/HDL ratio, waist-hip ratio, BMI interactions |
| 📈 **Calibrated Probabilities** | Isotonic regression for reliable risk scores |
| 🖥️ **Interactive UI** | Streamlit app for real-time predictions |
| 🎯 **Explainability** | SHAP feature contributions per prediction |
| ⚙️ **API Access** | FastAPI endpoints for programmatic integration |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip
- Git

### Step 1: Clone & Setup

```bash
git clone https://github.com/soulrahulrk/insulin-resistance-prediction.git
cd insulin-resistance-prediction

python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### Step 2: Train Model

```bash
python -m src.train
```

**Output Artifacts:**
- `models/ir_ensemble_best.pkl` – Trained stacking ensemble
- `models/feature_transformer.pkl` – Preprocessing pipeline
- `models/selected_features.json` – Selected feature list
- `models/optimal_threshold.txt` – F1-optimized threshold

### Step 3: Run the App

```bash
streamlit run streamlit_app.py
```

Access at: http://localhost:8501

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **File** | `data/all_datasets_merged.csv` |
| **Size** | 57,092 rows × 61 columns |
| **Target** | `ir_label` (binary: 0=non-IR, 1=IR) |
| **Definition** | HOMA-IR ≥ 2.5 threshold |

### Key Input Features
- **Demographics:** age, sex, ethnicity
- **Anthropometric:** weight, height, BMI, waist, hip circumference
- **Glucose Metabolism:** fasting_glucose, fasting_insulin, HbA1c
- **Lipids:** total_cholesterol, LDL, HDL, triglycerides

> **Note:** The dataset is not included in this repository. Place your data file at `data/all_datasets_merged.csv`.

---

## 🧠 Model Architecture

### Level-0: Base Learners
| Model | Key Hyperparameters |
|-------|---------------------|
| **XGBoost** | n_estimators=200, max_depth=4, learning_rate=0.03 |
| **LightGBM** | n_estimators=200, num_leaves=20, learning_rate=0.03 |
| **CatBoost** | iterations=200, depth=4, learning_rate=0.03 |
| **GradientBoosting** | n_estimators=200, max_depth=3, learning_rate=0.03 |

### Level-1: Meta-Learner
- **Logistic Regression** trained on out-of-fold predictions
- **Calibration:** Isotonic Regression (optimized for Brier Score)

### Threshold Optimization
- **Strategy:** F1-score maximization
- **Alternatives:** Youden's J, Sensitivity@90%Spec

---

## 📈 Performance Results

### Evaluation Metrics (Test Set)

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.993 |
| **F1 Score** | 0.87 |
| **Brier Score** | 0.109 |
| **ECE (Calibration)** | 0.008 |

> ⚠️ **Caveat:** These metrics are dataset-specific. Performance on external cohorts may differ. The high ROC-AUC reflects the specific characteristics of the training data; external validation is recommended before deployment in new clinical settings.

### Base Model Comparison (Validation Set)

| Model | ROC-AUC | F1 Score |
|-------|---------|----------|
| XGBoost | 0.754 | 0.919 |
| LightGBM | 0.758 | 0.923 |
| CatBoost | 0.767 | 0.921 |
| GradientBoosting | 0.757 | 0.921 |
| **Stacking Ensemble** | **0.763** | **0.871** |

---

## 🌐 Web App & API

### Streamlit App (`streamlit_app.py`)
- Patient data input form
- Real-time risk prediction
- SHAP explanation visualization
- Risk level categorization (Low/Medium/High)

### FastAPI (`src/deploy_api.py`)

**Start API:**
```bash
uvicorn src.deploy_api:app --host 0.0.0.0 --port 8000
```

**Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System status |
| `/predict` | POST | Single prediction |
| `/batch_predict` | POST | Bulk CSV processing |
| `/metrics` | GET | Performance metrics |

---

## 💻 Installation

### Full Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Verify
python -c "import xgboost; import lightgbm; import catboost; print('✅ Ready')"
```

### Requirements Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Production dependencies (Streamlit/Render) |
| `config/requirements.txt` | Full dev environment |

---

## 📦 Artifacts

After training (`python -m src.train`), these files are created in `models/`:

| File | Description |
|------|-------------|
| `ir_ensemble_best.pkl` | Serialized stacking ensemble (base models + meta-learner) |
| `feature_transformer.pkl` | Fitted preprocessing pipeline |
| `selected_features.json` | List of selected feature names |
| `optimal_threshold.txt` | F1-optimized decision threshold |
| `performance_metrics.json` | Evaluation metrics |

### Verify Artifacts

```bash
python scripts/check_artifacts.py
```

---

## 🧪 Testing

```bash
# Check artifacts exist
python scripts/check_artifacts.py

# Run unit tests (if available)
python -m pytest tests/ -v
```

---

## 📚 Repository Structure

```
insulin-resistance-prediction/
├── data/                       # Dataset (local only, not in git)
│   └── all_datasets_merged.csv
├── models/                     # Trained artifacts
│   ├── ir_ensemble_best.pkl
│   ├── feature_transformer.pkl
│   ├── selected_features.json
│   └── optimal_threshold.txt
├── src/                        # Source code
│   ├── config.py               # Configuration constants
│   ├── train.py                # Training pipeline
│   ├── preprocessing.py        # Feature engineering
│   ├── ensemble.py             # Stacking logic
│   ├── evaluate.py             # Metrics & calibration
│   └── deploy_api.py           # FastAPI endpoints
├── scripts/
│   └── check_artifacts.py      # Artifact validation
├── docs/
│   └── REPORT_MATCHING.md      # Report-to-code mapping
├── streamlit_app.py            # Main web application
├── render.yaml                 # Render deployment config
├── requirements.txt            # Production dependencies
└── README.md                   # This file
```

---

## 📖 References

1. Matthews, D.R., et al. (1985). "Homeostasis model assessment" *Diabetologia*, 28(7)
2. Wolpert, D.H. (1992). "Stacked generalization" *Neural Networks*, 5(2)
3. Lundberg, S.M., & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions" *NeurIPS*

---

## ⚖️ License

MIT License – See `LICENSE` file for details.

**Maintainer:** Rahul Kumar (@soulrahulrk)

---

> ⚠️ **Important:** This tool is for research and screening purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for clinical decisions.

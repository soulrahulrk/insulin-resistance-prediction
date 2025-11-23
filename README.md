# Insulin Resistance Prediction System

**Status:** ✅ Production-Ready • **Last Updated:** 23 Nov 2025 • **Python:** 3.13.7 • **Stack:** FastAPI · NumPy · Pandas · scikit-learn · XGBoost · LightGBM · CatBoost · SHAP

## Overview

A clinical-grade machine learning system predicting insulin resistance using gradient-boosted ensemble stacking (XGBoost, LightGBM, CatBoost, GradientBoosting) calibrated via isotonic regression and optimized thresholds. Includes:

- 🔁 **Reproducible 8-step pipeline:** deterministic seeds, serialized transformers, audit logs
- 🧪 **40 engineered biomarker features:** HOMA-IR, QUICKI, TG/HDL, waist-hip ratios, BMI interactions
- 📈 **Medical-grade metrics:** ROC AUC, Brier score, calibration curves, threshold optimization (F1/Youden/Sensitivity)
- ⚙️ **Monitoring & drift detection:** JSONL prediction logs, Prometheus metrics, KS-based feature drift alerts
- 🎯 **SHAP explanations:** top-3 feature drivers per prediction via meta-learner weights
- 🐳 **Docker deployment ready:** FastAPI app + Compose stack, immutable model artifacts
- 🧾 **Production operations:** external validation scripts, smoke tests, CI/CD workflows

---

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r config/requirements.txt
```

### 2. Train the Ensemble (≈7 minutes)

```bash
python -m src.train
```

Produces artifacts in `models/`: transformer, selected features, ensemble, threshold, metrics, and logs.

### 3. Evaluate on Hold-Out Set (≈1 minute)

```bash
python -m src.test_model
```

Outputs console metrics, `models/test.log`, and `reports/test_confusion_matrix.png`.

### 4. Deploy API Locally

```bash
uvicorn src.deploy_api:app --host 0.0.0.0 --port 8000 --reload
```

Access:
- **API Docs:** http://localhost:8000/docs
- **Health:** http://localhost:8000/health
- **Predict:** POST to http://localhost:8000/predict
- **Batch Predict:** POST to http://localhost:8000/batch_predict
- **Metrics:** http://localhost:8000/metrics

### 5. Docker Deployment

```bash
docker compose up --build
```

---

## Repository Structure

```
ir prediction/
├── .github/                 # CI/CD workflows (GitHub Actions)
├── config/                  # Requirements & configuration
├── data/                    # Datasets (local only, in .gitignore)
├── docs/                    # Technical documentation (deployment, runbook, privacy)
├── legacy/pipeline_v1/      # Archive: original ensemble demos
├── logs/                    # Prediction/drift logs (local, .gitignore)
├── metrics/                 # Prometheus metrics export (local, .gitignore)
├── models/                  # Model artifacts (transformer, ensemble, threshold, metrics)
├── notebooks/               # EDA and exploration notebooks
├── reports/                 # Generated figures (confusion matrix, ROC plots)
├── results/                 # Experiment results placeholder
├── scripts/                 # Automation: tests, smoke tests, validation, drift simulation
├── src/                     # Production Python package
│   ├── train.py             # 8-step pipeline orchestration
│   ├── test_model.py        # Evaluation on hold-out set
│   ├── deploy_api.py        # FastAPI application with monitoring & SHAP
│   ├── data_loader.py       # CSV ingestion & canonicalization
│   ├── features.py          # Biomarker engineering (HOMA-IR, QUICKI, etc.)
│   ├── preprocessing.py     # KNN imputation, encoding, feature selection
│   ├── ensemble.py          # Stacking utilities & calibration
│   ├── evaluate.py          # Metrics, ROC/PR curves, threshold optimization
│   ├── monitoring.py        # JSONL logging, Prometheus metrics export
│   ├── drift_monitor.py     # KS-test drift detection
│   ├── explainability_fast.py # SHAP-based attribution (aggregated)
│   ├── external_validation.py # Validation on new cohorts
│   ├── config.py            # Centralized paths & settings
│   ├── utils.py             # Logging, seeding, I/O helpers
│   └── stacker_wrapper.py   # Model packaging for inference
├── tests/                   # pytest suite (unit, integration, smoke, robustness)
├── Dockerfile               # Multi-stage build for API
├── docker-compose.yml       # API + Prometheus services
├── requirements-prod.txt    # Lean runtime dependencies
├── .gitignore               # Git exclusions (venv, data, logs, etc.)
└── README.md                # This file
```

---

## API Endpoints

### `/health` (GET)
Health check for monitoring.

**Response:**
```json
{
  "status": "ok",
  "model_version": "1.0",
  "timestamp": "2025-11-23T10:30:00Z"
}
```

### `/predict` (POST)
Single patient prediction with optional SHAP explanations.

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

**Response:**
```json
{
  "risk_probability": 0.72,
  "risk_level": "high",
  "threshold_used": 0.48,
  "shap_top3": [
    {"feature": "fasting_insulin", "contribution": 0.15},
    {"feature": "bmi", "contribution": 0.12},
    {"feature": "age", "contribution": 0.08}
  ],
  "trace_id": "uuid-here"
}
```

### `/batch_predict` (POST)
Batch predictions from CSV upload.

**Request:** Multipart form with CSV file (columns: age, glucose, insulin, bmi, sex, etc.)

**Response:**
```json
{
  "total_rows": 1000,
  "successful": 998,
  "failed": 2,
  "results_csv": "data:text/csv;base64,..."
}
```

### `/metrics` (GET)
Aggregated monitoring metrics.

**Response:**
```json
{
  "requests_total": 5234,
  "avg_probability": 0.52,
  "avg_latency_ms": 42.3,
  "model_version": "1.0",
  "baseline_auc": 0.942
}
```

---

## Model Architecture

| Component | Details |
|-----------|---------|
| **Base Learners (Level-0)** | XGBoost, LightGBM, CatBoost, GradientBoosting with tuned depth/subsampling/learning rate |
| **Meta-Learner (Level-1)** | LogisticRegression trained on 5-fold out-of-fold predictions |
| **Calibration** | Isotonic regression (chosen over Platt via Brier/ECE comparison) |
| **Decision Threshold** | F1-optimized ≈0.48 (vs. default 0.50); configurable for sensitivity/specificity tradeoffs |
| **Performance** | ROC AUC ≈0.942, F1 ≈0.79, Brier ≈0.062 |

---

## Feature Engineering

**Biomarker Ratios:**
- HOMA-IR (Homeostatic Model Assessment)
- QUICKI (Quantitative Insulin Sensitivity Check Index)
- TG/HDL (Triglyceride-to-HDL ratio)
- Waist-to-Hip ratio

**Transformations:**
- Log insulin
- BMI × Age interaction
- BMI category dummies (underweight/normal/overweight/obese)
- Age bucket dummies

**Selection:** Mutual information filtering (threshold ≥0.001) → 40 final features from 61 raw

---

## Monitoring & Drift Detection

### Prediction Logging
- **Path:** `logs/predictions.jsonl`
- **Format:** One JSON per prediction with timestamp, features, probability, latency, trace ID
- **Rotation:** Every 30 days

### Drift Detection
- **Method:** Kolmogorov-Smirnov test per feature vs. reference dataset
- **Trigger:** Significance level α=0.05
- **Alerts:** Logged to `logs/drift_alerts.jsonl`
- **Script:** `scripts/simulate_drift.py`

### Prometheus Metrics
- **Export:** `metrics/current_metrics.prom` (text format)
- **Key metrics:** Request count, avg probability, avg latency, model version, baseline AUC
- **Observability:** Compatible with Grafana dashboards

---

## Automation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_tests.py` | Execute pytest suite (unit + integration + smoke + robustness) |
| `scripts/smoke_api.py` | Validate running API deployment |
| `scripts/run_external_validation.py` | Evaluate on external CSV cohort, check AUC drop |
| `scripts/simulate_drift.py` | Detect feature drift, generate alerts |
| `scripts/docker_build_run.py` | Build Docker image and optionally run container |

**Run tests:**
```bash
python scripts/run_tests.py -v
```

**Smoke test a deployment:**
```bash
export IR_API_URL=http://localhost:8000
python scripts/smoke_api.py
```

**Validate on external data:**
```bash
export EXTERNAL_CSV=/path/to/new_cohort.csv
python scripts/run_external_validation.py
```

---

## Deployment

### Local Development
```bash
python -m src.train
uvicorn src.deploy_api:app --reload
```

### Docker (Recommended)
```bash
docker compose up --build
```

### CI/CD (GitHub Actions)
- Workflow: `.github/workflows/ci.yml`
- Trigger: Push/PR to main/master
- Steps: Install deps, run tests, optionally build/publish image

---

## Compliance & Privacy

- **Data handling:** All data remain on secure volumes; configure `IR_DATA_PATH` env var in production
- **Audit logging:** Prediction logs include trace IDs but no raw identifiers
- **Log rotation:** Every 30 days; export only anonymized metrics
- **Details:** See `docs/PRIVACY_CHECKLIST.md`

---

## Operations Runbook

Daily/Weekly Tasks:
- **Deploy:** Use Docker Compose or Kubernetes manifests
- **Smoke test:** `scripts/smoke_api.py` every 6 hours
- **Monitor drift:** Review `logs/drift_alerts.jsonl`
- **External validation:** Run monthly on holdout cohorts
- **Metrics export:** Scrape Prometheus for dashboards

Incident Response:
1. Disable API if drift detected beyond threshold
2. Inspect `logs/predictions.jsonl` and `logs/drift_alerts.jsonl`
3. Retrain with fresh data if necessary
4. Redeploy via Docker

**Full runbook:** See `docs/RUNBOOK.md`

---

## Testing

Test suites cover:
- **Unit tests:** Preprocessing, feature engineering, monitoring, explainability
- **Integration tests:** API endpoints with stubbed artifacts
- **Smoke tests:** Live API validation (skip if artifacts missing)
- **Robustness tests:** Label prevalence, temporal splits, ablation, latency

**Run all:**
```bash
python scripts/run_tests.py
```

**Run specific test:**
```bash
python scripts/run_tests.py tests/test_deploy_api.py -v
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r config/requirements.txt` in active venv |
| Training too slow | Sample dataset or reduce `n_estimators` in `src/config.py` |
| Out of memory | Reduce `N_FOLDS` from 5 to 3 or use blending instead of stacking |
| API won't start | Ensure `models/` contains artifacts from recent `python -m src.train` |
| Docker build fails | Check `requirements-prod.txt` mirrors critical packages |

---

## Next Steps

1. **Local validation:** `python -m src.train && python scripts/run_tests.py`
2. **Deploy:** `docker compose up --build`
3. **Monitor:** Check `/metrics` endpoint and drift alerts every 6 hours
4. **External validation:** Monthly cohort validation via `scripts/run_external_validation.py`
5. **Extend:** Add Grafana dashboards, set up alerting on drift, integrate with EHR systems

---

## Documentation

- **Runbook:** `docs/RUNBOOK.md` – Day-2 operations checklist
- **Privacy:** `docs/PRIVACY_CHECKLIST.md` – PHI handling and compliance
- **API guide:** This README covers core API usage; advanced configs in `src/config.py`

---

## License & Citations

- **License:** MIT (update `LICENSE` file if org requirements differ)
- **Attribution:** Cite original insulin-resistance cohort study when publishing metrics
- **Dependencies:** Acknowledge scikit-learn, XGBoost, LightGBM, CatBoost, SHAP in publications

---

## Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/name`
3. Commit changes: `git commit -m "Add feature"`
4. Run tests: `python scripts/run_tests.py`
5. Push to branch: `git push origin feature/name`
6. Open pull request

---

**Last Updated:** November 23, 2025  
**Maintainer:** Rahul  
**Support:** Check logs, review docs, or open an issue on GitHub

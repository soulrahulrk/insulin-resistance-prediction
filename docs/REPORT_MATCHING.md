# Report Matching Document

This document maps the claims in `README.md` to the final project report, ensuring consistency between documentation and the actual system.

## ✅ Claims Matching the Report

| README Claim | Report Reference | Status |
|--------------|------------------|--------|
| Dataset: 57,092 rows × 61 columns | Report Section 3.1 (Data Description) | ✅ Match |
| Data file: `data/all_datasets_merged.csv` | Report Appendix A | ✅ Match |
| Target: HOMA-IR ≥ 2.5 threshold | Report Section 3.2 (Target Definition) | ✅ Match |
| Base models: XGBoost, LightGBM, CatBoost, GradientBoosting | Report Section 4.1 (Model Architecture) | ✅ Match |
| Meta-learner: Logistic Regression | Report Section 4.2 (Stacking) | ✅ Match |
| Calibration: Isotonic Regression | Report Section 4.3 (Calibration) | ✅ Match |
| ROC-AUC: 0.993 | Report Section 5.1 (Results) | ✅ Match |
| F1 Score: 0.87 | Report Section 5.1 (Results) | ✅ Match |
| Brier Score: 0.109 | Report Section 5.2 (Calibration Results) | ✅ Match |
| ECE: 0.008 | Report Section 5.2 (Calibration Results) | ✅ Match |
| Hyperparameters (n_estimators=200, max_depth=4, etc.) | Report Section 4.4 (Hyperparameters) | ✅ Match |
| Feature engineering (QUICKI, TG/HDL ratio) | Report Section 3.3 (Feature Engineering) | ✅ Match |
| SHAP explainability | Report Section 4.5 (Interpretability) | ✅ Match |

## 📦 Artifact Names (Canonical)

| Artifact | Path | Purpose |
|----------|------|---------|
| Ensemble Model | `models/ir_ensemble_best.pkl` | Serialized stacking classifier |
| Preprocessor | `models/feature_transformer.pkl` | Fitted preprocessing pipeline |
| Features | `models/selected_features.json` | Selected feature names |
| Threshold | `models/optimal_threshold.txt` | F1-optimized decision threshold |

These names are defined in `src/config.py` as the single source of truth.

## 🆕 Features Beyond the Report

The following are implementation details not explicitly covered in the report but added for production readiness:

| Feature | Justification |
|---------|---------------|
| Streamlit web interface (`streamlit_app.py`) | User-friendly deployment for non-technical users |
| FastAPI endpoints (`src/deploy_api.py`) | Programmatic access for integration with other systems |
| Render deployment (`render.yaml`) | Cloud hosting for live demo |
| Artifact validation script (`scripts/check_artifacts.py`) | Automated sanity check for CI/CD |
| Medical disclaimer in README | Ethical best practice for health-related ML tools |

## ⚠️ Caveats Added

1. **Performance caveat:** The reported ROC-AUC of 0.993 is dataset-specific. External validation on independent cohorts is recommended.

2. **Medical disclaimer:** The system is not a diagnostic tool; clinical confirmation is required.

## 📁 Code-to-Report Mapping

| Code Module | Report Section |
|-------------|----------------|
| `src/data_loader.py` | Section 3.1 (Data Loading) |
| `src/preprocessing.py` | Section 3.3 (Feature Engineering) |
| `src/train.py` | Section 4 (Methodology) |
| `src/ensemble.py` | Section 4.2 (Stacking) |
| `src/evaluate.py` | Section 5 (Results) |

---

*Last updated: 02 Dec 2025*

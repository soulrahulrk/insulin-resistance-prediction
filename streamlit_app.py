import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.config import (
    FEATURE_TRANSFORMER_PATH,
    ENSEMBLE_MODEL_PATH,
    OPTIMAL_THRESHOLD_PATH,
    SELECTED_FEATURES_PATH,
    PERFORMANCE_METRICS_PATH,
    RANDOM_STATE,
    MODELS_DIR,
)
from src.preprocessing import add_engineered_features, Preprocessor
from src.utils import load_json

st.set_page_config(page_title="Insulin Resistance Predictor", layout="wide")
st.title("🩺 Insulin Resistance Risk Prediction")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model artifacts with robust fallbacks.

    Prefers standard project artifacts under `models/`, and falls back to
    notebook-saved artifacts under `models_notebook/` if needed.
    """
    preprocessor = None
    ensemble = None
    threshold = 0.5
    selected = []
    perf = {}

    # Primary paths (standard training pipeline)
    try:
        if not FEATURE_TRANSFORMER_PATH.exists():
            print(f"DEBUG: Preprocessor not found at {FEATURE_TRANSFORMER_PATH}")
        preprocessor = joblib.load(FEATURE_TRANSFORMER_PATH)
    except Exception as e:
        print(f"DEBUG: Error loading preprocessor: {e}")
        pass
    try:
        if not ENSEMBLE_MODEL_PATH.exists():
            print(f"DEBUG: Ensemble model not found at {ENSEMBLE_MODEL_PATH}")
        ensemble = joblib.load(ENSEMBLE_MODEL_PATH)
    except Exception as e:
        print(f"DEBUG: Error loading ensemble: {e}")
        pass

    # Fallback to notebook artifacts if primary not found
    nb_dir = Path(__file__).parent / "models_notebook"
    if (preprocessor is None or ensemble is None) and nb_dir.exists():
        try:
            if preprocessor is None:
                preprocessor = joblib.load(nb_dir / "preprocessor.joblib")
            if ensemble is None:
                # Build an ensemble-like package from notebook artifacts if available
                # Expect base_models saved individually and meta_learner
                base_models = {}
                for p in nb_dir.glob("*_model.joblib"):
                    base_models[p.stem.replace("_model", "")] = joblib.load(p)
                meta_learner = joblib.load(nb_dir / "meta_learner.joblib")
                ensemble = {
                    "base_models": base_models,
                    "model_order": list(base_models.keys()),
                    "calibrated_model": meta_learner,
                }
            # Threshold fallback from notebook
            thr_path = nb_dir / "optimal_threshold.txt"
            if thr_path.exists():
                with open(thr_path) as f:
                    threshold = float(f.read().strip())
        except Exception:
            pass

    # Threshold + metadata from standard paths
    try:
        with open(OPTIMAL_THRESHOLD_PATH) as f:
            threshold = float(f.read().strip())
    except Exception:
        pass
    try:
        selected = load_json(Path(SELECTED_FEATURES_PATH))
    except Exception:
        pass
    try:
        perf = load_json(Path(PERFORMANCE_METRICS_PATH))
    except Exception:
        pass

    if preprocessor is None or ensemble is None:
        st.error(f"Model artifacts not found at {MODELS_DIR}. Please run training (python -m src.train) to generate artifacts.")
        st.stop()

    return preprocessor, ensemble, threshold, selected, perf

preprocessor, ensemble, threshold, selected_features, perf_metrics = load_artifacts()

st.sidebar.header("Input Patient Data")
age = st.sidebar.number_input("Age (years)", 1, 120, 45)
fasting_glucose = st.sidebar.number_input("Fasting Glucose (mg/dL)", 40.0, 600.0, 110.0)
fasting_insulin = st.sidebar.number_input("Fasting Insulin (µIU/mL)", 0.1, 300.0, 15.0)
bmi = st.sidebar.number_input("BMI", 10.0, 70.0, 28.0)
triglycerides = st.sidebar.number_input("Triglycerides (mg/dL)", 30.0, 1000.0, 150.0)
hdl_cholesterol = st.sidebar.number_input("HDL Cholesterol (mg/dL)", 5.0, 150.0, 45.0)
waist_circumference = st.sidebar.number_input("Waist Circumference (cm)", 40.0, 200.0, 90.0)
hip_circumference = st.sidebar.number_input("Hip Circumference (cm)", 40.0, 200.0, 100.0)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
show_shap = st.sidebar.checkbox("Show SHAP Explanation", value=False)

input_dict = {
    "age": age,
    "fasting_glucose": fasting_glucose,
    "fasting_insulin": fasting_insulin,
    "bmi": bmi,
    "triglycerides": triglycerides,
    "hdl_cholesterol": hdl_cholesterol,
    "waist_circumference": waist_circumference,
    "hip_circumference": hip_circumference,
    "sex": sex,
}

if st.button("Predict Risk"):
    df = pd.DataFrame([input_dict])
    df = add_engineered_features(df)
    # Ensure expected columns exist
    for col in preprocessor.numeric_cols or []:
        if col not in df.columns:
            df[col] = np.nan
    for col in preprocessor.categorical_cols or []:
        if col not in df.columns:
            df[col] = "__missing__"

    X_transformed, feature_names = preprocessor.transform(df)
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    base_models = ensemble["base_models"]
    model_order = ensemble.get("model_order") or list(base_models.keys())
    meta_features = []
    for name in model_order:
        meta_features.append(base_models[name].predict_proba(X_df)[:, 1])
    meta_stack = np.column_stack(meta_features)
    calibrated = ensemble["calibrated_model"]
    prob = float(calibrated.predict_proba(meta_stack)[:, 1][0])
    pred = int(prob >= threshold)

    def risk_bucket(p: float) -> str:
        if p < 0.33: return "Low"
        if p < 0.67: return "Medium"
        return "High"

    col1, col2, col3 = st.columns(3)
    col1.metric("Probability", f"{prob:.4f}")
    col2.metric("Prediction", "IR" if pred == 1 else "Non-IR")
    col3.metric("Risk Level", risk_bucket(prob))

    st.subheader("Raw Inputs")
    input_display = pd.DataFrame({"Feature": list(input_dict.keys()), "Value": [str(v) for v in input_dict.values()]})
    st.dataframe(input_display, use_container_width=True)

    if show_shap:
        try:
            import shap
            explainer = shap.Explainer(calibrated)
            shap_values = explainer(meta_stack)
            st.subheader("SHAP Feature Contributions")
            shap_df = pd.DataFrame({"feature": model_order, "value": shap_values.values[0]})
            st.bar_chart(shap_df.set_index("feature"))
        except Exception as e:
            st.warning(f"Unable to compute SHAP values: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Artifacts Loaded:")
st.sidebar.write(f"Selected Features: {len(selected_features)}")
st.sidebar.write(f"Threshold: {threshold:.3f}")
if perf_metrics:
    stacking = perf_metrics.get("stacking", {})
    st.sidebar.write(f"Val AUC: {stacking.get('val_auc')}")
    st.sidebar.write(f"Val F1: {stacking.get('val_f1')}")

st.markdown("---")
st.caption("Model artifacts assumed pre-trained. Run training pipeline before first use if missing.")

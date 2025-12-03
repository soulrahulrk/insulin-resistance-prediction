"""Streamlit app for Insulin Resistance Risk Prediction.

A machine learning system for predicting insulin resistance using a 
stacking ensemble of gradient-boosted models (XGBoost, LightGBM, CatBoost, 
GradientBoosting) with a calibrated Logistic Regression meta-learner.

Author: Rahul Kumar
License: MIT
"""

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

from src.config import (
    FEATURE_TRANSFORMER_PATH,
    ENSEMBLE_MODEL_PATH,
    OPTIMAL_THRESHOLD_PATH,
    SELECTED_FEATURES_PATH,
    PERFORMANCE_METRICS_PATH,
    MODELS_DIR,
)
from src.utils import load_json

# Page configuration
st.set_page_config(
    page_title="Insulin Resistance Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
        text-align: center;
    }
    .stNumberInput > div > div > input {
        font-size: 16px;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .result-box h2 {
        color: white;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🩺 Insulin Resistance Risk Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Stacking Ensemble ML Model</p>', unsafe_allow_html=True)

# Define feature ranges for validation and randomization
FEATURE_RANGES = {
    "age": {"min": 18, "max": 90, "default": 45, "step": 1, "type": "int", "label": "Age", "unit": "years"},
    "bmi": {"min": 15.0, "max": 50.0, "default": 28.0, "step": 0.1, "type": "float", "label": "BMI", "unit": "kg/m²"},
    "blood_pressure": {"min": 60.0, "max": 180.0, "default": 80.0, "step": 1.0, "type": "float", "label": "Blood Pressure", "unit": "mmHg"},
    "skin_thickness": {"min": 5.0, "max": 100.0, "default": 25.0, "step": 1.0, "type": "float", "label": "Skin Thickness", "unit": "mm"},
    "pregnancies": {"min": 0, "max": 17, "default": 1, "step": 1, "type": "int", "label": "Pregnancies", "unit": ""},
    "diabetes_pedigree": {"min": 0.05, "max": 2.5, "default": 0.5, "step": 0.01, "type": "float", "label": "Diabetes Pedigree", "unit": ""},
    "hba1c": {"min": 4.0, "max": 14.0, "default": 5.5, "step": 0.1, "type": "float", "label": "HbA1c", "unit": "%"},
    "stress_level": {"min": 1, "max": 10, "default": 5, "step": 1, "type": "int", "label": "Stress Level", "unit": ""},
    "stresslevel": {"min": 1, "max": 10, "default": 5, "step": 1, "type": "int", "label": "Stress Level", "unit": ""},
    "alcohol": {"min": 0.0, "max": 20.0, "default": 2.0, "step": 0.5, "type": "float", "label": "Alcohol", "unit": "drinks/wk"},
    "smoking": {"min": 0, "max": 40, "default": 0, "step": 1, "type": "int", "label": "Smoking", "unit": "cigs/day"},
    "exercisehours": {"min": 0.0, "max": 20.0, "default": 3.0, "step": 0.5, "type": "float", "label": "Exercise", "unit": "hrs/wk"},
    "exercise_hours_week": {"min": 0.0, "max": 20.0, "default": 3.0, "step": 0.5, "type": "float", "label": "Exercise", "unit": "hrs/wk"},
    "sleep_hours": {"min": 3.0, "max": 12.0, "default": 7.0, "step": 0.5, "type": "float", "label": "Sleep", "unit": "hrs/night"},
    "sleepquality": {"min": 1, "max": 10, "default": 7, "step": 1, "type": "int", "label": "Sleep Quality", "unit": ""},
    "dietscore": {"min": 0, "max": 100, "default": 60, "step": 5, "type": "int", "label": "Diet Score", "unit": ""},
    "health_literacy": {"min": 0, "max": 100, "default": 60, "step": 5, "type": "int", "label": "Health Literacy", "unit": ""},
    "nutrition_quality": {"min": 0, "max": 100, "default": 60, "step": 5, "type": "int", "label": "Nutrition Quality", "unit": ""},
    "educationyears": {"min": 0, "max": 25, "default": 12, "step": 1, "type": "int", "label": "Education", "unit": "years"},
    "healthcareaccess": {"min": 0, "max": 100, "default": 70, "step": 5, "type": "int", "label": "Healthcare Access", "unit": "%"},
    "healthcare_access": {"min": 0, "max": 100, "default": 70, "step": 5, "type": "int", "label": "Healthcare Access", "unit": "%"},
    "healthcare_spend_usd": {"min": 0, "max": 10000, "default": 500, "step": 100, "type": "int", "label": "Healthcare Spend", "unit": "USD"},
    "internetaccess": {"min": 0, "max": 100, "default": 80, "step": 5, "type": "int", "label": "Internet Access", "unit": "%"},
    "electricityaccess": {"min": 0, "max": 100, "default": 99, "step": 5, "type": "int", "label": "Electricity Access", "unit": "%"},
    "mobilephone": {"min": 0, "max": 100, "default": 90, "step": 5, "type": "int", "label": "Mobile Phone", "unit": "%"},
    "medical_equipment_index": {"min": 0, "max": 100, "default": 70, "step": 5, "type": "int", "label": "Medical Equipment", "unit": ""},
    "prevention_access": {"min": 0, "max": 100, "default": 60, "step": 5, "type": "int", "label": "Prevention Access", "unit": "%"},
    "specialist_availability": {"min": 0, "max": 100, "default": 50, "step": 5, "type": "int", "label": "Specialist Availability", "unit": "%"},
    "diabetesprevalence": {"min": 0.0, "max": 30.0, "default": 8.0, "step": 0.5, "type": "float", "label": "Diabetes Prevalence", "unit": "%"},
    "diabetes_risk_factor": {"min": 0.0, "max": 10.0, "default": 3.0, "step": 0.1, "type": "float", "label": "Diabetes Risk Factor", "unit": ""},
    "outcome": {"min": 0, "max": 1, "default": 0, "step": 1, "type": "int", "label": "Prior Diabetes", "unit": "0=No, 1=Yes"},
    "source_index": {"min": 0, "max": 10, "default": 1, "step": 1, "type": "int", "label": "Data Source", "unit": ""},
}


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model artifacts."""
    preprocessor = None
    ensemble = None
    threshold = 0.5
    selected = []
    perf = {}

    try:
        preprocessor = joblib.load(FEATURE_TRANSFORMER_PATH)
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        
    try:
        ensemble = joblib.load(ENSEMBLE_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading ensemble model: {e}")

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


def generate_random_values(input_features: list) -> dict:
    """Generate random values within valid ranges for all features."""
    random_values = {}
    
    for feature in input_features:
        if feature == "age_bmi":
            continue  # Computed feature
        
        if feature in FEATURE_RANGES:
            info = FEATURE_RANGES[feature]
            if info["type"] == "int":
                random_values[feature] = int(np.random.randint(info["min"], info["max"] + 1))
            else:
                random_values[feature] = round(np.random.uniform(info["min"], info["max"]), 2)
        else:
            # For features without defined ranges
            random_values[feature] = round(np.random.uniform(0, 100), 1)
    
    return random_values


def add_engineered_features_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features for prediction."""
    df = df.copy()
    if "age" in df.columns and "bmi" in df.columns:
        df["age_bmi"] = df["age"] * df["bmi"]
    return df


def get_risk_bucket(prob: float) -> tuple:
    """Get risk level and emoji based on probability."""
    if prob < 0.33:
        return "Low", "🟢", "success"
    if prob < 0.67:
        return "Medium", "🟡", "warning"
    return "High", "🔴", "error"


# Load artifacts
preprocessor, ensemble, threshold, selected_features, perf_metrics = load_artifacts()

# Filter selected_features to exclude computed ones for input
input_features = [f for f in selected_features if f != "age_bmi"]

# Initialize session state
if "random_counter" not in st.session_state:
    st.session_state.random_counter = 0

# Group features by category (remove duplicates)
FEATURE_GROUPS = {
    "👤 Demographics": ["age", "pregnancies", "educationyears"],
    "📏 Biometrics & Labs": ["bmi", "blood_pressure", "skin_thickness", "diabetes_pedigree", "hba1c"],
    "🏃 Lifestyle": ["stress_level", "stresslevel", "alcohol", "smoking", "exercisehours", 
                    "exercise_hours_week", "sleep_hours", "sleepquality", "dietscore", 
                    "health_literacy", "nutrition_quality"],
    "🏥 Health System": ["healthcareaccess", "healthcare_access", "healthcare_spend_usd",
                        "medical_equipment_index", "prevention_access", "specialist_availability"],
    "🩸 Diabetes": ["diabetesprevalence", "diabetes_risk_factor", "outcome"],
}

# ============= SIDEBAR =============
st.sidebar.header("📊 Patient Data Input")

# Randomize button with counter to force refresh
col_rand1, col_rand2 = st.sidebar.columns([3, 1])
with col_rand1:
    randomize_clicked = st.button("🎲 Randomize All", type="secondary", width="stretch")
with col_rand2:
    reset_clicked = st.button("↺", help="Reset to defaults")

if randomize_clicked:
    st.session_state.random_counter += 1
    st.session_state.random_values = generate_random_values(input_features)

if reset_clicked:
    st.session_state.random_counter = 0
    if "random_values" in st.session_state:
        del st.session_state.random_values

st.sidebar.markdown("---")

# Collect inputs
input_dict = {}

for group_name, group_features in FEATURE_GROUPS.items():
    # Filter to only features that exist in input_features
    relevant = [f for f in group_features if f in input_features]
    if not relevant:
        continue
    
    st.sidebar.subheader(group_name)
    
    for feature in relevant:
        # Get value from random_values if exists, otherwise use default
        if "random_values" in st.session_state and feature in st.session_state.random_values:
            current_val = st.session_state.random_values[feature]
        elif feature in FEATURE_RANGES:
            current_val = FEATURE_RANGES[feature]["default"]
        else:
            current_val = 50.0
        
        if feature in FEATURE_RANGES:
            info = FEATURE_RANGES[feature]
            label_text = f"{info['label']} ({info['unit']})" if info['unit'] else info['label']
            
            if info["type"] == "int":
                input_dict[feature] = st.sidebar.number_input(
                    label_text,
                    min_value=int(info["min"]),
                    max_value=int(info["max"]),
                    value=int(current_val),
                    step=int(info["step"]),
                    format="%d",
                    key=f"{feature}_{st.session_state.random_counter}"
                )
            else:
                input_dict[feature] = st.sidebar.number_input(
                    label_text,
                    min_value=float(info["min"]),
                    max_value=float(info["max"]),
                    value=float(current_val),
                    step=float(info["step"]),
                    format="%.1f",
                    key=f"{feature}_{st.session_state.random_counter}"
                )
        else:
            input_dict[feature] = st.sidebar.number_input(
                feature.replace('_', ' ').title(),
                value=float(current_val),
                format="%.1f",
                key=f"{feature}_{st.session_state.random_counter}"
            )

# Handle uncategorized features
all_categorized = [f for group in FEATURE_GROUPS.values() for f in group]
other_features = [f for f in input_features if f not in all_categorized]

if other_features:
    st.sidebar.subheader("📋 Other Factors")
    for feature in other_features:
        if "random_values" in st.session_state and feature in st.session_state.random_values:
            current_val = st.session_state.random_values[feature]
        elif feature in FEATURE_RANGES:
            current_val = FEATURE_RANGES[feature]["default"]
        else:
            current_val = 50.0
        
        if feature in FEATURE_RANGES:
            info = FEATURE_RANGES[feature]
            label_text = f"{info['label']} ({info['unit']})" if info['unit'] else info['label']
            
            if info["type"] == "int":
                input_dict[feature] = st.sidebar.number_input(
                    label_text,
                    min_value=int(info["min"]),
                    max_value=int(info["max"]),
                    value=int(current_val),
                    step=int(info["step"]),
                    format="%d",
                    key=f"{feature}_{st.session_state.random_counter}"
                )
            else:
                input_dict[feature] = st.sidebar.number_input(
                    label_text,
                    min_value=float(info["min"]),
                    max_value=float(info["max"]),
                    value=float(current_val),
                    step=float(info["step"]),
                    format="%.1f",
                    key=f"{feature}_{st.session_state.random_counter}"
                )
        else:
            input_dict[feature] = st.sidebar.number_input(
                feature.replace('_', ' ').title(),
                value=float(current_val),
                format="%.1f",
                key=f"{feature}_{st.session_state.random_counter}"
            )

# ============= MAIN CONTENT =============
# Info section
st.info("""
**🧠 About This Tool**

This system predicts insulin resistance risk using an ensemble of machine learning models (XGBoost, LightGBM, CatBoost, GradientBoosting) with {} features.

⚠️ **For educational purposes only** — Not a diagnostic tool. Consult a healthcare professional for medical decisions.
""".format(len(input_features)))

st.markdown("---")

# Prediction button
predict_clicked = st.button("🔮 Predict Insulin Resistance Risk", type="primary", width="stretch")

if predict_clicked:
    with st.spinner("Analyzing..."):
        # Build input DataFrame
        df = pd.DataFrame([input_dict])
        
        # Add engineered features
        df = add_engineered_features_simple(df)
        
        # Ensure all expected columns exist with default values
        for col in preprocessor.numeric_cols or []:
            if col not in df.columns:
                df[col] = np.nan
        for col in preprocessor.categorical_cols or []:
            if col not in df.columns:
                df[col] = "__missing__"
        
        # Transform using preprocessor
        X_transformed, feature_names = preprocessor.transform(df)
        X_df = pd.DataFrame(X_transformed, columns=feature_names)
        
        # Get predictions from base models
        base_models = ensemble["base_models"]
        model_order = ensemble.get("model_order") or list(base_models.keys())
        
        meta_features = []
        for name in model_order:
            proba = base_models[name].predict_proba(X_df)[:, 1]
            meta_features.append(proba)
        
        meta_stack = np.column_stack(meta_features)
        
        # Get calibrated prediction
        calibrated = ensemble["calibrated_model"]
        prob = float(calibrated.predict_proba(meta_stack)[:, 1][0])
        pred = int(prob >= threshold)
        
        risk_level, risk_emoji, risk_type = get_risk_bucket(prob)
    
    # Display results
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    # Main result box
    prediction_text = "Insulin Resistant" if pred == 1 else "Not Insulin Resistant"
    st.markdown(f"""
    <div class="result-box">
        <h2>{risk_emoji} {risk_level} Risk</h2>
        <h1 style="font-size: 3rem; margin: 1rem 0;">{prob:.1%}</h1>
        <h3>{prediction_text}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    st.progress(prob, text=f"Risk Score: {prob:.1%}")
    
    # Interpretation
    st.markdown("### 📝 Clinical Interpretation")
    if risk_type == "success":
        st.success("""
        **Low Risk**: Based on the provided inputs, the model indicates a low probability 
        of insulin resistance. Continue maintaining a healthy lifestyle with regular exercise, 
        balanced diet, and routine health check-ups.
        """)
    elif risk_type == "warning":
        st.warning("""
        **Moderate Risk**: The model suggests moderate probability of insulin resistance. 
        Consider lifestyle modifications including increased physical activity, dietary changes 
        (reduced refined carbs/sugars), and consult a healthcare provider for further evaluation.
        """)
    else:
        st.error("""
        **High Risk**: The model indicates elevated probability of insulin resistance. 
        Strongly recommend consulting a healthcare professional for proper evaluation, 
        including fasting glucose, insulin levels, and HbA1c testing.
        """)
    
    # Expandable sections
    with st.expander("📋 Input Summary", expanded=False):
        # Create a nice table
        display_df = pd.DataFrame({
            "Feature": [FEATURE_RANGES.get(k, {}).get("label", k.replace("_", " ").title()) for k in input_dict.keys()],
            "Value": list(input_dict.values())
        })
        st.dataframe(display_df, width="stretch", hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Insulin Resistance Prediction System</strong></p>
    <p>Built with ❤️ using Streamlit, XGBoost, LightGBM, CatBoost</p>
    <p>⚠️ This tool is for research and educational purposes only.</p>
    <p><a href="https://github.com/soulrahulrk/insulin-resistance-prediction" target="_blank">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)

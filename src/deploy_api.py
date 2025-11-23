"""FastAPI deployment application with monitoring and explainability."""

from __future__ import annotations

import hashlib
import io
import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    DATA_PATH,
    ENSEMBLE_MODEL_PATH,
    FEATURE_TRANSFORMER_PATH,
    OPTIMAL_THRESHOLD_PATH,
    PERFORMANCE_METRICS_PATH,
    RANDOM_STATE,
    SELECTED_FEATURES_PATH,
)
from src.explainability_fast import explain_single
from src.features import compute_engineered_features
from src.monitoring import compute_aggregate_metrics, export_prometheus, record_prediction
from src.drift_monitor import alert_if_drift, compute_feature_drift
from src.utils import configure_logger, load_json

logger = configure_logger(__name__)

# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Insulin Resistance Prediction API",
    description="Production ML API for insulin resistance classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE (loaded on startup)
# ============================================================================
MODEL: Dict | None = None
PREPROCESSOR = None
SELECTED_FEATURES: List[str] | None = None
OPTIMAL_THRESHOLD = 0.5
MODEL_VERSION = "unknown"
PERFORMANCE_BASELINE: Dict = {}
PREDICTION_BUFFER: deque = deque(maxlen=500)
REFERENCE_FEATURES: pd.DataFrame | None = None
DRIFT_FEATURES: List[str] = []
DRIFT_BUFFER: deque = deque(maxlen=200)
DRIFT_MIN_SAMPLES = 50


def load_artifacts():
    """Load model artifacts on app startup."""
    global MODEL, PREPROCESSOR, SELECTED_FEATURES, OPTIMAL_THRESHOLD, MODEL_VERSION, PERFORMANCE_BASELINE
    global REFERENCE_FEATURES, DRIFT_FEATURES
    
    try:
        MODEL = joblib.load(ENSEMBLE_MODEL_PATH)
        logger.info(f"✓ Loaded ensemble model from {ENSEMBLE_MODEL_PATH}")
        MODEL_VERSION = str(int(Path(ENSEMBLE_MODEL_PATH).stat().st_mtime))
    except FileNotFoundError:
        logger.error(f"Model not found at {ENSEMBLE_MODEL_PATH}")
        raise
    
    try:
        PREPROCESSOR = joblib.load(FEATURE_TRANSFORMER_PATH)
        logger.info(f"✓ Loaded preprocessor from {FEATURE_TRANSFORMER_PATH}")
    except FileNotFoundError:
        logger.error(f"Preprocessor not found at {FEATURE_TRANSFORMER_PATH}")
        raise
    
    try:
        SELECTED_FEATURES = load_json(SELECTED_FEATURES_PATH)
        logger.info(f"✓ Loaded {len(SELECTED_FEATURES)} selected features")
    except FileNotFoundError:
        logger.error(f"Selected features not found at {SELECTED_FEATURES_PATH}")
        raise
    
    try:
        with open(OPTIMAL_THRESHOLD_PATH, 'r') as f:
            OPTIMAL_THRESHOLD = float(f.read().strip())
        logger.info(f"✓ Loaded optimal threshold: {OPTIMAL_THRESHOLD}")
    except FileNotFoundError:
        logger.warning("Threshold file not found, using default 0.5")
        OPTIMAL_THRESHOLD = 0.5

    try:
        PERFORMANCE_BASELINE = load_json(PERFORMANCE_METRICS_PATH)
        logger.info("✓ Baseline metrics loaded: %s", json.dumps(PERFORMANCE_BASELINE.get('stacking', {})))
    except FileNotFoundError:
        PERFORMANCE_BASELINE = {}
        logger.warning("Performance metrics file missing; baseline comparisons disabled")

    try:
        reference_df = pd.read_csv(DATA_PATH)
        if 'ir_label' in reference_df.columns:
            reference_df = reference_df.drop(columns=['ir_label'])
        reference_df = compute_engineered_features(reference_df)
        transformed, feature_names = PREPROCESSOR.transform(reference_df)
        ref_frame = pd.DataFrame(transformed, columns=feature_names)
        if len(ref_frame) > 2000:
            ref_frame = ref_frame.sample(n=2000, random_state=RANDOM_STATE).reset_index(drop=True)
        REFERENCE_FEATURES = ref_frame
        DRIFT_FEATURES = feature_names
        logger.info("✓ Loaded %d reference rows for drift monitoring", len(ref_frame))
    except Exception as exc:  # pragma: no cover - optional path
        REFERENCE_FEATURES = None
        DRIFT_FEATURES = []
        logger.warning("Unable to initialize drift reference dataset: %s", exc)


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================
class PatientFeatures(BaseModel):
    """Schema for single patient prediction."""
    age: float = Field(..., description="Age in years")
    fasting_glucose: float = Field(..., description="Fasting glucose in mg/dL")
    fasting_insulin: float = Field(..., description="Fasting insulin in µIU/mL")
    bmi: float = Field(..., description="Body Mass Index")
    sex: Optional[str] = Field(None, description="Gender (M/F)")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 45,
                "fasting_glucose": 120,
                "fasting_insulin": 15,
                "bmi": 28.5,
                "sex": "M"
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    probability: float = Field(..., description="IR probability (0-1)")
    prediction: int = Field(..., description="Binary prediction (0/1)")
    threshold_used: float = Field(..., description="Threshold for classification")
    risk_level: str = Field(..., description="Risk category: low/medium/high")
    shap_top3: Optional[List[Dict[str, float]]] = Field(None, description="Top-3 SHAP feature contributions")


class BatchPredictionResponse(BaseModel):
    """Schema for batch predictions."""
    predictions: List[PredictionResponse]
    total_records: int
    successful: int
    failed: int


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def compute_risk_level(probability: float) -> str:
    """Determine risk level from probability.
    
    Args:
        probability: IR probability.
        
    Returns:
        Risk level string.
    """
    if probability < 0.33:
        return "low"
    elif probability < 0.67:
        return "medium"
    else:
        return "high"


def preprocess_features(features_dict: Dict) -> pd.DataFrame:
    """Preprocess input features.
    
    Args:
        features_dict: Input feature dictionary.
        
    Returns:
        Preprocessed feature dataframe.
    """
    df = pd.DataFrame([features_dict])
    df = compute_engineered_features(df)

    # Ensure all expected columns exist before transformation
    for col in getattr(PREPROCESSOR, "numeric_cols", []):
        if col not in df.columns:
            df[col] = np.nan
    for col in getattr(PREPROCESSOR, "categorical_cols", []):
        if col not in df.columns:
            df[col] = "__missing__"

    X_transformed, feature_names = PREPROCESSOR.transform(df)
    transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    return transformed_df


def _predict_probability(features: pd.DataFrame) -> np.ndarray:
    base_models = MODEL.get('base_models')
    if base_models is None:
        raise RuntimeError("Base models missing from ensemble artifact")
    model_order = MODEL.get('model_order') or list(base_models.keys())
    meta_features = []
    for name in model_order:
        estimator = base_models[name]
        meta_features.append(estimator.predict_proba(features)[:, 1])
    meta_stack = np.column_stack(meta_features)
    calibrated = MODEL.get('calibrated_model')
    if calibrated is None:
        raise RuntimeError("Calibrated model missing from ensemble artifact")
    return calibrated.predict_proba(meta_stack)[:, 1]


def _hash_features(features: pd.DataFrame) -> str:
    payload = features.round(6).to_json(orient="records", double_precision=6)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _record_drift_sample(features: pd.DataFrame) -> None:
    """Track features for drift detection and emit alerts when necessary."""
    if REFERENCE_FEATURES is None or not DRIFT_FEATURES:
        return
    for _, row in features.iterrows():
        DRIFT_BUFFER.append(row.to_dict())
    if len(DRIFT_BUFFER) < DRIFT_MIN_SAMPLES:
        return
    recent_df = pd.DataFrame(list(DRIFT_BUFFER))
    recent_df = recent_df.reindex(columns=DRIFT_FEATURES)
    drifted = compute_feature_drift(REFERENCE_FEATURES, recent_df, DRIFT_FEATURES)
    if drifted:
        alert_if_drift(drifted)
        logger.warning("Drift alert triggered for: %s", ", ".join(drifted))
        DRIFT_BUFFER.clear()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model artifacts on app startup."""
    logger.info("Loading model artifacts...")
    load_artifacts()
    logger.info("✓ App startup complete")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(patient: PatientFeatures, explain: bool = False):
    """Predict IR for single patient.
    
    Args:
        patient: Patient features.
        
    Returns:
        Prediction response.
    """
    try:
        # Preprocess
        features_dict = patient.dict()
        X = preprocess_features(features_dict)
        start = time.perf_counter()
        
        y_prob = float(_predict_probability(X)[0])
        
        # Classify
        y_pred = 1 if y_prob >= OPTIMAL_THRESHOLD else 0
        risk_level = compute_risk_level(y_prob)
        shap_payload = None
        if explain:
            shap_payload = explain_single(X, MODEL).get('top3')

        latency_ms = (time.perf_counter() - start) * 1000.0
        trace_id = uuid.uuid4().hex
        record = {
            "timestamp": time.time(),
            "probability": y_prob,
            "prediction": y_pred,
            "threshold_used": OPTIMAL_THRESHOLD,
            "risk_level": risk_level,
            "model_version": MODEL_VERSION,
            "trace_id": trace_id,
            "latency_ms": latency_ms,
            "shap_top3": shap_payload,
            "features_hash": _hash_features(X),
        }
        PREDICTION_BUFFER.append(record)
        record_prediction(record)
        _record_drift_sample(X)
        metrics = compute_aggregate_metrics(PREDICTION_BUFFER)
        export_prometheus(metrics)
        
        return PredictionResponse(
            probability=float(y_prob),
            prediction=int(y_pred),
            threshold_used=float(OPTIMAL_THRESHOLD),
            risk_level=risk_level,
            shap_top3=shap_payload
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Predict IR for batch of patients from CSV.
    
    Args:
        file: CSV file with patient records.
        
    Returns:
        Batch prediction response.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode()))
        
        predictions = []
        successful = 0
        failed = 0
        
        for idx, row in df.iterrows():
            try:
                features_dict = row.to_dict()
                X = preprocess_features(features_dict)
                y_prob = float(_predict_probability(X)[0])
                y_pred = 1 if y_prob >= OPTIMAL_THRESHOLD else 0
                risk_level = compute_risk_level(y_prob)
                record = {
                    "timestamp": time.time(),
                    "probability": y_prob,
                    "prediction": y_pred,
                    "threshold_used": OPTIMAL_THRESHOLD,
                    "risk_level": risk_level,
                    "model_version": MODEL_VERSION,
                    "trace_id": uuid.uuid4().hex,
                    "latency_ms": 0.0,
                    "shap_top3": None,
                    "features_hash": _hash_features(X),
                }
                PREDICTION_BUFFER.append(record)
                record_prediction(record)
                _record_drift_sample(X)
                
                predictions.append(PredictionResponse(
                    probability=float(y_prob),
                    prediction=int(y_pred),
                    threshold_used=float(OPTIMAL_THRESHOLD),
                    risk_level=risk_level
                ))
                successful += 1
            except Exception as e:
                logger.warning(f"Failed to predict row {idx}: {str(e)}")
                failed += 1
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_records=len(df),
            successful=successful,
            failed=failed
        )
        metrics = compute_aggregate_metrics(PREDICTION_BUFFER)
        export_prometheus(metrics)
        return response
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/metrics")
async def metrics_endpoint():
    """Return simple aggregated metrics for monitoring."""
    metrics = compute_aggregate_metrics(PREDICTION_BUFFER)
    metrics["model_version"] = MODEL_VERSION
    metrics["baseline_auc"] = PERFORMANCE_BASELINE.get("stacking", {}).get("val_auc")
    return metrics


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)

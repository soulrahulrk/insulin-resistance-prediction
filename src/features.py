"""Feature engineering utilities."""

import numpy as np
import pandas as pd

from src.utils import configure_logger

logger = configure_logger(__name__)


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered features per current specification.

    Features added:
      - homa_ir (raw) and homa_ir_log
      - quicki
      - tg_hdl_ratio
      - waist_hip_ratio
      - age_bmi
      - fasting_glucose_log, fasting_insulin_log, triglycerides_log

    Legacy categorical bins removed for leaner feature space.
    """
    df = df.copy()

    if "fasting_glucose" in df.columns and "fasting_insulin" in df.columns:
        glucose = df["fasting_glucose"].fillna(df["fasting_glucose"].median())
        insulin = df["fasting_insulin"].fillna(df["fasting_insulin"].median())
        # homa_ir and homa_ir_log removed to prevent leakage as they are direct proxies for the label
        safe_glucose = glucose.clip(lower=1)
        safe_insulin = insulin.clip(lower=1)
        df["quicki"] = 1.0 / (np.log(safe_insulin) + np.log(safe_glucose))
        logger.info("✓ Computed QUICKI (HOMA-IR excluded for leakage prevention)")

    if "triglycerides" in df.columns and "hdl_cholesterol" in df.columns:
        tg = df["triglycerides"].fillna(df["triglycerides"].median())
        hdl = df["hdl_cholesterol"].fillna(df["hdl_cholesterol"].median()).clip(lower=0.1)
        df["tg_hdl_ratio"] = tg / hdl
        logger.info("✓ Computed TG/HDL ratio")

    if "waist_circumference" in df.columns and "hip_circumference" in df.columns:
        waist = df["waist_circumference"].fillna(df["waist_circumference"].median())
        hip = df["hip_circumference"].fillna(df["hip_circumference"].median()).clip(lower=0.1)
        df["waist_hip_ratio"] = waist / hip
        logger.info("✓ Computed waist_hip_ratio")

    if "age" in df.columns and "bmi" in df.columns:
        age = df["age"].fillna(df["age"].median())
        bmi = df["bmi"].fillna(df["bmi"].median())
        df["age_bmi"] = age * bmi
        logger.info("✓ Computed age_bmi interaction")

    if "fasting_glucose" in df.columns:
        df["fasting_glucose_log"] = np.log1p(df["fasting_glucose"])
    if "fasting_insulin" in df.columns:
        df["fasting_insulin_log"] = np.log1p(df["fasting_insulin"])
    if "triglycerides" in df.columns:
        df["triglycerides_log"] = np.log1p(df["triglycerides"])

    return df


def select_most_important_features(df: pd.DataFrame, y: pd.Series, n_features: int = 30) -> list:
    """Select top N most important features using mutual information.
    
    Args:
        df: Input feature matrix.
        y: Target variable.
        n_features: Number of top features to select.
        
    Returns:
        List of selected feature names.
    """
    from sklearn.feature_selection import mutual_info_classif
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = df[numeric_cols]
    
    mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
    ranked_features = sorted(zip(numeric_cols, mi_scores), key=lambda x: x[1], reverse=True)
    
    selected = [feat for feat, score in ranked_features[:n_features]]
    logger.info(f"Selected top {len(selected)} features by mutual information")
    
    return selected

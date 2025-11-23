"""Feature engineering utilities."""

import numpy as np
import pandas as pd

from src.utils import configure_logger

logger = configure_logger(__name__)


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all engineered features for IR prediction.
    
    Args:
        df: Input DataFrame with raw features.
        
    Returns:
        DataFrame with additional engineered columns.
    """
    df = df.copy()
    
    # 1. HOMA-IR Index
    if "fasting_glucose" in df.columns and "fasting_insulin" in df.columns:
        glucose = df["fasting_glucose"].fillna(df["fasting_glucose"].median())
        insulin = df["fasting_insulin"].fillna(df["fasting_insulin"].median())
        df["homa_ir"] = (glucose * insulin) / 405.0
        df["homa_ir"] = np.log1p(df["homa_ir"])
        logger.info("✓ Computed HOMA-IR")
    
    # 2. QUICKI Index
    if "fasting_glucose" in df.columns and "fasting_insulin" in df.columns:
        glucose = df["fasting_glucose"].fillna(df["fasting_glucose"].median()).clip(lower=1)
        insulin = df["fasting_insulin"].fillna(df["fasting_insulin"].median()).clip(lower=1)
        df["quicki"] = 1.0 / (np.log(insulin) + np.log(glucose))
        logger.info("✓ Computed QUICKI")
    
    # 3. Triglyceride to HDL Ratio
    if "triglycerides" in df.columns and "hdl_cholesterol" in df.columns:
        tg = df["triglycerides"].fillna(df["triglycerides"].median())
        hdl = df["hdl_cholesterol"].fillna(df["hdl_cholesterol"].median()).clip(lower=0.1)
        df["tg_hdl_ratio"] = tg / hdl
        logger.info("✓ Computed TG/HDL ratio")
    
    # 4. Waist-to-Hip Ratio
    if "waist_circumference" in df.columns and "hip_circumference" in df.columns:
        waist = df["waist_circumference"].fillna(df["waist_circumference"].median())
        hip = df["hip_circumference"].fillna(df["hip_circumference"].median()).clip(lower=0.1)
        df["waist_hip_ratio"] = waist / hip
        logger.info("✓ Computed waist-to-hip ratio")
    
    # 5. Age × BMI Interaction
    if "age" in df.columns and "bmi" in df.columns:
        age = df["age"].fillna(df["age"].median())
        bmi = df["bmi"].fillna(df["bmi"].median())
        df["age_bmi_interaction"] = age * bmi
        logger.info("✓ Computed age × BMI interaction")
    
    # 6. BMI Categories
    if "bmi" in df.columns:
        bmi = df["bmi"]
        df["bmi_category_underweight"] = (bmi < 18.5).astype(int)
        df["bmi_category_normal"] = ((bmi >= 18.5) & (bmi < 25)).astype(int)
        df["bmi_category_overweight"] = ((bmi >= 25) & (bmi < 30)).astype(int)
        df["bmi_category_obese"] = (bmi >= 30).astype(int)
        logger.info("✓ Computed BMI categories")
    
    # 7. Age Bins
    if "age" in df.columns:
        age = df["age"]
        df["age_group_under_30"] = (age < 30).astype(int)
        df["age_group_30_45"] = ((age >= 30) & (age < 45)).astype(int)
        df["age_group_45_60"] = ((age >= 45) & (age < 60)).astype(int)
        df["age_group_over_60"] = (age >= 60).astype(int)
        logger.info("✓ Computed age groups")
    
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

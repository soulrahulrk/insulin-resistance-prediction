"""Audit script for data quality and leakage detection."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import DATA_PATH, RANDOM_STATE
from src.data_loader import load_data
from src.preprocessing import add_engineered_features, Preprocessor
from src.utils import configure_logger

logger = configure_logger(__name__)

def audit_data():
    logger.info("STARTING DATA AUDIT")
    
    # 1. Load Data
    df = load_data(DATA_PATH, drop_label_leak_features=False) # Keep homa_ir for audit if present
    logger.info(f"Loaded data shape: {df.shape}")
    
    # 2. Check Missingness
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info(f"Missing values:\n{missing}")
    else:
        logger.info("No missing values found (after loader processing)")

    # 3. Engineered Features
    df = add_engineered_features(df)
    logger.info(f"Data shape after engineering: {df.shape}")
    
    # 4. Label Distribution
    if 'ir_label' in df.columns:
        logger.info(f"Label distribution:\n{df['ir_label'].value_counts(normalize=True)}")
    
    # 5. Correlation Analysis
    if 'ir_label' in df.columns:
        numeric_df = df.select_dtypes(include=[np.number])
        corrs = numeric_df.corrwith(df['ir_label']).sort_values(ascending=False)
        logger.info("Top 10 Positive Correlations with ir_label:")
        logger.info(corrs.head(10))
        logger.info("Top 10 Negative Correlations with ir_label:")
        logger.info(corrs.tail(10))
        
        # Check for leakage
        leakage = corrs[abs(corrs) > 0.95]
        if len(leakage) > 1: # ir_label itself is 1.0
            logger.warning(f"POTENTIAL LEAKAGE DETECTED (Corr > 0.95):\n{leakage}")
            
    # 6. Feature Importance (Quick RF)
    if 'ir_label' in df.columns:
        logger.info("Running quick Random Forest for feature importance...")
        # Drop label and potential leakage for this check
        X = df.drop(columns=['ir_label'])
        # Also drop homa_ir if present as it dominates
        if 'homa_ir' in X.columns:
            X = X.drop(columns=['homa_ir'])
        if 'homa_ir_log' in X.columns:
            X = X.drop(columns=['homa_ir_log'])
            
        y = df['ir_label']
        
        # Simple imputation for RF
        X_numeric = X.select_dtypes(include=['number'])
        # Force conversion to numeric, coercing errors to NaN
        X_numeric = X_numeric.apply(pd.to_numeric, errors='coerce')
        X = X_numeric.fillna(X_numeric.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        logger.info(f"Quick RF AUC: {auc:.4f}")
        
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        logger.info("Top 10 Feature Importances:")
        logger.info(importances.head(10))

if __name__ == "__main__":
    audit_data()

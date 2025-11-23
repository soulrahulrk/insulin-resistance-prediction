"""Preprocessing and feature engineering."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
import joblib

from src.config import (
    KNN_NEIGHBORS, FEATURE_SELECTION_THRESHOLD, MAX_CATEGORICAL_CARDINALITY,
    RANDOM_STATE, FEATURE_TRANSFORMER_PATH, SELECTED_FEATURES_PATH
)
from src.utils import configure_logger, save_json

logger = configure_logger(__name__)


class Preprocessor:
    """Preprocessor for data transformation and feature engineering."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.knn_imputer = None
        self.ordinal_encoders = {}
        self.onehot_encoders = {}
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_names = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.target_col = None
        
    def fit(self, df: pd.DataFrame, y: pd.Series = None) -> "Preprocessor":
        """Fit preprocessor on data.
        
        Args:
            df: Input features DataFrame.
            y: Target series (required for feature selection).
            
        Returns:
            Self.
        """
        logger.info("Fitting preprocessor...")
        
        self.target_col = "ir_label"
        
        # Separate numeric and categorical columns
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Remove columns with all NaN values
        valid_numeric_cols = [col for col in self.numeric_cols if not df[col].isna().all()]
        if len(valid_numeric_cols) < len(self.numeric_cols):
            logger.warning(f"Removed {len(self.numeric_cols) - len(valid_numeric_cols)} numeric columns with all NaN values")
            self.numeric_cols = valid_numeric_cols
        
        logger.info(f"Numeric columns: {len(self.numeric_cols)}")
        logger.info(f"Categorical columns: {len(self.categorical_cols)}")
        
        # Fit KNN imputer for numeric columns
        if self.numeric_cols:
            self.knn_imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
            self.knn_imputer.fit(df[self.numeric_cols])
            logger.info("Fitted KNN imputer")
        
        # Fit categorical encoders
        for col in self.categorical_cols:
            if df[col].nunique() <= MAX_CATEGORICAL_CARDINALITY:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                enc.fit(df[[col]])
                self.onehot_encoders[col] = enc
            else:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                enc.fit(df[[col]])
                self.ordinal_encoders[col] = enc
        
        logger.info(f"Fitted {len(self.onehot_encoders)} OneHot and {len(self.ordinal_encoders)} Ordinal encoders")
        
        # Feature selection using mutual information
        if y is not None:
            X_imputed = df[self.numeric_cols].copy()
            if self.knn_imputer:
                X_imputed = pd.DataFrame(
                    self.knn_imputer.transform(X_imputed),
                    columns=self.numeric_cols,
                    index=df.index
                )
            
            mi_scores = mutual_info_classif(X_imputed, y, random_state=RANDOM_STATE)
            self.selected_features = [col for col, score in zip(self.numeric_cols, mi_scores)
                                     if score >= FEATURE_SELECTION_THRESHOLD]
            logger.info(f"Selected {len(self.selected_features)} features (MI threshold={FEATURE_SELECTION_THRESHOLD})")
        else:
            self.selected_features = self.numeric_cols
        
        return self
    
    def transform(self, df: pd.DataFrame) -> tuple:
        """Transform data.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Tuple of (X_transformed, column_names).
        """
        X = df.copy()

        # Ensure all expected columns exist
        for col in self.numeric_cols or []:
            if col not in X.columns:
                X[col] = np.nan
        for col in self.categorical_cols or []:
            if col not in X.columns:
                X[col] = "__missing__"
        
        # Impute numeric columns
        if self.numeric_cols and self.knn_imputer:
            X[self.numeric_cols] = self.knn_imputer.transform(X[self.numeric_cols])
        
        # Encode categorical columns (if any) BEFORE feature selection
        categorical_parts = []
        for col in self.categorical_cols:
            if col not in X.columns:
                continue  # Skip if column not in input
            if col in self.onehot_encoders:
                enc_data = self.onehot_encoders[col].transform(X[[col]])
                feature_names = self.onehot_encoders[col].get_feature_names_out([col])
                categorical_parts.append(pd.DataFrame(enc_data, columns=feature_names, index=X.index))
                X = X.drop(columns=[col])  # Drop original categorical column
            elif col in self.ordinal_encoders:
                enc_data = self.ordinal_encoders[col].transform(X[[col]])
                X[col] = enc_data  # Replace with encoded values
        
        # Concatenate encoded categorical features
        if categorical_parts:
            X = pd.concat([X.reset_index(drop=True)] + [c.reset_index(drop=True) for c in categorical_parts], axis=1)
        
        # Select features (now includes both numeric and encoded categorical)
        available_features = [f for f in self.selected_features if f in X.columns]
        X = X[available_features]
        
        return X, X.columns.tolist()
    
    def fit_transform(self, df: pd.DataFrame, y: pd.Series = None) -> tuple:
        """Fit and transform data in one step.
        
        Args:
            df: Input DataFrame.
            y: Target series.
            
        Returns:
            Tuple of (X_transformed, column_names).
        """
        self.fit(df, y)
        return self.transform(df)
    
    def save(self, path: Path = FEATURE_TRANSFORMER_PATH) -> None:
        """Save preprocessor to disk.
        
        Args:
            path: Path to save preprocessor.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved preprocessor to {path}")
        
        # Save selected features as JSON
        save_json(SELECTED_FEATURES_PATH, self.selected_features)
        logger.info(f"Saved selected features to {SELECTED_FEATURES_PATH}")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features based on EDA findings.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        DataFrame with additional engineered features.
    """
    df = df.copy()
    
    # HOMA-IR: (glucose * insulin) / 405
    if "fasting_glucose" in df.columns and "fasting_insulin" in df.columns:
        df["homa_ir"] = (df["fasting_glucose"] * df["fasting_insulin"]) / 405.0
        df["homa_ir"] = np.log1p(df["homa_ir"])  # Log transform
        logger.info("Computed HOMA-IR")
    
    # QUICKI: 1 / (log(insulin) + log(glucose))
    if "fasting_glucose" in df.columns and "fasting_insulin" in df.columns:
        safe_glucose = df["fasting_glucose"].clip(lower=1)
        safe_insulin = df["fasting_insulin"].clip(lower=1)
        df["quicki"] = 1.0 / (np.log(safe_insulin) + np.log(safe_glucose))
        logger.info("Computed QUICKI")
    
    # TG/HDL ratio
    if "triglycerides" in df.columns and "hdl_cholesterol" in df.columns:
        hdl_safe = df["hdl_cholesterol"].clip(lower=0.1)
        df["tg_hdl_ratio"] = df["triglycerides"] / hdl_safe
        logger.info("Computed TG/HDL ratio")
    
    # Waist-to-hip ratio
    if "waist_circumference" in df.columns and "hip_circumference" in df.columns:
        hip_safe = df["hip_circumference"].clip(lower=0.1)
        df["waist_hip_ratio"] = df["waist_circumference"] / hip_safe
        logger.info("Computed waist-to-hip ratio")
    
    # Age × BMI interaction
    if "age" in df.columns and "bmi" in df.columns:
        df["age_bmi"] = df["age"] * df["bmi"]
        logger.info("Computed age × BMI interaction")
    
    # Log transform fasting_insulin
    if "fasting_insulin" in df.columns:
        df["fasting_insulin_log"] = np.log1p(df["fasting_insulin"])
    
    logger.info(f"Added engineered features. Total columns now: {len(df.columns)}")
    return df

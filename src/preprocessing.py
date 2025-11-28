"""Preprocessing and feature engineering."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
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
        self.knn_imputer: Optional[KNNImputer] = None  # Only for fasting_insulin
        self.medians: dict[str, float] = {}
        self.ordinal_encoders: dict[str, OrdinalEncoder] = {}
        self.onehot_encoders: dict[str, OneHotEncoder] = {}
        self.scaler = StandardScaler()
        self.selected_features: List[str] | None = None
        self.feature_names: List[str] | None = None
        self.numeric_cols: List[str] | None = None
        self.categorical_cols: List[str] | None = None
        self.target_col: Optional[str] = None
        
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "Preprocessor":
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
        
        # Selective imputation strategy:
        # - fasting_insulin via KNN
        # - other numeric via median
        if "fasting_insulin" in self.numeric_cols:
            self.knn_imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
            self.knn_imputer.fit(df[["fasting_insulin"]])
            logger.info("Fitted KNN imputer for fasting_insulin")

        for col in self.numeric_cols:
            if col == "fasting_insulin":
                continue
            series = df[col]
            if not series.isna().all():
                self.medians[col] = float(series.median())
        logger.info(f"Stored medians for {len(self.medians)} numeric columns")
        
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
        
        # Feature selection after applying selective imputation
        if y is not None:
            temp = df[self.numeric_cols].copy()
            if self.knn_imputer and "fasting_insulin" in temp.columns:
                temp["fasting_insulin"] = self.knn_imputer.transform(temp[["fasting_insulin"]]).ravel()
            for col in self.numeric_cols:
                if col == "fasting_insulin":
                    continue
                temp[col] = temp[col].fillna(self.medians.get(col, 0.0))
            mi_scores = mutual_info_classif(temp[self.numeric_cols], y, random_state=RANDOM_STATE)
            self.selected_features = [col for col, score in zip(self.numeric_cols, mi_scores)
                                      if score >= FEATURE_SELECTION_THRESHOLD]
            logger.info(f"Selected {len(self.selected_features)} features (MI threshold={FEATURE_SELECTION_THRESHOLD})")
        else:
            self.selected_features = self.numeric_cols
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
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
        
        # Impute fasting_insulin via KNN
        if self.knn_imputer and "fasting_insulin" in self.numeric_cols and "fasting_insulin" in X.columns:
            X["fasting_insulin"] = self.knn_imputer.transform(X[["fasting_insulin"]]).ravel()
        # Median impute remaining numeric columns
        for col in self.numeric_cols:
            if col == "fasting_insulin":
                continue
            if col in X.columns:
                X[col] = X[col].fillna(self.medians.get(col, 0.0))
        
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
    
    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, List[str]]:
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
    """Add engineered features per specification.

    Creates: homa_ir, homa_ir_log, quicki, tg_hdl_ratio, waist_hip_ratio,
    age_bmi, fasting_insulin_log, fasting_glucose_log, triglycerides_log.
    """
    df = df.copy()

    if "fasting_glucose" in df.columns and "fasting_insulin" in df.columns:
        # homa_ir and homa_ir_log removed to prevent leakage
        safe_glucose = df["fasting_glucose"].clip(lower=1)
        safe_insulin = df["fasting_insulin"].clip(lower=1)
        df["quicki"] = 1.0 / (np.log(safe_insulin) + np.log(safe_glucose))
        logger.info("Computed QUICKI (HOMA-IR excluded for leakage prevention)")

    if "triglycerides" in df.columns and "hdl_cholesterol" in df.columns:
        hdl_safe = df["hdl_cholesterol"].clip(lower=0.1)
        df["tg_hdl_ratio"] = df["triglycerides"] / hdl_safe
        logger.info("Computed tg_hdl_ratio")

    if "waist_circumference" in df.columns and "hip_circumference" in df.columns:
        hip_safe = df["hip_circumference"].clip(lower=0.1)
        df["waist_hip_ratio"] = df["waist_circumference"] / hip_safe
        logger.info("Computed waist_hip_ratio")

    if "age" in df.columns and "bmi" in df.columns:
        df["age_bmi"] = df["age"] * df["bmi"]
        logger.info("Computed age_bmi interaction")

    if "fasting_insulin" in df.columns:
        df["fasting_insulin_log"] = np.log1p(df["fasting_insulin"])
    if "fasting_glucose" in df.columns:
        df["fasting_glucose_log"] = np.log1p(df["fasting_glucose"])
    if "triglycerides" in df.columns:
        df["triglycerides_log"] = np.log1p(df["triglycerides"])

    logger.info(f"Added engineered features. Total columns now: {len(df.columns)}")
    return df

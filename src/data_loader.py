"""Data loading module."""

import pandas as pd
import numpy as np
from src.config import DATA_PATH
from src.utils import configure_logger

logger = configure_logger(__name__)


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to snake_case.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        DataFrame with snake_case column names.
    """
    df = df.copy()
    df.columns = [col.lower().replace(" ", "_").replace("-", "_") for col in df.columns]
    return df


def load_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load and validate insulin resistance dataset.
    
    Args:
        data_path: Path to CSV file.
        
    Returns:
        Validated DataFrame.
        
    Raises:
        FileNotFoundError: If data file not found.
        ValueError: If required columns missing or data invalid.
    """
    logger.info(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}") from e
    
    # Canonicalize column names
    df = canonicalize_columns(df)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Create IR label if not present
    if "ir_label" not in df.columns:
        logger.info("Creating 'ir_label' from glucose and insulin using HOMA-IR")
        
        # Check for required columns
        if "glucose" not in df.columns or "insulin" not in df.columns:
            raise ValueError("Cannot create ir_label: 'glucose' and 'insulin' columns required. "
                           f"Available columns: {df.columns.tolist()}")
        
        # Calculate HOMA-IR: (glucose * insulin) / 405
        # HOMA-IR > 2.5 typically indicates insulin resistance
        df["homa_ir"] = (df["glucose"] * df["insulin"]) / 405.0
        df["ir_label"] = (df["homa_ir"] > 2.5).astype(int)
        
        # Remove rows with missing values in glucose or insulin
        initial_len = len(df)
        df = df.dropna(subset=["glucose", "insulin", "ir_label"])
        if len(df) < initial_len:
            logger.warning(f"Dropped {initial_len - len(df)} rows with missing glucose/insulin")
        
        logger.info(f"Created 'ir_label' based on HOMA-IR threshold 2.5")
    
    logger.info(f"Target variable 'ir_label' distribution:\n{df['ir_label'].value_counts()}")
    
    # Basic sanity checks
    if "age" in df.columns:
        invalid_ages = (df["age"] < 0) | (df["age"] > 150)
        if invalid_ages.sum() > 0:
            logger.warning(f"Found {invalid_ages.sum()} records with invalid age values")
            df = df[~invalid_ages]
    
    if "fasting_glucose" in df.columns:
        invalid_glucose = (df["fasting_glucose"] < 40) | (df["fasting_glucose"] > 600)
        if invalid_glucose.sum() > 0:
            logger.warning(f"Found {invalid_glucose.sum()} records with glucose outside [40, 600]")
            df = df[~invalid_glucose]
    
    logger.info(f"Data validation complete: {len(df)} records remaining")
    return df

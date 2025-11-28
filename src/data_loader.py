"""Data loading module."""

import pandas as pd
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


def load_data(data_path: str = DATA_PATH, drop_label_leak_features: bool = True) -> pd.DataFrame:
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
    
    # Harmonize glucose/insulin naming to fasting_glucose/fasting_insulin
    if "glucose" in df.columns and "fasting_glucose" not in df.columns:
        df = df.rename(columns={"glucose": "fasting_glucose"})
    if "insulin" in df.columns and "fasting_insulin" not in df.columns:
        df = df.rename(columns={"insulin": "fasting_insulin"})

    # Create IR label if not present
    if "ir_label" not in df.columns:
        logger.info("Creating 'ir_label' from fasting_glucose and fasting_insulin using HOMA-IR")
        
        # We need both columns to create the label
        df = df.dropna(subset=["fasting_glucose", "fasting_insulin"])
        
        homa_ir = (df["fasting_glucose"] * df["fasting_insulin"]) / 405.0
        df["ir_label"] = (homa_ir > 2.5).astype(int)
        
        if drop_label_leak_features:
            # Remove direct label-construction artifact to reduce trivial leakage
            if "homa_ir" in df.columns:
                df = df.drop(columns=["homa_ir"])
        logger.info(f"Created 'ir_label'. Rows remaining: {len(df)}")
    
    if "ir_label" in df.columns:
        logger.info(f"Target variable 'ir_label' distribution:\n{df['ir_label'].value_counts()}")
    
    # Basic sanity checks
    if "age" in df.columns:
        invalid_ages = (df["age"] < 0) | (df["age"] > 150)
        if invalid_ages.sum() > 0:
            logger.warning(f"Found {invalid_ages.sum()} records with invalid age values")
            df = df[~invalid_ages]
    
    if "fasting_glucose" in df.columns:
        df["fasting_glucose"] = df["fasting_glucose"].clip(lower=40, upper=600)
    if "fasting_insulin" in df.columns:
        df["fasting_insulin"] = df["fasting_insulin"].clip(lower=0.1)
    
    logger.info(f"Data validation complete: {len(df)} records remaining")
    return df

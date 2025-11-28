"""Utility functions for the ML pipeline.

Enhanced for strict reproducibility and centralized artifact logging.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from src.config import LOG_FORMAT, LOG_LEVEL, LOGS_DIR, TRAIN_LOG_PATH, RANDOM_STATE


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set random seed for reproducibility across common libs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    np.random.seed(seed)
    # Optional: deterministic for torch if available
    # Deep learning frameworks optional; skipped if unavailable.


def ensure_reproducibility() -> None:
    """Invoke all reproducibility safeguards (idempotent)."""
    set_seed(RANDOM_STATE)
    # Limit threads to reduce nondeterminism in linear algebra (optional override later)
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(var, "1")


def configure_logger(name: str = "ir_prediction", log_path: Path | None = None) -> logging.Logger:
    """Configure and return logger.
    
    Args:
        name: Logger name.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Use TRAIN_LOG_PATH for training-related logs if not explicitly overridden
    if log_path is None and ("train" in name or name == "ir_prediction"):
        target_path = TRAIN_LOG_PATH
    else:
        target_path = log_path or (LOGS_DIR / f"{name}.log")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(target_path, encoding='utf-8')
    file_handler.setLevel(LOG_LEVEL)
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def save_json(path: Path, obj: Any) -> None:
    """Save object to JSON file.
    
    Args:
        path: File path to save to.
        obj: Object to save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(path: Path) -> Any:
    """Load object from JSON file.
    
    Args:
        path: File path to load from.
        
    Returns:
        Loaded object.
    """
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(path: Path, obj: Any) -> None:
    """Save object to pickle file.
    
    Args:
        path: File path to save to.
        obj: Object to save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """Load object from pickle file.
    
    Args:
        path: File path to load from.
        
    Returns:
        Loaded object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


FloatArray = NDArray[np.float64]


def compute_ece(y_true: FloatArray, y_prob: FloatArray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities (confidence scores).
        n_bins: Number of bins for ECE computation.
        
    Returns:
        ECE value between 0 and 1.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if in_bin.sum() == 0:
            continue
        
        bin_acc = (y_true[in_bin] == (y_prob[in_bin] >= 0.5)).astype(float).mean()
        bin_conf = y_prob[in_bin].mean()
        ece += np.abs(bin_acc - bin_conf) * in_bin.mean()
    
    return ece

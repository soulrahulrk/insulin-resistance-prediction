"""Model evaluation module."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc, matthews_corrcoef,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
import json

from src.config import PERFORMANCE_METRICS_PATH
from src.utils import configure_logger, compute_ece, save_json

logger = configure_logger(__name__)


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        
    Returns:
        Dictionary of metrics.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'brier_score': float(brier_score_loss(y_true, y_prob)),
        'ece': float(compute_ece(y_true, y_prob)),
    }
    
    return metrics


def calibration_compare(model, X_val: np.ndarray, y_val: np.ndarray) -> tuple:
    """Compare sigmoid and isotonic calibration methods.
    
    Args:
        model: Fitted base model.
        X_val: Validation features.
        y_val: Validation target.
        
    Returns:
        Tuple of (best_calibrated_model, calibration_stats).
    """
    results = {}
    
    for method in ['sigmoid', 'isotonic']:
        cal_model = CalibratedClassifierCV(model, method=method, cv=3)
        cal_model.fit(X_val, y_val)
        
        y_prob_cal = cal_model.predict_proba(X_val)[:, 1]
        
        stats = {
            'brier_score': float(brier_score_loss(y_val, y_prob_cal)),
            'ece': float(compute_ece(y_val, y_prob_cal)),
        }
        
        results[method] = (cal_model, stats)
        logger.info(f"{method}: Brier={stats['brier_score']:.4f}, ECE={stats['ece']:.4f}")
    
    # Choose best calibration method
    best_method = min(results.keys(), key=lambda m: results[m][1]['brier_score'])
    best_model, best_stats = results[best_method]
    
    logger.info(f"✓ Selected calibration method: {best_method}")
    return best_model, best_stats


def find_optimal_thresholds(y_val: np.ndarray, y_prob: np.ndarray) -> dict:
    """Find optimal decision thresholds using multiple strategies.
    
    Args:
        y_val: Validation target.
        y_prob: Predicted probabilities.
        
    Returns:
        Dictionary of thresholds and metrics.
    """
    fpr, tpr, thresholds_roc = roc_curve(y_val, y_prob)
    
    # F1-maximization
    best_f1 = 0.0
    best_threshold_f1 = 0.5
    for thresh in np.arange(0.0, 1.01, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold_f1 = thresh
    
    # Youden's J
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold_youden = thresholds_roc[best_idx]
    
    # Sensitivity >= 0.90 with highest specificity
    best_threshold_sens = 0.5
    best_spec = 0.0
    for thresh in np.arange(0.0, 1.01, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        sens = recall_score(y_val, y_pred, zero_division=0)
        if sens >= 0.90:
            spec = 1.0 - (np.sum((y_pred == 1) & (y_val == 0)) / np.sum(y_val == 0))
            if spec > best_spec:
                best_spec = spec
                best_threshold_sens = thresh
    
    result = {
        'f1_max': {
            'threshold': float(best_threshold_f1),
            'f1': float(best_f1),
        },
        'youden': {
            'threshold': float(best_threshold_youden),
            'j_score': float(np.max(j_scores)),
        },
        'sensitivity_first': {
            'threshold': float(best_threshold_sens),
            'sensitivity': 0.90,
            'specificity': float(best_spec),
        },
    }
    
    logger.info(f"Optimal thresholds: F1={result['f1_max']['threshold']:.2f}, "
                f"Youden={result['youden']['threshold']:.2f}, "
                f"SensFirst={result['sensitivity_first']['threshold']:.2f}")
    
    return result

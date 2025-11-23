"""Ensemble training module with stacking."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.config import (
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS,
    GRADBOOST_PARAMS, N_FOLDS, RANDOM_STATE, META_LEARNER_C_GRID
)
from src.utils import configure_logger
from src.evaluate import evaluate_model

logger = configure_logger(__name__)


def get_oof_predictions(models: Dict, X_train: pd.DataFrame, y_train: pd.Series, 
                       X_val: pd.DataFrame, n_folds: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Generate out-of-fold (OOF) predictions for stacking.
    
    Args:
        models: Dict of base models.
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        n_folds: Number of folds for CV.
        
    Returns:
        Tuple of (train_meta_features, val_meta_features).
    """
    n_models = len(models)
    n_train, n_val = len(X_train), len(X_val)
    
    train_meta = np.zeros((n_train, n_models))
    val_meta = np.zeros((n_val, n_models))
    
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    for model_idx, (name, model) in enumerate(models.items()):
        logger.info(f"Generating OOF for model {model_idx+1}/{n_models}: {name}")
        
        val_preds = np.zeros(n_val)
        
        for fold, (train_idx, hold_idx) in enumerate(kf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_hold = X_train.iloc[hold_idx]
            
            model_fold = clone(model)  # Clone model
            model_fold.fit(X_fold_train, y_fold_train)
            
            train_meta[hold_idx, model_idx] = model_fold.predict_proba(X_fold_hold)[:, 1]
            val_preds += model_fold.predict_proba(X_val)[:, 1] / n_folds
        
        val_meta[:, model_idx] = val_preds
        logger.info(f"✓ Completed OOF for {name}")
    
    logger.info(f"OOF features shape: train={train_meta.shape}, val={val_meta.shape}")
    return train_meta, val_meta


def train_meta_learner(meta_X_train: np.ndarray, meta_y_train: np.ndarray) -> Tuple:
    """Train meta-learner with optimal hyperparameters.
    
    Args:
        meta_X_train: OOF training features.
        meta_y_train: Training target.
        
    Returns:
        Tuple of (fitted_meta_learner_pipeline, best_C).
    """
    logger.info(f"Training meta-learner on {meta_X_train.shape} OOF features")
    
    best_C = 1.0
    best_score = 0.0
    
    for C in META_LEARNER_C_GRID:
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(meta_X_train)
        
        meta_model = LogisticRegression(C=C, class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)
        scores = cross_val_score(meta_model, scaled_X, meta_y_train, cv=3, scoring='roc_auc')
        
        mean_score = scores.mean()
        logger.info(f"C={C}: CV AUC={mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_C = C
    
    logger.info(f"✓ Best C={best_C} (CV AUC={best_score:.4f})")
    
    # Train final meta-learner with best C
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(meta_X_train)
    meta_model = LogisticRegression(C=best_C, class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)
    meta_model.fit(scaled_X, meta_y_train)
    
    return (scaler, meta_model), best_C


def build_stacking_classifier(base_models: Dict, meta_learner_pipeline: Tuple, 
                             meta_X_train: np.ndarray, meta_X_val: np.ndarray,
                             y_val: np.ndarray) -> Tuple[Dict, np.ndarray]:
    """Build and validate stacking classifier.
    
    Args:
        base_models: Dict of trained base models.
        meta_learner_pipeline: Tuple of (scaler, meta_model).
        meta_X_train: OOF training features for meta-learner.
        meta_X_val: OOF validation features for prediction.
        y_val: Validation target.
        
    Returns:
        Tuple of (stacker_dict, validation_predictions).
    """
    scaler, meta_model = meta_learner_pipeline
    
    # Make validation predictions
    scaled_meta_X_val = scaler.transform(meta_X_val)
    val_proba = meta_model.predict_proba(scaled_meta_X_val)[:, 1]

    # Evaluate the stacking model
    metrics = evaluate_model(y_val, val_proba)
    logger.info(f"Stacking classifier validated - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")

    stacker = {
        'base_models': base_models,
        'scaler': scaler,
        'meta_learner': meta_model,
        'val_auc': metrics['roc_auc'],
        'val_f1': metrics['f1']
    }

    return stacker, val_proba

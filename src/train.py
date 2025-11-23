"""Main training pipeline."""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from src.config import (
    DATA_PATH, RANDOM_STATE, TEST_SIZE, VAL_SIZE, TRAIN_SIZE,
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS, GRADBOOST_PARAMS,
    ENSEMBLE_MODEL_PATH, FEATURE_TRANSFORMER_PATH, SELECTED_FEATURES_PATH,
    OPTIMAL_THRESHOLD_PATH, PERFORMANCE_METRICS_PATH, BASE_MODELS_METRICS_PATH
)
from src.data_loader import load_data
from src.preprocessing import Preprocessor
from src.features import compute_engineered_features
from src.ensemble import get_oof_predictions, train_meta_learner, build_stacking_classifier
from src.evaluate import evaluate_model, calibration_compare, find_optimal_thresholds
from src.utils import configure_logger, set_seed, save_json, save_pickle
from src.stacker_wrapper import StackerWrapper

logger = configure_logger(__name__)
set_seed(RANDOM_STATE)


def train_base_learners(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train base learner models.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        
    Returns:
        Dictionary of trained models.
    """
    models = {}
    metrics_list = []
    
    # XGBoost
    logger.info("Training XGBoost...")
    start = time.time()
    xgb = XGBClassifier(**XGBOOST_PARAMS)
    xgb.fit(X_train, y_train, verbose=False)
    models['xgboost'] = xgb
    logger.info(f"✓ XGBoost trained in {time.time() - start:.2f}s")
    
    # LightGBM
    logger.info("Training LightGBM...")
    start = time.time()
    lgb = LGBMClassifier(**LIGHTGBM_PARAMS)
    lgb.fit(X_train, y_train)
    models['lightgbm'] = lgb
    logger.info(f"✓ LightGBM trained in {time.time() - start:.2f}s")
    
    # CatBoost
    logger.info("Training CatBoost...")
    start = time.time()
    cb = CatBoostClassifier(**CATBOOST_PARAMS)
    cb.fit(X_train, y_train)
    models['catboost'] = cb
    logger.info(f"✓ CatBoost trained in {time.time() - start:.2f}s")
    
    # Gradient Boosting
    logger.info("Training GradientBoosting...")
    start = time.time()
    gb = GradientBoostingClassifier(**GRADBOOST_PARAMS)
    gb.fit(X_train, y_train)
    models['gradboost'] = gb
    logger.info(f"✓ GradientBoosting trained in {time.time() - start:.2f}s")
    
    logger.info(f"✓ All {len(models)} base models trained")
    return models


def train_pipeline():
    """Execute complete training pipeline."""
    logger.info("="*80)
    logger.info("STARTING INSULIN RESISTANCE PREDICTION PIPELINE")
    logger.info("="*80)
    
    start_time = time.time()
    
    # 1. Load data
    logger.info("\n[1/8] LOADING DATA...")
    df = load_data(DATA_PATH)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # 2. Add engineered features
    logger.info("\n[2/8] ENGINEERING FEATURES...")
    df = compute_engineered_features(df)
    logger.info(f"Total features after engineering: {len(df.columns)}")
    
    # 3. Preprocess
    logger.info("\n[3/8] PREPROCESSING...")
    X = df.drop(columns=['ir_label'])
    y = df['ir_label']
    
    preprocessor = Preprocessor()
    X_processed, feature_names = preprocessor.fit_transform(X, y)
    preprocessor.save()
    
    logger.info(f"Selected {len(preprocessor.selected_features)} features")
    
    # 4. Train/val/test split (70/15/15)
    logger.info("\n[4/8] SPLITTING DATA...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_processed, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE), stratify=y_temp, random_state=RANDOM_STATE
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 5. Train base learners
    logger.info("\n[5/8] TRAINING BASE LEARNERS...")
    base_models = train_base_learners(X_train, y_train)
    
    # Evaluate base models on validation
    base_metrics = {}
    for name, model in base_models.items():
        y_prob = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_model(y_val.values, y_prob)
        base_metrics[name] = metrics
        logger.info(f"{name} - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
    
    # Save base model metrics
    metrics_df = pd.DataFrame(base_metrics).T
    metrics_df.to_csv(BASE_MODELS_METRICS_PATH)
    logger.info(f"Saved base model metrics to {BASE_MODELS_METRICS_PATH}")
    
    # 6. Build stacking ensemble
    logger.info("\n[6/8] BUILDING STACKING ENSEMBLE...")
    train_meta, val_meta = get_oof_predictions(base_models, X_train, y_train, X_val)
    meta_learner_pipeline, best_C = train_meta_learner(train_meta, y_train.values)
    stacker, stacker_val_proba = build_stacking_classifier(
        base_models, meta_learner_pipeline, train_meta, val_meta, y_val.values
    )
    
    logger.info(f"Stacking - AUC: {stacker['val_auc']:.4f}, F1: {stacker['val_f1']:.4f}")

    # 7. Calibrate
    logger.info("\n[7/8] CALIBRATING ENSEMBLE...")
    # The meta-learner is already a standard scikit-learn classifier, so we can use it directly.
    # We calibrate it on the out-of-fold validation predictions (val_meta).
    cal_model, cal_stats = calibration_compare(
        stacker['meta_learner'], val_meta, y_val.values
    )
    logger.info(f"Calibration - Brier: {cal_stats['brier_score']:.4f}, ECE: {cal_stats['ece']:.4f}")

    # 8. Optimize thresholds
    logger.info("\n[8/8] OPTIMIZING THRESHOLDS...")
    thresholds = find_optimal_thresholds(y_val.values, stacker_val_proba)
    optimal_threshold = thresholds['f1_max']['threshold']
    
    # Save optimal threshold
    with open(OPTIMAL_THRESHOLD_PATH, 'w') as f:
        f.write(str(optimal_threshold))
    logger.info(f"Saved optimal threshold: {optimal_threshold:.4f}")
    
    # Save final ensemble
    final_model_package = {
        'base_models': base_models,
        'model_order': list(base_models.keys()),
        'calibrated_model': cal_model,
    }
    save_pickle(ENSEMBLE_MODEL_PATH, final_model_package)
    logger.info(f"Saved ensemble model to {ENSEMBLE_MODEL_PATH}")
    
    # Save performance metrics
    performance_metrics = {
        'base_models': base_metrics,
        'stacking': {
            'val_auc': stacker['val_auc'],
            'val_f1': stacker['val_f1'],
        },
        'calibration': cal_stats,
        'thresholds': thresholds,
        'optimal_threshold': float(optimal_threshold),
    }
    save_json(PERFORMANCE_METRICS_PATH, performance_metrics)
    
    total_time = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING COMPLETE in {total_time:.2f}s")
    logger.info(f"{'='*80}\n")
    
    return stacker, optimal_threshold


if __name__ == "__main__":
    try:
        train_pipeline()
        logger.info("✓ Training pipeline completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"✗ Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

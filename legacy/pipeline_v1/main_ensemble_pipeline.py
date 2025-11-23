"""
MAIN ENSEMBLE ORCHESTRATION PIPELINE
Insulin Resistance Prediction

Workflow:
  1. Load preprocessed data
  2. Train base learners (XGB, LightGBM, CatBoost, GradBoost)
  3. Stacking: Cross-val meta-features → Logistic Regression
  4. Blending: Hold-out meta-features → Ridge
  5. Calibration: Isotonic regression on hold-out set
  6. Threshold Optimization: Medical ROI maximization
  7. Comparison & Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from ensemble_stacking import (
    StackingEnsemble, BlendingEnsemble, CalibrationWrapper, 
    ThresholdOptimizer, evaluate_ensemble, clone_model
)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'random_state': 42,
    'test_size': 0.20,
    'val_size': 0.10,  # For calibration
    'n_jobs': -1,
    'verbose': True
}

RESULTS_DIR = Path('results/ensemble')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: LOAD & PREPARE DATA
# ============================================================================

def load_and_prepare_data(data_path, target_col='ir_label', test_size=0.20, val_size=0.10):
    """
    Load, scale, and split data into train, validation, and test.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    print("[PIPELINE] Loading data...")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(columns=[target_col, 'patient_id'], errors='ignore')
    y = df[target_col]
    
    feature_names = X.columns.tolist()
    print(f"  Features: {len(feature_names)}, Samples: {len(X)}, Positive: {(y==1).sum()} ({100*y.mean():.1f}%)")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=CONFIG['random_state'], stratify=y
    )
    
    # Second split: train vs val (for calibration)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=CONFIG['random_state'], stratify=y_temp
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for model compatibility
    X_train = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_val = pd.DataFrame(X_val_scaled, columns=feature_names, index=X_val.index)
    X_test = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
    
    print(f"  Train: {X_train.shape[0]}, Val (Calib): {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


# ============================================================================
# STEP 2: TRAIN BASE LEARNERS (LEVEL-0)
# ============================================================================

def train_base_learners(X_train, y_train, verbose=True):
    """
    Train XGBoost, LightGBM, CatBoost, and Gradient Boosting.
    
    Returns:
        dict of {model_name: trained_model}
    """
    print("\n[PIPELINE] Training Level-0 Base Learners...")
    
    base_models = {
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=CONFIG['random_state'],
            early_stopping_rounds=20, verbose=0
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=CONFIG['random_state'], verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.05,
            subsample=0.8, random_state=CONFIG['random_state'],
            verbose=0, thread_count=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=CONFIG['random_state'],
            validation_fraction=0.1, n_iter_no_change=20
        )
    }
    
    trained_models = {}
    base_results = {}
    
    for model_name, model in base_models.items():
        print(f"  Training {model_name}...", end=" ")
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        # Evaluate on training set
        y_pred_proba = model.predict_proba(X_train)[:, 1]
        auc = roc_auc_score(y_train, y_pred_proba)
        base_results[model_name] = auc
        print(f"AUC={auc:.4f} ✓")
    
    return trained_models, base_results


# ============================================================================
# STEP 3: STACKING ENSEMBLE
# ============================================================================

def create_stacking_ensemble(base_models, X_train, X_val, X_test, y_train, y_val, verbose=True):
    """Train stacking ensemble with Logistic Regression as meta-learner."""
    
    print("\n[PIPELINE] Creating STACKING Ensemble (5-fold CV)...")
    
    meta_learner = LogisticRegression(max_iter=1000, random_state=CONFIG['random_state'])
    
    stacker = StackingEnsemble(
        level0_models=base_models,
        level1_model=meta_learner,
        n_splits=5,
        random_state=CONFIG['random_state']
    )
    
    stacker.fit(X_train, y_train, verbose=verbose)
    
    y_pred_proba_val = stacker.predict_proba(X_val)[:, 1]
    y_pred_proba_test = stacker.predict_proba(X_test)[:, 1]
    
    auc_val = roc_auc_score(y_val, y_pred_proba_val)
    auc_test = roc_auc_score(y_test, y_pred_proba_test)
    
    print(f"  Stacking AUC (Val): {auc_val:.4f}, AUC (Test): {auc_test:.4f}")
    
    return stacker, y_pred_proba_val, y_pred_proba_test


# ============================================================================
# STEP 4: BLENDING ENSEMBLE
# ============================================================================

def create_blending_ensemble(base_models, X_train, X_val, X_test, y_train, y_val, verbose=True):
    """Train blending ensemble with Ridge as meta-learner."""
    
    print("\n[PIPELINE] Creating BLENDING Ensemble (hold-out split)...")
    
    meta_learner = Ridge(alpha=1.0)
    
    blender = BlendingEnsemble(
        level0_models=base_models,
        level1_model=meta_learner,
        val_size=0.4,
        random_state=CONFIG['random_state']
    )
    
    blender.fit(X_train, y_train, verbose=verbose)
    
    y_pred_proba_val = blender.predict_proba(X_val)[:, 1]
    y_pred_proba_test = blender.predict_proba(X_test)[:, 1]
    
    auc_val = roc_auc_score(y_val, y_pred_proba_val)
    auc_test = roc_auc_score(y_test, y_pred_proba_test)
    
    print(f"  Blending AUC (Val): {auc_val:.4f}, AUC (Test): {auc_test:.4f}")
    
    return blender, y_pred_proba_val, y_pred_proba_test


# ============================================================================
# STEP 5: MODEL CALIBRATION
# ============================================================================

def calibrate_ensemble(ensemble_model, X_val, y_val, method='isotonic', model_name='Ensemble'):
    """Apply isotonic regression or Platt scaling for calibration."""
    
    print(f"\n[PIPELINE] Calibrating {model_name} ({method})...")
    
    calibrated = CalibrationWrapper(ensemble_model, method=method)
    calibrated.fit(X_val, y_val)
    
    y_pred_proba_calib = calibrated.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba_calib)
    print(f"  Calibrated {model_name} AUC (Val): {auc:.4f}")
    
    return calibrated


# ============================================================================
# STEP 6: THRESHOLD OPTIMIZATION
# ============================================================================

def optimize_thresholds(y_val, y_pred_proba_val, y_test, y_pred_proba_test, model_name='Ensemble'):
    """Find optimal thresholds for F1, Youden, sensitivity@90% specificity."""
    
    print(f"\n[PIPELINE] Optimizing thresholds for {model_name}...")
    
    strategies = ['f1', 'youden', 'sensitivity@specificity']
    results = {}
    
    for strategy in strategies:
        if strategy == 'sensitivity@specificity':
            optimizer = ThresholdOptimizer(y_val, y_pred_proba_val, strategy=strategy)
            threshold, score = optimizer.find_optimal(specificity_target=0.90)
        else:
            optimizer = ThresholdOptimizer(y_val, y_pred_proba_val, strategy=strategy)
            threshold, score = optimizer.find_optimal()
        
        # Evaluate on test
        y_pred = (y_pred_proba_test >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba_test)
        
        results[strategy] = {
            'threshold': threshold,
            'val_score': score,
            'test_f1': f1,
            'test_auc': auc
        }
        
        print(f"  {strategy:30s} → threshold={threshold:.3f}, test_f1={f1:.4f}, test_auc={auc:.4f}")
    
    return results


# ============================================================================
# STEP 7: COMPREHENSIVE COMPARISON
# ============================================================================

def compare_all_models(results_dict, X_test, y_test):
    """Generate comparison table and visualization."""
    
    print("\n[PIPELINE] Generating final comparison...\n")
    
    comparison = []
    
    for model_name, preds in results_dict.items():
        y_proba = preds['y_proba_test']
        auc = roc_auc_score(y_test, y_proba)
        
        # F1 at optimal threshold
        y_pred = (y_proba >= preds['optimal_threshold_f1']).astype(int)
        f1 = f1_score(y_test, y_pred)
        
        comparison.append({
            'Model': model_name,
            'AUC-ROC': auc,
            'F1 (F1-opt)': f1,
            'Threshold (F1)': preds['optimal_threshold_f1']
        })
    
    df_comparison = pd.DataFrame(comparison).sort_values('AUC-ROC', ascending=False)
    print(df_comparison.to_string(index=False))
    
    # Save to CSV
    df_comparison.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
    print(f"\n  Saved: {RESULTS_DIR / 'model_comparison.csv'}")
    
    return df_comparison


# ============================================================================
# STEP 8: VISUALIZATION
# ============================================================================

def plot_roc_curves(results_dict, y_test, figsize=(12, 8)):
    """Plot ROC curves for all models."""
    
    plt.figure(figsize=figsize)
    
    for model_name, preds in results_dict.items():
        y_proba = preds['y_proba_test']
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Ensemble Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(RESULTS_DIR / 'roc_curves_comparison.png', dpi=300)
    print(f"\n  Saved: {RESULTS_DIR / 'roc_curves_comparison.png'}")
    
    plt.show()


def plot_threshold_sensitivity(results_dict, y_test):
    """Show F1 vs Threshold for top models."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    thresholds = np.linspace(0, 1, 101)
    
    for idx, (model_name, preds) in enumerate(list(results_dict.items())[:2]):
        y_proba = preds['y_proba_test']
        
        f1_scores = []
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        
        ax = axes[idx]
        ax.plot(thresholds, f1_scores, linewidth=2, label='F1 Score')
        ax.axvline(preds['optimal_threshold_f1'], color='red', linestyle='--', 
                   label=f"Optimal={preds['optimal_threshold_f1']:.3f}")
        ax.set_xlabel('Threshold', fontsize=11)
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'threshold_sensitivity.png', dpi=300)
    print(f"  Saved: {RESULTS_DIR / 'threshold_sensitivity.png'}")
    
    plt.show()


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main(data_path='data/cleaned_all_datasets_merged.csv'):
    """Run full ensemble pipeline."""
    
    print("\n" + "="*80)
    print("ADVANCED ENSEMBLE FRAMEWORK FOR INSULIN RESISTANCE PREDICTION")
    print("="*80)
    
    # 1. Load & Prepare
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_prepare_data(
        data_path, 
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size']
    )
    
    # 2. Train Base Learners
    base_models, base_results = train_base_learners(X_train, y_train, verbose=CONFIG['verbose'])
    
    # Save base results
    with open(RESULTS_DIR / 'base_learners_auc.json', 'w') as f:
        json.dump(base_results, f, indent=2)
    
    # 3. Stacking Ensemble
    stacker, stacker_val_proba, stacker_test_proba = create_stacking_ensemble(
        base_models, X_train, X_val, X_test, y_train, y_val
    )
    
    # 4. Blending Ensemble
    blender, blender_val_proba, blender_test_proba = create_blending_ensemble(
        base_models, X_train, X_val, X_test, y_train, y_val
    )
    
    # 5. Calibration
    stacker_calibrated = calibrate_ensemble(stacker, X_val, y_val, method='isotonic', model_name='Stacking')
    blender_calibrated = calibrate_ensemble(blender, X_val, y_val, method='isotonic', model_name='Blending')
    
    stacker_cal_proba = stacker_calibrated.predict_proba(X_test)[:, 1]
    blender_cal_proba = blender_calibrated.predict_proba(X_test)[:, 1]
    
    # 6. Threshold Optimization
    stacker_thresholds = optimize_thresholds(y_val, stacker_val_proba, y_test, stacker_test_proba, 'Stacking')
    blender_thresholds = optimize_thresholds(y_val, blender_val_proba, y_test, blender_test_proba, 'Blending')
    stacker_cal_thresholds = optimize_thresholds(y_val, stacker_calibrated.predict_proba(X_val)[:, 1], 
                                                  y_test, stacker_cal_proba, 'Stacking (Calibrated)')
    blender_cal_thresholds = optimize_thresholds(y_val, blender_calibrated.predict_proba(X_val)[:, 1], 
                                                  y_test, blender_cal_proba, 'Blending (Calibrated)')
    
    # 7. Aggregate Results
    results_dict = {
        'XGBoost': {
            'y_proba_test': base_models['XGBoost'].predict_proba(X_test)[:, 1],
            'optimal_threshold_f1': 0.5
        },
        'LightGBM': {
            'y_proba_test': base_models['LightGBM'].predict_proba(X_test)[:, 1],
            'optimal_threshold_f1': 0.5
        },
        'Stacking (Uncalibrated)': {
            'y_proba_test': stacker_test_proba,
            'optimal_threshold_f1': stacker_thresholds['f1']['threshold']
        },
        'Stacking (Calibrated)': {
            'y_proba_test': stacker_cal_proba,
            'optimal_threshold_f1': stacker_cal_thresholds['f1']['threshold']
        },
        'Blending (Uncalibrated)': {
            'y_proba_test': blender_test_proba,
            'optimal_threshold_f1': blender_thresholds['f1']['threshold']
        },
        'Blending (Calibrated)': {
            'y_proba_test': blender_cal_proba,
            'optimal_threshold_f1': blender_cal_thresholds['f1']['threshold']
        }
    }
    
    # 8. Compare & Visualize
    df_comparison = compare_all_models(results_dict, X_test, y_test)
    plot_roc_curves(results_dict, y_test)
    plot_threshold_sensitivity(results_dict, y_test)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Results saved to: {RESULTS_DIR}\n")


if __name__ == '__main__':
    main()

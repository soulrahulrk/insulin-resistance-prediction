#!/usr/bin/env python3
"""
Quick-Start Guide for Advanced Ensemble Framework
Insulin Resistance Prediction

Run this to see a minimal working example.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

from ensemble_stacking import StackingEnsemble, ThresholdOptimizer, evaluate_ensemble

# ============================================================================
# EXAMPLE 1: STACKING WITH SYNTHETIC DATA
# ============================================================================

def example_stacking_synthetic():
    """Minimal stacking example on synthetic data."""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: STACKING ENSEMBLE (Synthetic Data)")
    print("="*80)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=5000, n_features=30, n_informative=15,
        n_redundant=5, n_classes=2, random_state=42
    )
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    
    # Define Level-0 models
    base_models = {
        'XGBoost': XGBClassifier(n_estimators=50, max_depth=5, random_state=42, verbose=0),
        'LightGBM': LGBMClassifier(n_estimators=50, max_depth=5, random_state=42, verbose=-1),
    }
    
    # Level-1 meta-learner
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    
    # Create and train stacker
    stacker = StackingEnsemble(
        level0_models=base_models,
        level1_model=meta_learner,
        n_splits=5,
        random_state=42
    )
    
    print("\nTraining stacker...")
    stacker.fit(X_train, y_train, verbose=True)
    
    # Predict
    y_proba = stacker.predict_proba(X_test)[:, 1]
    
    # Evaluate
    auc = roc_auc_score(y_test, y_proba)
    print(f"\nStacking Ensemble AUC: {auc:.4f}")
    
    return stacker


# ============================================================================
# EXAMPLE 2: THRESHOLD OPTIMIZATION
# ============================================================================

def example_threshold_optimization():
    """Show how to find optimal thresholds."""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: THRESHOLD OPTIMIZATION")
    print("="*80)
    
    # Synthetic predictions
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 1000)
    y_proba = np.random.beta(2, 5, 1000)  # Realistic probabilities
    y_proba[y_true == 1] += 0.3  # Shift positive class higher
    y_proba = np.clip(y_proba, 0, 1)
    
    # Split for optimization
    y_val = y_true[:500]
    y_proba_val = y_proba[:500]
    y_test = y_true[500:]
    y_proba_test = y_proba[500:]
    
    # Optimize thresholds
    strategies = ['f1', 'youden']
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        optimizer = ThresholdOptimizer(y_val, y_proba_val, strategy=strategy)
        thresh, score = optimizer.find_optimal()
        
        # Apply to test
        y_pred = (y_proba_test >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred)
        
        print(f"  Optimal Threshold: {thresh:.3f}")
        print(f"  Validation Score: {score:.4f}")
        print(f"  Test F1: {f1:.4f}")


# ============================================================================
# EXAMPLE 3: LOADING REAL DATA (If Available)
# ============================================================================

def example_real_data(data_path):
    """Run stacking on real cleaned dataset."""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: STACKING ON REAL DATA")
    print("="*80)
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded: {df.shape}")
        
        # Prepare
        X = df.drop(columns=['ir_label', 'patient_id'], errors='ignore')
        y = df['ir_label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X.columns
        )
        
        y_train = pd.Series(y_train.values)
        y_test = pd.Series(y_test.values)
        
        # Stacking
        base_models = {
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=0),
            'LightGBM': LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1),
        }
        
        meta_learner = LogisticRegression(max_iter=1000)
        
        stacker = StackingEnsemble(base_models, meta_learner, n_splits=5)
        stacker.fit(X_train, y_train, verbose=True)
        
        y_proba = stacker.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"\nStacking AUC (Real Data): {auc:.4f}")
        
    except FileNotFoundError:
        print(f"  ⚠ Data file not found: {data_path}")
        print("  Skipping real data example.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("ADVANCED ENSEMBLE FRAMEWORK - QUICK START")
    print("="*80)
    
    # Run examples
    example_stacking_synthetic()
    example_threshold_optimization()
    example_real_data('data/cleaned_all_datasets_merged.csv')
    
    print("\n" + "="*80)
    print("Next: Run main_ensemble_pipeline.py for full pipeline")
    print("="*80 + "\n")

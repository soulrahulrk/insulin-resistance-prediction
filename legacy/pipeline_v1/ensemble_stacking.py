"""
Advanced Ensemble Framework: Stacking, Blending, Meta-Learning & Calibration
For Insulin Resistance Prediction

Modules:
  1. Cross-Model Stacking (Level-0 & Level-1)
  2. Blending with Hold-Out Validation Set
  3. Meta-Learning (Logistic Regression, Ridge, SVM)
  4. Model Calibration (Isotonic, Platt Scaling)
  5. Threshold Optimization (Medical ROI-Based)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, f1_score,
    confusion_matrix, classification_report, brier_score_loss
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. STACKING FRAMEWORK (Level-0 + Level-1 Meta-Learner)
# ============================================================================

class StackingEnsemble:
    """
    Cross-validation based stacking with multiple Level-0 learners.
    
    Workflow:
      1. Train Level-0 models on k-fold splits
      2. Generate out-of-fold predictions (train meta-features)
      3. Train Level-0 models on full data (for test predictions)
      4. Train Level-1 meta-learner on meta-features
      5. Stack: meta-learner predicts from Level-0 test predictions
    """
    
    def __init__(self, level0_models, level1_model, n_splits=5, random_state=42):
        """
        Args:
            level0_models: dict of {name: model_instance}
            level1_model: meta-learner (LogReg, Ridge, SVM, etc.)
            n_splits: k-fold splits (5 recommended for ~57k samples)
        """
        self.level0_models = level0_models
        self.level1_model = level1_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        self.meta_train = None
        self.level0_fitted = {}
        self.meta_learner_fitted = None
        self.feature_names = None
        
    def fit(self, X_train, y_train, verbose=True):
        """
        Generate meta-features via k-fold cross-validation.
        
        Args:
            X_train: shape (n_train, n_features)
            y_train: shape (n_train,)
        """
        n_models = len(self.level0_models)
        meta_train = np.zeros((X_train.shape[0], n_models))
        
        if verbose:
            print(f"[STACKING] Generating meta-features via {self.n_splits}-fold CV...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            if verbose:
                print(f"  Fold {fold_idx + 1}/{self.n_splits}...", end=" ")
            
            for model_idx, (model_name, model) in enumerate(self.level0_models.items()):
                model_clone = clone_model(model)
                model_clone.fit(X_tr, y_tr)
                
                # Out-of-fold predictions (for meta-features)
                meta_train[val_idx, model_idx] = model_clone.predict_proba(X_val)[:, 1]
            
            if verbose:
                print("✓")
        
        self.meta_train = meta_train
        
        # Fit Level-0 models on full training data
        if verbose:
            print(f"[STACKING] Fitting Level-0 models on full data...")
        
        for model_name, model in self.level0_models.items():
            model_clone = clone_model(model)
            model_clone.fit(X_train, y_train)
            self.level0_fitted[model_name] = model_clone
            if verbose:
                print(f"  {model_name}: ✓")
        
        # Fit Level-1 meta-learner
        if verbose:
            print(f"[STACKING] Training Level-1 meta-learner...")
        
        self.meta_learner_fitted = clone_model(self.level1_model)
        self.meta_learner_fitted.fit(self.meta_train, y_train)
        
        if verbose:
            print(f"  Meta-learner AUC (training): {roc_auc_score(y_train, self.meta_learner_fitted.predict_proba(self.meta_train)[:, 1]):.4f}")
        
        return self
    
    def predict_proba(self, X_test):
        """
        Generate Level-0 predictions on test set, then meta-predict.
        
        Returns:
            array of shape (n_test, 2) with probabilities
        """
        meta_test = np.zeros((X_test.shape[0], len(self.level0_models)))
        
        for model_idx, (model_name, model) in enumerate(self.level0_fitted.items()):
            meta_test[:, model_idx] = model.predict_proba(X_test)[:, 1]
        
        # Meta-predict
        return self.meta_learner_fitted.predict_proba(meta_test)
    
    def predict(self, X_test, threshold=0.5):
        """Binary predictions with custom threshold."""
        proba = self.predict_proba(X_test)[:, 1]
        return (proba >= threshold).astype(int)


# ============================================================================
# 2. BLENDING FRAMEWORK (Hold-Out Validation Set)
# ============================================================================

class BlendingEnsemble:
    """
    Simpler alternative to stacking using a hold-out validation set.
    
    Workflow:
      1. Split train into train (60%) and validation (40%)
      2. Fit Level-0 models on train
      3. Generate predictions on validation → meta-features
      4. Fit Level-1 on meta-features
      5. Generate Level-0 predictions on test
      6. Blend with Level-1 meta-learner
    
    Pros: Faster, simpler, less leakage risk
    Cons: Smaller train set for Level-0, unused data in validation
    """
    
    def __init__(self, level0_models, level1_model, val_size=0.4, random_state=42):
        self.level0_models = level0_models
        self.level1_model = level1_model
        self.val_size = val_size
        self.random_state = random_state
        
        self.level0_fitted = {}
        self.meta_learner_fitted = None
        
    def fit(self, X_train, y_train, verbose=True):
        """Train Level-0 on train subset, meta-learner on validation subset."""
        
        # Split into train and validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, 
            test_size=self.val_size, 
            random_state=self.random_state,
            stratify=y_train
        )
        
        if verbose:
            print(f"[BLENDING] Train: {X_tr.shape[0]}, Validation: {X_val.shape[0]}")
        
        # Fit Level-0 on train subset
        if verbose:
            print(f"[BLENDING] Fitting Level-0 models...")
        
        for model_name, model in self.level0_models.items():
            model_clone = clone_model(model)
            model_clone.fit(X_tr, y_tr)
            self.level0_fitted[model_name] = model_clone
            if verbose:
                print(f"  {model_name}: ✓")
        
        # Generate meta-features on validation
        meta_val = np.zeros((X_val.shape[0], len(self.level0_models)))
        for model_idx, (model_name, model) in enumerate(self.level0_fitted.items()):
            meta_val[:, model_idx] = model.predict_proba(X_val)[:, 1]
        
        # Fit meta-learner on validation meta-features
        if verbose:
            print(f"[BLENDING] Training Level-1 meta-learner...")
        
        self.meta_learner_fitted = clone_model(self.level1_model)
        self.meta_learner_fitted.fit(meta_val, y_val)
        
        if verbose:
            print(f"  Meta-learner AUC (validation): {roc_auc_score(y_val, self.meta_learner_fitted.predict_proba(meta_val)[:, 1]):.4f}")
        
        return self
    
    def predict_proba(self, X_test):
        """Generate Level-0 predictions, then blend."""
        meta_test = np.zeros((X_test.shape[0], len(self.level0_models)))
        
        for model_idx, (model_name, model) in enumerate(self.level0_fitted.items()):
            meta_test[:, model_idx] = model.predict_proba(X_test)[:, 1]
        
        return self.meta_learner_fitted.predict_proba(meta_test)
    
    def predict(self, X_test, threshold=0.5):
        proba = self.predict_proba(X_test)[:, 1]
        return (proba >= threshold).astype(int)


# ============================================================================
# 3. MODEL CALIBRATION
# ============================================================================

class CalibrationWrapper:
    """
    Calibrate any classifier using isotonic regression or Platt scaling.
    
    Medical Use Case:
      - Isotonic: More flexible, better for medical decisions (preferred)
      - Platt: Parametric, good for probabilistic outputs
    """
    
    def __init__(self, model, method='isotonic'):
        """
        Args:
            model: Fitted classifier with predict_proba()
            method: 'isotonic' or 'platt'
        """
        self.model = model
        self.method = method
        self.calibrator = CalibratedClassifierCV(
            base_estimator=model, 
            method=method, 
            cv=5
        )
    
    def fit(self, X_calib, y_calib):
        """Fit calibrator on calibration data (separate hold-out set)."""
        self.calibrator.fit(X_calib, y_calib)
        return self
    
    def predict_proba(self, X_test):
        """Return calibrated probabilities."""
        return self.calibrator.predict_proba(X_test)
    
    def predict(self, X_test, threshold=0.5):
        proba = self.predict_proba(X_test)[:, 1]
        return (proba >= threshold).astype(int)


# ============================================================================
# 4. THRESHOLD OPTIMIZATION FOR MEDICAL USE CASES
# ============================================================================

class ThresholdOptimizer:
    """
    Find optimal decision threshold based on medical priorities.
    
    Common Objectives:
      - Maximize F1 (balanced precision-recall)
      - Maximize Sensitivity (recall) at fixed specificity
      - Minimize False Positives (high specificity)
      - Maximize Youden Index (Sens + Spec - 1)
    """
    
    def __init__(self, y_true, y_proba, strategy='f1'):
        """
        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities
            strategy: 'f1', 'youden', 'sensitivity@specificity', 'specificity@sensitivity'
        """
        self.y_true = y_true
        self.y_proba = y_proba
        self.strategy = strategy
        self.optimal_threshold = None
        self.optimal_score = None
        
    def find_optimal(self, specificity_target=None, sensitivity_target=None):
        """
        Find threshold optimizing the chosen strategy.
        
        Args:
            specificity_target: For 'sensitivity@specificity' strategy
            sensitivity_target: For 'specificity@sensitivity' strategy
        """
        
        thresholds = np.linspace(0, 1, 101)
        scores = []
        
        if self.strategy == 'f1':
            for thresh in thresholds:
                y_pred = (self.y_proba >= thresh).astype(int)
                f1 = f1_score(self.y_true, y_pred, zero_division=0)
                scores.append(f1)
            self.optimal_threshold = thresholds[np.argmax(scores)]
            self.optimal_score = max(scores)
            
        elif self.strategy == 'youden':
            precisions, recalls, _ = precision_recall_curve(self.y_true, self.y_proba)
            fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
            youden = tpr + (1 - fpr) - 1  # Sensitivity + Specificity - 1
            idx = np.argmax(youden)
            self.optimal_threshold = _[idx] if _ is not None else thresholds[idx]
            self.optimal_score = youden[idx]
            
        elif self.strategy == 'sensitivity@specificity':
            for thresh in thresholds:
                y_pred = (self.y_proba >= thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                if abs(spec - specificity_target) < 0.01:  # Tolerance
                    scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0)  # Sensitivity
            if scores:
                idx = np.argmax(scores)
                self.optimal_threshold = thresholds[idx]
                self.optimal_score = max(scores)
        
        return self.optimal_threshold, self.optimal_score
    
    def plot_roc_with_threshold(self, figsize=(10, 6)):
        """Plot ROC curve with optimal threshold marked."""
        fpr, tpr, roc_thresholds = roc_curve(self.y_true, self.y_proba)
        auc = roc_auc_score(self.y_true, self.y_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})', linewidth=2)
        
        if self.optimal_threshold is not None:
            idx = np.argmin(np.abs(roc_thresholds - self.optimal_threshold))
            plt.scatter(fpr[idx], tpr[idx], color='red', s=100, label=f'Optimal Threshold={self.optimal_threshold:.3f}', zorder=5)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (Strategy: {self.strategy})')
        plt.legend()
        plt.grid(alpha=0.3)
        return plt.gcf()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clone_model(model):
    """Deep copy of a scikit-learn model."""
    import copy
    return copy.deepcopy(model)


def evaluate_ensemble(y_true, y_proba, y_pred=None, threshold=0.5, model_name="Ensemble"):
    """Comprehensive evaluation metrics."""
    
    if y_pred is None:
        y_pred = (y_proba >= threshold).astype(int)
    
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Threshold: {threshold:.3f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Brier Score (calibration): {brier:.4f}")
    print(f"  Sensitivity (Recall/TPR): {sensitivity:.4f}")
    print(f"  Specificity (TNR): {specificity:.4f}")
    print(f"  Precision (PPV): {precision:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    print(f"{'='*60}\n")
    
    return {
        'auc': auc,
        'f1': f1,
        'brier': brier,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'threshold': threshold
    }


# ============================================================================
# EXAMPLE: FULL ENSEMBLE PIPELINE
# ============================================================================

if __name__ == '__main__':
    print("[INFO] Advanced Ensemble Framework loaded successfully.")
    print("\nAvailable Classes:")
    print("  • StackingEnsemble      : 5-fold cross-validation stacking")
    print("  • BlendingEnsemble      : Hold-out blending (faster)")
    print("  • CalibrationWrapper    : Isotonic/Platt scaling")
    print("  • ThresholdOptimizer    : Medical ROI-based threshold search")
    print("\nUsage in main_ensemble_pipeline.py")

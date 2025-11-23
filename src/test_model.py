"""Evaluate trained ensemble on the deterministic test split."""

import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    ENSEMBLE_MODEL_PATH,
    FEATURE_TRANSFORMER_PATH,
    LOG_FORMAT,
    LOG_LEVEL,
    MODELS_DIR,
    OPTIMAL_THRESHOLD_PATH,
    RANDOM_STATE,
    SELECTED_FEATURES_PATH,
    TEST_SIZE,
)
from src.data_loader import load_data
from src.features import compute_engineered_features

REPORTS_DIR = MODELS_DIR.parent / "reports"
LOG_FILE = MODELS_DIR / "test.log"

def _setup_logger() -> logging.Logger:
    """Configure logger dedicated to test evaluation."""
    logger = logging.getLogger("test_evaluation")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.handlers = []

    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _prepare_test_data(preprocessor):
    """Recreate the deterministic test split used during training."""
    df = load_data()
    df = compute_engineered_features(df)

    if "ir_label" not in df.columns:
        raise ValueError("Dataset must contain 'ir_label' column for evaluation.")

    X = df.drop(columns=["ir_label"])
    y = df["ir_label"].reset_index(drop=True)

    X_processed, _ = preprocessor.transform(X)
    X_processed = X_processed.reset_index(drop=True)

    _, X_test, _, y_test = train_test_split(
        X_processed,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return X_test, y_test


def _build_meta_features(base_models, model_order, X_features):
    """Generate meta features by running each base model on the feature set."""
    if base_models is None:
        raise ValueError("Base models are missing from the saved ensemble. Retrain the pipeline to regenerate artifacts.")

    ordered_names = model_order or list(base_models.keys())
    meta_columns = []
    for name in ordered_names:
        if name not in base_models:
            raise KeyError(f"Base model '{name}' not found in saved artifacts.")
        model = base_models[name]
        meta_columns.append(model.predict_proba(X_features)[:, 1])

    return np.column_stack(meta_columns)


def evaluate_on_test_set():
    """Load saved artifacts and evaluate ensemble on the hold-out test set."""
    logger = _setup_logger()
    logger.info("=" * 39)
    logger.info("STARTING MODEL EVALUATION ON TEST SET")
    logger.info("=" * 39)

    try:
        # --- 1. Load Artifacts ---
        logger.info("[1/5] Loading saved model and artifacts...")
        model_package = joblib.load(ENSEMBLE_MODEL_PATH)
        if not isinstance(model_package, dict) or "calibrated_model" not in model_package:
            raise ValueError("Saved ensemble is missing required components. Please rerun training to regenerate artifacts.")

        base_models = model_package.get("base_models")
        model_order = model_package.get("model_order") or list(base_models.keys()) if base_models else []
        calibrated_model = model_package["calibrated_model"]
        preprocessor = joblib.load(FEATURE_TRANSFORMER_PATH)

        with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
            selected_features = json.load(f)
        with open(OPTIMAL_THRESHOLD_PATH, "r", encoding="utf-8") as f:
            optimal_threshold = float(f.read())

        logger.info("✓ Loaded calibrated stacker, base models, and feature transformer")
        logger.info(f"✓ Using {len(selected_features)} selected features")
        logger.info(f"✓ Applying optimal decision threshold: {optimal_threshold:.4f}")

        # --- 2. Load and Split Data ---
        logger.info("\n[2/5] Preparing deterministic test split...")
        X_test, y_test = _prepare_test_data(preprocessor)
        logger.info(f"✓ Test set ready: {X_test.shape[0]} samples × {X_test.shape[1]} features")

        # --- 3. Make Predictions ---
        logger.info("\n[3/5] Making predictions on the test set...")
        meta_X_test = _build_meta_features(base_models, model_order, X_test)
        y_pred_proba = calibrated_model.predict_proba(meta_X_test)[:, 1]
        y_pred_class = (y_pred_proba >= optimal_threshold).astype(int)
        logger.info("✓ Predictions generated")

        # --- 4. Evaluate Performance ---
        logger.info("\n[4/5] Evaluating performance metrics...")
        report = classification_report(y_test, y_pred_class, target_names=["Not Resistant", "Resistant"])
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)

        print("\n--- Test Set Performance ---")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Brier Score: {brier:.4f}")
        print("\nClassification Report:")
        print(report)

        logger.info(f"Test Set ROC AUC: {roc_auc:.4f}")
        logger.info(f"Test Set Brier Score: {brier:.4f}")
        logger.info("\nClassification Report:\n%s", report)

        # --- 5. Generate Confusion Matrix ---
        logger.info("\n[5/5] Generating confusion matrix plot...")
        cm = confusion_matrix(y_test, y_pred_class)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Resistant", "Resistant"],
            yticklabels=["Not Resistant", "Resistant"],
        )
        plt.title("Confusion Matrix - Test Set")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = REPORTS_DIR / "test_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        logger.info(f"✓ Confusion matrix saved to {plot_path}")
        print(f"\nConfusion matrix plot saved to {plot_path}")

        logger.info("\n" + "=" * 39)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 39)

    except Exception as exc:
        logger.error("✗ Evaluation failed: %s", exc, exc_info=True)
        print(f"An error occurred during evaluation. Check {LOG_FILE} for details.")

if __name__ == "__main__":
    evaluate_on_test_set()

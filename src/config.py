"""Configuration for Insulin Resistance Prediction System."""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent

def _resolve_data_path() -> Path:
    """Resolve canonical dataset path with fallbacks."""
    env_path = os.environ.get("IR_DATA_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        Path("/mnt/data/all_datasets_merged.csv"),
        PROJECT_ROOT / "data" / "all_datasets_merged.csv",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Last resort: default Linux-style path
    return Path("/mnt/data/all_datasets_merged.csv")


DATA_PATH = _resolve_data_path()
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR
LOGS_DIR = PROJECT_ROOT / "logs"
METRICS_DIR = PROJECT_ROOT / "metrics"
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "data"

# Ensure critical directories exist
for directory in (MODELS_DIR, LOGS_DIR, METRICS_DIR, TEST_DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# RANDOM STATE & REPRODUCIBILITY
# ============================================================================
RANDOM_STATE = 42

# ============================================================================
# DATA SPLIT RATIOS
# ============================================================================
TEST_SIZE = 0.15  # 15% test
VAL_SIZE = 0.15   # 15% validation from remaining
TRAIN_SIZE = 0.70  # 70% training
N_FOLDS = 2  # 2-fold cross-validation for stacking (Reduced for local run)

# ============================================================================
# PREPROCESSING
# ============================================================================
KNN_NEIGHBORS = 5
FEATURE_SELECTION_THRESHOLD = 0.001
MAX_CATEGORICAL_CARDINALITY = 50

# ============================================================================
# BASE LEARNER HYPERPARAMETERS
# ============================================================================
XGBOOST_PARAMS = {
    "n_estimators": 50,  # Reduced for local run
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "random_state": RANDOM_STATE,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "n_jobs": -1,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 50,  # Reduced for local run
    "num_leaves": 31,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

CATBOOST_PARAMS = {
    "iterations": 50,  # Reduced for local run
    "depth": 6,
    "learning_rate": 0.03,
    "loss_function": "Logloss",
    "verbose": False,
    "random_state": RANDOM_STATE,
}

GRADBOOST_PARAMS = {
    "n_estimators": 50,  # Reduced for local run
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}

# ============================================================================
# META-LEARNER HYPERPARAMETERS
# ============================================================================
META_LEARNER_C_GRID = [0.01, 0.1, 1.0, 10.0]

# ============================================================================
# MODEL ARTIFACTS PATHS
# ============================================================================
ENSEMBLE_MODEL_PATH = ARTIFACTS_DIR / "ir_ensemble_best.pkl"
FEATURE_TRANSFORMER_PATH = ARTIFACTS_DIR / "feature_transformer.pkl"
SELECTED_FEATURES_PATH = ARTIFACTS_DIR / "selected_features.json"
OPTIMAL_THRESHOLD_PATH = ARTIFACTS_DIR / "optimal_threshold.txt"
PERFORMANCE_METRICS_PATH = ARTIFACTS_DIR / "performance_metrics.json"
BASE_MODELS_METRICS_PATH = ARTIFACTS_DIR / "base_models_metrics.csv"
TRAIN_LOG_PATH = LOGS_DIR / "train.log"
PREDICTION_LOG_PATH = LOGS_DIR / "predictions.jsonl"
DRIFT_ALERT_LOG_PATH = LOGS_DIR / "drift_alerts.jsonl"
PROM_METRICS_PATH = METRICS_DIR / "current_metrics.prom"

# ============================================================================
# HYPERPARAMETER TUNING (OPT-IN)
# ============================================================================
ENABLE_OPTUNA = False  # Set to True to enable Optuna-based hyperparameter tuning
OPTUNA_N_TRIALS = 50

# ============================================================================
# API SETTINGS
# ============================================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

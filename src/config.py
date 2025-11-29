"""Central configuration constants for the Insulin Resistance Prediction System.

This module defines deterministic constants and artifact paths used across the
training, evaluation, and serving stack. All random seeds are fixed for
reproducibility (RANDOM_STATE = 42).
"""

from pathlib import Path
import os

# ----------------------------------------------------------------------------
# PROJECT & DATA
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent

# Fixed dataset path per specification (override via env if needed downstream)
_DEFAULT_DATA_PATH = "/mnt/data/all_datasets_merged.csv"
DATA_PATH = os.environ.get("IR_DATA_PATH", _DEFAULT_DATA_PATH)
# Fallback to project root CSV if specified path missing
if not Path(DATA_PATH).exists():
    local_candidate = PROJECT_ROOT / "data" / "all_datasets_merged.csv"
    if local_candidate.exists():
        DATA_PATH = str(local_candidate)

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "ensemble"
LOGS_DIR = PROJECT_ROOT / "logs"  # Separate logs from artifacts

for d in (MODELS_DIR, RESULTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# REPRODUCIBILITY
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_FOLDS = 5

# ----------------------------------------------------------------------------
# ARTIFACT PATHS (single source of truth)
# ----------------------------------------------------------------------------
ARTIFACT_PATHS = {
    "stacker": "models/ir_ensemble_best.pkl",
    "transformer": "models/feature_transformer.pkl",
    "selected_features": "models/selected_features.json",
    "optimal_threshold": "models/optimal_threshold.txt",
    "metrics": "models/performance_metrics.json",
    "base_metrics_csv": "models/base_models_metrics.csv",
}

ENSEMBLE_MODEL_PATH = PROJECT_ROOT / ARTIFACT_PATHS["stacker"]
FEATURE_TRANSFORMER_PATH = PROJECT_ROOT / ARTIFACT_PATHS["transformer"]
SELECTED_FEATURES_PATH = PROJECT_ROOT / ARTIFACT_PATHS["selected_features"]
OPTIMAL_THRESHOLD_PATH = PROJECT_ROOT / ARTIFACT_PATHS["optimal_threshold"]
PERFORMANCE_METRICS_PATH = PROJECT_ROOT / ARTIFACT_PATHS["metrics"]
BASE_MODELS_METRICS_PATH = PROJECT_ROOT / ARTIFACT_PATHS["base_metrics_csv"]
TRAIN_LOG_PATH = LOGS_DIR / "train.log"

# ----------------------------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------------------------
KNN_NEIGHBORS = 5  # Used for fasting_insulin imputation only
FEATURE_SELECTION_THRESHOLD = 0.001
MAX_CATEGORICAL_CARDINALITY = 50

# ----------------------------------------------------------------------------
# BASE MODEL HYPERPARAMETERS (reduced capacity to prevent overfitting)
# ----------------------------------------------------------------------------
BASE_PARAMS = {
    "xgb": {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "random_state": RANDOM_STATE,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "n_jobs": -1,
    },
    "lgb": {
        "n_estimators": 200,
        "num_leaves": 20,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.7,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    },
    "cat": {
        "iterations": 200,
        "depth": 4,
        "learning_rate": 0.03,
        "loss_function": "Logloss",
        "verbose": False,
        "random_state": RANDOM_STATE,
    },
    "gb": {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.8,
        "random_state": RANDOM_STATE,
    },
}

# Backwards-compatible aliases (existing modules may still import)
XGBOOST_PARAMS = BASE_PARAMS["xgb"]
LIGHTGBM_PARAMS = BASE_PARAMS["lgb"]
CATBOOST_PARAMS = BASE_PARAMS["cat"]
GRADBOOST_PARAMS = BASE_PARAMS["gb"]

# ----------------------------------------------------------------------------
# META LEARNER
# ----------------------------------------------------------------------------
META_LEARNER_C_GRID = [0.01, 0.1, 1.0, 10.0]

# ----------------------------------------------------------------------------
# TUNING FLAGS
# ----------------------------------------------------------------------------
OPTUNA_ENABLED = False
OPTUNA_N_TRIALS = 50

# Backward compatibility for legacy imports expecting ENABLE_OPTUNA
ENABLE_OPTUNA = OPTUNA_ENABLED

# ----------------------------------------------------------------------------
# API SETTINGS
# ----------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1

# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


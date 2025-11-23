"""External cohort validation utilities for the IR prediction system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from src.config import (
    ENSEMBLE_MODEL_PATH,
    FEATURE_TRANSFORMER_PATH,
    PERFORMANCE_METRICS_PATH,
    ARTIFACTS_DIR,
)
from src.features import compute_engineered_features
from src.monitoring import export_prometheus
from src.utils import configure_logger, load_json

LOGGER = configure_logger("external_validation")
EXTERNAL_REPORTS_DIR = Path("reports/external_validation")
EXTERNAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_artifacts(model_artifact_path: Path, feature_transformer_path: Path) -> Tuple[dict, object]:
    model_package = joblib.load(model_artifact_path)
    preprocessor = joblib.load(feature_transformer_path)
    return model_package, preprocessor


def _build_meta_features(base_models: Dict, model_order: list, features: pd.DataFrame) -> np.ndarray:
    ordered = model_order or list(base_models.keys())
    meta_cols = []
    for name in ordered:
        model = base_models[name]
        meta_cols.append(model.predict_proba(features)[:, 1])
    return np.column_stack(meta_cols)


def validate_on_external_dataset(
    csv_path: str | Path,
    model_artifact_path: str | Path = ENSEMBLE_MODEL_PATH,
    feature_transformer_path: str | Path = FEATURE_TRANSFORMER_PATH,
) -> Dict[str, float]:
    """Evaluate the calibrated ensemble on an external dataset."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"External dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = compute_engineered_features(df)
    if "ir_label" not in df.columns:
        raise ValueError("External dataset must include 'ir_label'.")

    y_true = df["ir_label"].astype(int)
    X = df.drop(columns=["ir_label"])

    model_package, preprocessor = _load_artifacts(Path(model_artifact_path), Path(feature_transformer_path))
    features_processed, feature_names = preprocessor.transform(X)
    features_processed = features_processed.reset_index(drop=True)

    base_models = model_package.get("base_models")
    if not base_models:
        raise ValueError("Base models missing from ensemble artifact. Retrain pipeline to regenerate them.")
    meta_features = _build_meta_features(base_models, model_package.get("model_order", []), features_processed)

    calibrated_model = model_package.get("calibrated_model")
    y_prob = calibrated_model.predict_proba(meta_features)[:, 1]
    y_pred = (y_prob >= float(load_json(PERFORMANCE_METRICS_PATH).get("optimal_threshold", 0.5)))

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float((y_pred == y_true).mean()),
    }

    baseline = load_json(PERFORMANCE_METRICS_PATH)
    baseline_auc = baseline.get("stacking", {}).get("val_auc") or baseline.get("calibration", {}).get("brier_score")
    if baseline_auc and (baseline_auc - metrics["roc_auc"]) > 0.10:
        raise RuntimeError(
            f"External ROC AUC dropped by more than 10% (baseline={baseline_auc:.3f}, external={metrics['roc_auc']:.3f})."
        )

    report_prefix = EXTERNAL_REPORTS_DIR / csv_path.stem
    metrics_path = Path(f"{report_prefix}_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"External ROC (AUC={metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("External Validation ROC Curve")
    plt.legend(loc="lower right")
    roc_path = Path(f"{report_prefix}_roc.png")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    LOGGER.info("External validation complete: metrics=%s", metrics)
    export_prometheus({"requests_total": 0, "avg_prob": metrics["roc_auc"], "avg_latency_ms": metrics["brier_score"]})
    return metrics

"""Integration tests for the FastAPI deployment module."""

from __future__ import annotations

import io
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import deploy_api


class DummyPreprocessor:
    """Minimal preprocessor stub used for API tests."""

    numeric_cols = ["age", "fasting_glucose", "fasting_insulin", "bmi"]
    categorical_cols: list[str] = []

    def transform(self, df: pd.DataFrame):
        frame = df.reindex(columns=self.numeric_cols)
        frame = frame.fillna(0).astype(float)
        if (frame["age"] < 0).any():
            raise ValueError("Age must be non-negative")
        return frame.reset_index(drop=True), self.numeric_cols


class DummyBaseModel:
    """Simple base model that derives probabilities from feature sums."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        sums = features.sum(axis=1).to_numpy(dtype=float)
        probs = 1 / (1 + np.exp(-sums / 100.0))
        return np.column_stack([1 - probs, probs])


class DummyCalibratedModel:
    """Calibrated model wrapper mimicking the stacking head."""

    def __init__(self):
        self.base_estimator = type("Estimator", (), {"coef_": np.array([[1.0]])})()

    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        logits = meta_features.sum(axis=1)
        probs = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])


@pytest.fixture()
def api_client(monkeypatch):
    """Provision a TestClient with stubbed artifacts and monitoring hooks."""

    captured_records: list[Dict] = []

    def fake_record(entry: Dict, log_path=None) -> None:  # pragma: no cover - side effect only
        captured_records.append(entry)

    monkeypatch.setattr(deploy_api, "record_prediction", fake_record)
    monkeypatch.setattr(deploy_api, "export_prometheus", lambda metrics: None)

    def fake_load_artifacts() -> None:
        deploy_api.MODEL = {
            "base_models": {"xgboost": DummyBaseModel()},
            "model_order": ["xgboost"],
            "calibrated_model": DummyCalibratedModel(),
        }
        deploy_api.PREPROCESSOR = DummyPreprocessor()
        deploy_api.SELECTED_FEATURES = DummyPreprocessor.numeric_cols
        deploy_api.OPTIMAL_THRESHOLD = 0.4
        deploy_api.MODEL_VERSION = "test-model"
        deploy_api.PERFORMANCE_BASELINE = {"stacking": {"val_auc": 0.91}}
        deploy_api.PREDICTION_BUFFER.clear()

    monkeypatch.setattr(deploy_api, "load_artifacts", fake_load_artifacts)

    with TestClient(deploy_api.app) as client:
        yield {"client": client, "records": captured_records}


def test_health_endpoint_returns_ok(api_client):
    client = api_client["client"]
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_logs_and_reports_metrics(api_client):
    client = api_client["client"]
    payload = {
        "age": 44,
        "fasting_glucose": 110.0,
        "fasting_insulin": 15.0,
        "bmi": 27.5,
        "sex": "F",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_level"] in {"low", "medium", "high"}
    assert data["shap_top3"] is None

    assert len(api_client["records"]) == 1
    metrics_response = client.get("/metrics")
    metrics = metrics_response.json()
    assert metrics["requests_total"] == 1
    assert metrics["model_version"] == "test-model"


def test_batch_prediction_handles_mixed_rows(api_client):
    client = api_client["client"]
    csv_payload = """age,fasting_glucose,fasting_insulin,bmi\n45,110,15,27\n-5,105,20,31"""
    files = {"file": ("patients.csv", io.BytesIO(csv_payload.encode("utf-8")), "text/csv")}
    response = client.post("/batch_predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert data["successful"] == 1
    assert data["failed"] == 1
    assert len(data["predictions"]) == 1
    assert len(api_client["records"]) == 1
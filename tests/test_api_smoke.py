"""Live artifact smoke tests for the FastAPI service."""

from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import deploy_api
from src.config import DATA_PATH, ENSEMBLE_MODEL_PATH, FEATURE_TRANSFORMER_PATH

pytestmark = pytest.mark.skipif(
    not (ENSEMBLE_MODEL_PATH.exists() and FEATURE_TRANSFORMER_PATH.exists() and DATA_PATH.exists()),
    reason="Required artifacts or dataset missing",
)


def _load_payload() -> dict:
    df = pd.read_csv(DATA_PATH)
    required = ["age", "fasting_glucose", "fasting_insulin", "bmi"]
    if any(col not in df.columns for col in required):
        pytest.skip("Dataset missing required columns for smoke payload")
    numeric = df[required].fillna(df[required].median())
    row = numeric.iloc[0]
    payload = {col: float(row[col]) for col in required}
    if "sex" in df.columns:
        non_null = df["sex"].dropna()
        if not non_null.empty:
            payload["sex"] = str(non_null.iloc[0])
    return payload


def test_fastapi_endpoints_accept_real_artifacts():
    payload = _load_payload()
    with TestClient(deploy_api.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.post("/predict", params={"explain": "true"}, json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["shap_top3"]
        assert body["risk_level"] in {"low", "medium", "high"}

        csv_buffer = io.StringIO()
        pd.DataFrame([payload, payload]).to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        batch = client.post(
            "/batch_predict",
            files={"file": ("patients.csv", csv_buffer.getvalue(), "text/csv")},
        )
        assert batch.status_code == 200
        batch_body = batch.json()
        assert batch_body["total_records"] == 2
        assert batch_body["successful"] >= 1
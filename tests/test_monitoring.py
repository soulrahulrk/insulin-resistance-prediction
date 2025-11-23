"""Unit tests for monitoring helpers."""

from __future__ import annotations

import json
from src.monitoring import compute_aggregate_metrics, export_prometheus, record_prediction


def test_record_prediction_appends_json_line(tmp_path):
    log_path = tmp_path / "predictions.log"
    entry = {"probability": 0.7, "prediction": 1, "trace_id": "abc123"}
    record_prediction(entry, log_path)
    content = log_path.read_text(encoding="utf-8").strip()
    loaded = json.loads(content)
    assert loaded["probability"] == entry["probability"]
    assert loaded["prediction"] == 1
    assert loaded["trace_id"] == "abc123"
    assert "timestamp" in loaded


def test_compute_aggregate_metrics_handles_buffer():
    buffer = [
        {"probability": 0.2, "latency_ms": 50},
        {"probability": 0.8, "latency_ms": 150},
    ]
    metrics = compute_aggregate_metrics(buffer)
    assert metrics["requests_total"] == 2
    assert metrics["avg_prob"] == 0.5
    assert metrics["avg_latency_ms"] == 100


def test_export_prometheus_format(tmp_path):
    out_path = tmp_path / "metrics.prom"
    metrics = {"requests_total": 5, "avg_prob": 0.42, "avg_latency_ms": 12.5}
    export_prometheus(metrics, out_path)
    written = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert written[0] == "ir_model_requests_total 5"
    assert written[1].startswith("ir_model_avg_prob 0.420000")
    assert written[2].startswith("ir_model_avg_latency_ms 12.500000")
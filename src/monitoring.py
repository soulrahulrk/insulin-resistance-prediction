"""Lightweight monitoring helpers for the IR prediction API."""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable

from src.config import PREDICTION_LOG_PATH, PROM_METRICS_PATH
from src.utils import configure_logger

LOGGER = configure_logger("monitoring")


def record_prediction(entry: Dict, log_path: Path = PREDICTION_LOG_PATH) -> None:
    """Append a single prediction event to the JSONL log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    safe_entry = entry.copy()
    safe_entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_entry, ensure_ascii=False) + "\n")
    LOGGER.debug("Logged prediction trace=%s", safe_entry.get("trace_id", "n/a"))


def compute_aggregate_metrics(buffer: Iterable[Dict]) -> Dict[str, float]:
    """Compute request aggregates from the in-memory buffer."""
    entries = list(buffer)
    if not entries:
        return {"requests_total": 0, "avg_prob": 0.0, "avg_latency_ms": 0.0}

    requests_total = len(entries)
    avg_prob = sum(item.get("probability", 0.0) for item in entries) / requests_total
    avg_latency = sum(item.get("latency_ms", 0.0) for item in entries) / requests_total

    return {
        "requests_total": requests_total,
        "avg_prob": avg_prob,
        "avg_latency_ms": avg_latency,
    }


def export_prometheus(metrics: Dict[str, float], out_path: Path = PROM_METRICS_PATH) -> None:
    """Write metrics in Prometheus text format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"ir_model_requests_total {int(metrics.get('requests_total', 0))}",
        f"ir_model_avg_prob {metrics.get('avg_prob', 0.0):.6f}",
        f"ir_model_avg_latency_ms {metrics.get('avg_latency_ms', 0.0):.6f}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.debug("Exported Prometheus metrics to %s", out_path)

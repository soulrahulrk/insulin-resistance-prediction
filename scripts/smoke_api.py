"""Run a lightweight smoke test against a deployed API instance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

DEFAULT_PAYLOAD = {
    "age": 42,
    "fasting_glucose": 115.0,
    "fasting_insulin": 12.0,
    "bmi": 27.1,
    "sex": "F",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the Insulin Resistance API")
    parser.add_argument("--base-url", default=os.getenv("IR_API_URL", "http://127.0.0.1:8000"), help="API base URL")
    parser.add_argument("--timeout", type=float, default=5.0, help="Request timeout in seconds")
    parser.add_argument(
        "--payload",
        type=Path,
        help="Optional path to a JSON file containing the patient payload",
    )
    args = parser.parse_args()

    payload = DEFAULT_PAYLOAD
    if args.payload:
        payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))

    base = args.base_url.rstrip("/")
    try:
        health_resp = requests.get(f"{base}/health", timeout=args.timeout)
        health_resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[SMOKE] Health check failed: {exc}", file=sys.stderr)
        return 1

    try:
        predict_resp = requests.post(f"{base}/predict", json=payload, timeout=args.timeout)
        predict_resp.raise_for_status()
        data = predict_resp.json()
    except requests.RequestException as exc:
        print(f"[SMOKE] Prediction request failed: {exc}", file=sys.stderr)
        return 1

    if not (0.0 <= data.get("probability", -1) <= 1.0):
        print("[SMOKE] Probability outside [0,1] bounds", file=sys.stderr)
        return 1

    print("[SMOKE] API healthy. Probability={prob:.3f} risk={risk}".format(
        prob=data["probability"], risk=data["risk_level"],
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

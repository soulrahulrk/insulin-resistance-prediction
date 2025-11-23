# IR Prediction Runbook

Operational checklist for deploying, validating, and monitoring the Insulin Resistance Prediction API.

## 1. Pre-flight

- **Artifacts**: Ensure `models/feature_transformer.pkl`, `models/ir_ensemble_best.pkl`, `models/selected_features.json`, `models/optimal_threshold.txt`, and `models/performance_metrics.json` are present.
- **Environment**: `python -m venv .venv && .venv/Scripts/activate` (Windows) then `pip install -r config/requirements.txt`.
- **Smoke tests**: `python scripts/run_tests.py` locally before every push.

## 2. Deployment Steps

1. **Docker build**

   ```bash
   python scripts/docker_build_run.py --tag ir-api:latest
   ```

2. **Run container**

   ```bash
   python scripts/docker_build_run.py --tag ir-api:latest --run --port 8000 \
       --env IR_DATA_PATH=/mnt/data/all_datasets_merged.csv
   ```

3. **Post-deploy verification**

   ```bash
   IR_API_URL=http://127.0.0.1:8000 python scripts/smoke_api.py
   ```

## 3. Monitoring & Drift

- **Prediction logs**: `logs/predictions.jsonl` (one entry per request with trace ID, latency, SHAP excerpts).
- **Prometheus**: scrape `metrics/current_metrics.prom` or hit `/metrics` for JSON snapshot.
- **Drift alerts**: `logs/drift_alerts.jsonl` is appended when KS-test detects shifts vs. reference cohort.
- **Manual drift simulation**:

  ```bash
  python scripts/simulate_drift.py --reference data/all_datasets_merged.csv --noise-scale 0.35
  ```

## 4. External Validation

- Export the new cohort to CSV and run:

  ```bash
  EXTERNAL_CSV=/path/to/new_cohort.csv ./scripts/run_external_validation.sh --output reports/external_validation/new_metrics.json
  ```

- Pipeline aborts if ROC AUC drops >10% relative to `models/performance_metrics.json`.

## 5. Incident Response

- **API fails health**: restart container, inspect `logs/api.log` (if enabled) and `logs/predictions.jsonl`.
- **Metrics stale**: check filesystem permissions on `metrics/` mount; regenerate via `curl /metrics`.
- **Drift alert**: triage features listed in `logs/drift_alerts.jsonl`, rerun training if sustained.
- **Model regression**: rerun `python -m src.train` followed by `python -m src.test_model` and redeploy artifacts.

## 6. CI Expectations

- GitHub Actions workflow (`.github/workflows/ci.yml`) runs `scripts/run_tests.sh -q` on Python 3.11 for every push/PR.
- Merges require green CI plus a successful smoke test on the target environment.

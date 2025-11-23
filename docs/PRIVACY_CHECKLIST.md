# Privacy & PHI Checklist

## Data Handling

- ✅ **Source control**: `data/` remains local; never commit PHI datasets to git.
- ✅ **Environment variables**: configure `IR_DATA_PATH` or secure mounts for production cohorts.
- ✅ **Access logging**: retain API request logs (`logs/predictions.jsonl`) for 30 days, then rotate.

## Minimization & Anonymization

- Remove direct identifiers (name, MRN, email, phone) before CSV export.
- Hash quasi-identifiers if they must remain (e.g., `patient_id` → SHA-256 salt).
- Round or bucket continuous fields (age bins, BMI bins) when sharing outside care team.

## Retention & Deletion

- Store external validation outputs under `reports/external_validation/` with access controls.
- Follow a 90-day retention window for raw exported cohorts; archive or delete afterwards.
- Document deletion events in the deployment log (append-only text file recommended).

## Compliance Checks Before Deployment

- Confirm Business Associate Agreements (BAAs) for any cloud storage used.
- Ensure TLS termination in front of FastAPI when exposed beyond localhost.
- Run `python scripts/run_tests.py` and `IR_API_URL=... python scripts/smoke_api.py` prior to handling PHI.
- Verify Prometheus scrape endpoints exclude raw PHI (only aggregated metrics allowed).

## Incident Response

- If PHI leakage suspected, disable API, rotate credentials, and notify compliance officer within 24 hours.
- Use `logs/predictions.jsonl` trace IDs to audit affected requests.

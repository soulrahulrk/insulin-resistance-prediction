#!/usr/bin/env bash
set -euo pipefail
uvicorn src.deploy_api:app --host 0.0.0.0 --port 8000 --workers 1

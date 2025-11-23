#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${EXTERNAL_CSV:-}" ]]; then
  echo "EXTERNAL_CSV environment variable is required" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

python scripts/run_external_validation.py "$EXTERNAL_CSV" "$@"

#!/usr/bin/env bash
set -euo pipefail

# Ensure we're running from repo root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

python scripts/run_tests.py "$@"

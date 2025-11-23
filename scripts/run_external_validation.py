"""CLI helper to run external cohort validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # Ensure `src` package is importable when invoked from anywhere
    sys.path.insert(0, str(ROOT))

from src.external_validation import validate_on_external_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the model on an external dataset")
    parser.add_argument("csv_path", type=Path, help="Path to the external cohort CSV file")
    parser.add_argument(
        "--output", type=Path, help="Optional path to save metrics JSON (defaults to reports/external_validation)"
    )
    args = parser.parse_args()

    metrics = validate_on_external_dataset(args.csv_path)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

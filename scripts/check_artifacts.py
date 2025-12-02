#!/usr/bin/env python3
"""
Artifact and data file validation script.

Checks that all required model artifacts and the dataset exist.
Exits with code 0 if all present, code 1 if any are missing.

Usage:
    python scripts/check_artifacts.py
"""

import sys
from pathlib import Path

# Project root (script is in scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# Expected artifacts (canonical names from src/config.py)
REQUIRED_ARTIFACTS = [
    "models/ir_ensemble_best.pkl",
    "models/feature_transformer.pkl",
    "models/selected_features.json",
    "models/optimal_threshold.txt",
]

# Optional artifacts (warn if missing but don't fail)
OPTIONAL_ARTIFACTS = [
    "models/performance_metrics.json",
    "models/base_models_metrics.csv",
]

# Data file
DATA_FILE = "data/all_datasets_merged.csv"


def check_file(filepath: Path, required: bool = True) -> bool:
    """Check if a file exists and print status."""
    exists = filepath.exists()
    status = "✅" if exists else ("❌" if required else "⚠️")
    label = "MISSING" if not exists else "OK"
    print(f"  {status} {filepath.relative_to(PROJECT_ROOT)} ... {label}")
    return exists


def main() -> int:
    """Run all checks and return exit code."""
    print("=" * 60)
    print("Insulin Resistance Prediction - Artifact Check")
    print("=" * 60)
    
    all_ok = True
    
    # Check required artifacts
    print("\n[Required Artifacts]")
    for artifact in REQUIRED_ARTIFACTS:
        path = PROJECT_ROOT / artifact
        if not check_file(path, required=True):
            all_ok = False
    
    # Check optional artifacts
    print("\n[Optional Artifacts]")
    for artifact in OPTIONAL_ARTIFACTS:
        path = PROJECT_ROOT / artifact
        check_file(path, required=False)
    
    # Check data file
    print("\n[Data File]")
    data_path = PROJECT_ROOT / DATA_FILE
    data_exists = check_file(data_path, required=False)
    if not data_exists:
        print("    (Data file is local only - not required in git)")
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All required artifacts present. System ready.")
        return 0
    else:
        print("❌ Some required artifacts are missing!")
        print("   Run: python -m src.train")
        return 1


if __name__ == "__main__":
    sys.exit(main())

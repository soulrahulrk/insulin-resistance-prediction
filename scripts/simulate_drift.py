"""Simulate feature drift and emit alerts for observability playbooks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drift_monitor import alert_if_drift, compute_feature_drift  # noqa: E402

DEFAULT_FEATURES = ["fasting_glucose", "fasting_insulin", "bmi", "age"]


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate drift by perturbing features and logging alerts")
    parser.add_argument("--reference", type=Path, default=Path("all_datasets_merged.csv"), help="Reference dataset")
    parser.add_argument("--candidate", type=Path, help="Optional candidate dataset (skips simulation if provided)")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES, help="Features to monitor")
    parser.add_argument("--sample-size", type=int, default=250, help="Rows sampled from reference for simulation")
    parser.add_argument("--noise-scale", type=float, default=0.25, help="Gaussian noise scale applied to candidate data")
    args = parser.parse_args()

    reference_df = _load_dataset(args.reference)

    if args.candidate:
        candidate_df = _load_dataset(args.candidate)
    else:
        rng = np.random.default_rng(42)
        sample_n = min(args.sample_size, len(reference_df))
        candidate_df = reference_df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        for feature in args.features:
            if feature in candidate_df.columns:
                noise = rng.normal(loc=0.0, scale=args.noise_scale, size=len(candidate_df))
                candidate_df[feature] = candidate_df[feature] * (1 + noise)

    drifted = compute_feature_drift(reference_df, candidate_df, args.features)
    if drifted:
        print(f"Drift detected for features: {', '.join(drifted)}")
    else:
        print("No statistically significant drift detected.")
    alert_if_drift(drifted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

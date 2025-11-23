"""Robustness and sensitivity checks for the IR pipeline."""

from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PATH, FEATURE_TRANSFORMER_PATH


@pytest.fixture(scope="module")
def reference_df():
    if not DATA_PATH.exists():  # pragma: no cover - optional environment
        pytest.skip("Primary dataset not available")
    df = pd.read_csv(DATA_PATH)
    if "ir_label" not in df.columns:
        pytest.skip("Dataset lacks ir_label column")
    return df


@pytest.fixture(scope="module")
def preprocessor():
    if not FEATURE_TRANSFORMER_PATH.exists():  # pragma: no cover - optional environment
        pytest.skip("Feature transformer artifact missing")
    return joblib.load(FEATURE_TRANSFORMER_PATH)


def test_bootstrap_label_prevalence_stability(reference_df):
    sample = reference_df.sample(n=min(2000, len(reference_df)), random_state=0)
    baseline = sample["ir_label"].mean()
    deviations = []
    for seed in range(3):
        boot = sample.sample(n=len(sample), replace=True, random_state=seed)
        deviations.append(abs(boot["ir_label"].mean() - baseline))
    assert max(deviations) < 0.05


def test_temporal_split_respects_order(reference_df):
    if "age" not in reference_df.columns:
        pytest.skip("Dataset lacks age column for pseudo-temporal split")
    ordered = reference_df.sort_values("age").reset_index(drop=True)
    split_idx = int(len(ordered) * 0.7)
    train = ordered.iloc[:split_idx]
    test = ordered.iloc[split_idx:]
    assert not train.empty and not test.empty
    assert train["age"].max() <= test["age"].min()


def test_preprocessor_handles_missing_fasting_insulin(preprocessor, reference_df):
    if "fasting_insulin" not in reference_df.columns:
        pytest.skip("Dataset lacks fasting_insulin column")
    sample = reference_df.sample(n=min(256, len(reference_df)), random_state=1)
    full_X, _ = preprocessor.transform(sample)
    ablated = sample.drop(columns=["fasting_insulin"])
    ablated_X, _ = preprocessor.transform(ablated)
    assert ablated_X.shape == full_X.shape
    assert np.isfinite(ablated_X.values).all()


def test_preprocessor_latency_under_load(preprocessor, reference_df):
    batch = reference_df.sample(n=min(512, len(reference_df)), random_state=2)
    start = time.perf_counter()
    _ = preprocessor.transform(batch)
    duration = time.perf_counter() - start
    assert duration < 5.0

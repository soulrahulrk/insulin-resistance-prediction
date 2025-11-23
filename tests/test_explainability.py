"""Unit tests for fast SHAP explainability utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.explainability_fast import explain_single


class _CalibratedHead:
    def __init__(self):
        self.base_estimator = type("Estimator", (), {"coef_": np.array([[1.0]])})()

    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        logits = meta_features.sum(axis=1)
        probs = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])


def _build_model_package() -> dict:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "a": rng.normal(size=200),
        "b": rng.normal(loc=0.5, size=200),
        "c": rng.normal(loc=-0.25, size=200),
    })
    y = (X["a"] + 0.8 * X["b"] - 0.2 * X["c"] > 0).astype(int)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(X, y)
    return {
        "base_models": {"tree": tree},
        "model_order": ["tree"],
        "calibrated_model": _CalibratedHead(),
    }


def test_fast_shap_returns_top_features():
    package = _build_model_package()
    sample = pd.DataFrame({"a": [0.1], "b": [0.2], "c": [-0.5]})
    result = explain_single(sample, package)
    top3 = result["top3"]
    assert 1 <= len(top3) <= 3
    features = {entry["feature"] for entry in top3}
    assert features.issubset({"a", "b", "c"})
    assert all(isinstance(entry["contribution"], float) for entry in top3)

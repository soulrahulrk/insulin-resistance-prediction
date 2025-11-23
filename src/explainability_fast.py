"""Fast SHAP-style explanations aggregated across stacked base learners."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd
import shap

from src.utils import configure_logger

LOGGER = configure_logger("explainability")


class FastSHAPExplainer:
    """Approximate feature attributions by weighting base-model SHAP values."""

    def __init__(self, model_package: Dict):
        self.model_package = model_package
        self.base_models = model_package.get("base_models", {})
        self.model_order = model_package.get("model_order") or list(self.base_models.keys())
        self.meta_weights = self._extract_meta_weights(model_package.get("calibrated_model"))
        self._explainers: Dict[str, shap.TreeExplainer] = {}

    def _extract_meta_weights(self, calibrated_model) -> np.ndarray:
        if calibrated_model is None:
            return np.ones(len(self.model_order), dtype=float)
        base_estimator = getattr(calibrated_model, "base_estimator", None)
        if base_estimator is None or not hasattr(base_estimator, "coef_"):
            return np.ones(len(self.model_order), dtype=float)
        coefs = np.ravel(base_estimator.coef_)
        if coefs.size < len(self.model_order):
            padded = np.ones(len(self.model_order), dtype=float)
            padded[:coefs.size] = coefs
            coefs = padded
        return coefs

    def _get_tree_explainer(self, model_name: str):
        if model_name not in self._explainers:
            model = self.base_models.get(model_name)
            if model is None:
                raise KeyError(f"Base model '{model_name}' missing from ensemble package")
            self._explainers[model_name] = shap.TreeExplainer(model)
        return self._explainers[model_name]

    def explain(self, sample: pd.DataFrame) -> List[Dict[str, float]]:
        if sample.shape[0] != 1:
            raise ValueError("FastSHAPExplainer expects a single-row DataFrame")
        contributions: Dict[str, float] = {}
        sample_values = sample.reset_index(drop=True)

        for idx, name in enumerate(self.model_order):
            if name not in self.base_models:
                continue
            explainer = self._get_tree_explainer(name)
            shap_values = explainer.shap_values(sample_values)
            if isinstance(shap_values, list):
                shap_row = shap_values[1]
            else:
                shap_row = shap_values
            weight = self.meta_weights[idx] if idx < len(self.meta_weights) else 1.0
            for feature, contribution in zip(sample_values.columns, shap_row[0]):
                scalar = float(np.asarray(contribution).ravel()[0])
                contributions[feature] = contributions.get(feature, 0.0) + scalar * float(weight)

        top_items = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        return [
            {"feature": feature, "contribution": value}
            for feature, value in top_items
        ]


@lru_cache(maxsize=1)
def get_explainer(model_id: str, model_package_serialized: bytes) -> FastSHAPExplainer:
    """Cache FastSHAPExplainer instances per model id for reuse."""
    import joblib

    model_package = joblib.loads(model_package_serialized)
    return FastSHAPExplainer(model_package)


def explain_single(sample: pd.DataFrame, model_package: Dict) -> Dict[str, List[Dict[str, float]]]:
    """Convenience wrapper returning the top three features."""
    explainer = FastSHAPExplainer(model_package)
    top3 = explainer.explain(sample)
    LOGGER.debug("Computed fast explanations for %s", sample.columns.tolist())
    return {"top3": top3}

"""Simple statistical drift detection utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from scipy.stats import ks_2samp

from src.config import DRIFT_ALERT_LOG_PATH
from src.utils import configure_logger

LOGGER = configure_logger("drift_monitor")


def compute_feature_drift(
    reference_df: pd.DataFrame,
    new_df: pd.DataFrame,
    numeric_features: Iterable[str],
    ks_alpha: float = 0.01,
) -> List[str]:
    """Return features whose distributions differ via KS-test."""
    drifted: List[str] = []
    for feature in numeric_features:
        if feature not in reference_df.columns or feature not in new_df.columns:
            continue
        ref_series = reference_df[feature].dropna()
        new_series = new_df[feature].dropna()
        if ref_series.empty or new_series.empty:
            continue
        stat, p_value = ks_2samp(ref_series, new_series)
        LOGGER.debug("KS test feature=%s stat=%.4f p=%.4f", feature, stat, p_value)
        if p_value < ks_alpha:
            drifted.append(feature)
    return drifted


def alert_if_drift(drift_list: List[str], alert_path: Path = DRIFT_ALERT_LOG_PATH) -> None:
    """Append drift alerts to JSONL log when drift is detected."""
    if not drift_list:
        LOGGER.info("No drift detected; alert log not updated")
        return
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_features": drift_list,
    }
    with alert_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    LOGGER.warning("Drift detected for features: %s", ", ".join(drift_list))

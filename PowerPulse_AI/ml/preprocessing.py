"""Training-only feature ranges and raw column selection.

Scaling, when a linear model is selected, stays inside that model's pipeline
and is fitted on training rows only. This module does not fit a scaler on the
full file.
"""

from __future__ import annotations

import pandas as pd

from ml.config import BASE_FEATURES, CYCLE_FEATURE


def raw_columns_for(feature_columns: list[str]) -> list[str]:
    columns = list(BASE_FEATURES)
    if CYCLE_FEATURE in feature_columns:
        columns.append(CYCLE_FEATURE)
    return columns


def training_ranges(frame: pd.DataFrame, columns: list[str]) -> dict:
    ranges = {}
    for column in columns:
        series = frame[column].astype(float)
        ranges[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return ranges

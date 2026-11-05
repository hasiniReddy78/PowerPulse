"""Deterministic features that can be computed from one discharge record.

The builder does not learn statistics from the target, and it does not read
capacity, SoH, condition, or battery identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml.config import BASE_FEATURES, ENGINEERED_FEATURES


class DischargeFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build the selected model matrix from raw discharge measurements."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X, y=None):
        if not self.columns:
            raise ValueError("Feature columns must be provided before fitting.")
        self.feature_columns_ = list(self.columns)
        self.n_features_in_ = int(np.shape(X)[1]) if np.ndim(X) == 2 else None
        return self

    def transform(self, X):
        if not hasattr(self, "feature_columns_"):
            raise RuntimeError("DischargeFeatureBuilder must be fitted before transform.")
        engineered = engineer_discharge_frame(_as_frame(X))
        missing = [column for column in self.feature_columns_ if column not in engineered.columns]
        if missing:
            raise ValueError(f"Missing features required by the saved model: {missing}")
        return engineered.loc[:, self.feature_columns_].to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_columns_, dtype=object)


def _as_frame(values) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.copy()
    raise TypeError(
        "Measurements must be a pandas DataFrame with named columns so the "
        "saved feature order cannot be guessed from positions alone."
    )


def engineer_discharge_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add voltage range and temperature rise without using the target."""
    missing = [column for column in BASE_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing raw measurements: {missing}")

    engineered = frame.copy()
    engineered["voltage_range"] = engineered["voltage_max"] - engineered["voltage_min"]
    engineered["temperature_rise"] = (
        engineered["temperature_mean"] - engineered["ambient_temperature"]
    )
    engineered["temperature_peak_rise"] = (
        engineered["temperature_max"] - engineered["ambient_temperature"]
    )
    return engineered


def assert_no_leakage(columns: list[str]) -> None:
    banned = {"battery_id", "capacity_ah", "soh_percent", "condition"}
    found = sorted(banned.intersection(columns))
    if found:
        raise ValueError(f"Leakage columns were selected as features: {found}")
    unknown = [
        column
        for column in columns
        if column not in BASE_FEATURES + ENGINEERED_FEATURES + ["discharge_cycle"]
    ]
    if unknown:
        raise ValueError(f"Unknown feature columns: {unknown}")

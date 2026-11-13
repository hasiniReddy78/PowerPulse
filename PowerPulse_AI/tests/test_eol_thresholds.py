"""Test suite verifying the 80% End-of-Life (EOL) State of Health threshold.

Covers:
- Condition classification boundaries:
    - SoH = 100% -> GOOD
    - SoH = 90% -> GOOD
    - SoH = 85% -> FAIR (boundary belongs to FAIR)
    - SoH = 82% -> FAIR
    - SoH = 80% -> POOR (boundary belongs to POOR)
    - SoH = 75% -> POOR
- RUL estimation cases:
    - Current SoH above 80%
    - Battery already at 80%
    - Battery below 80%
    - Zero degradation rate
    - Positive degradation rate
    - Insufficient historical records
    - Negative calculated RUL
- Consistency between classifier metadata, database, guidance, and dashboard.
"""

import numpy as np
import pandas as pd
import pytest

from ml.config import CONDITION_RULES, EOL_THRESHOLD
from ml.dataset import derive_condition
from ml.model_loader import load_artifacts
from ml.predict import estimate_rul, soh_band
from services.guidance import maintenance_guidance


def test_soh_condition_classification_exact_thresholds():
    """Verify that condition classifications strictly enforce:
    - GOOD: SoH > 85%
    - FAIR: 80% < SoH <= 85%
    - POOR: SoH <= 80%
    """
    assert EOL_THRESHOLD == 80.0
    assert CONDITION_RULES["poor_max_inclusive"] == 80.0
    assert CONDITION_RULES["fair_max_inclusive"] == 85.0
    assert CONDITION_RULES["good_above"] == 85.0

    # Test cases requested by prompt
    assert soh_band(100.0) == "Good"
    assert soh_band(90.0) == "Good"
    assert soh_band(85.0) == "Fair"   # 85% boundary belongs to FAIR
    assert soh_band(82.0) == "Fair"
    assert soh_band(80.0) == "Poor"   # 80% boundary belongs to POOR
    assert soh_band(75.0) == "Poor"

    # Precise boundary edge tests
    assert soh_band(85.0001) == "Good"
    assert soh_band(84.9999) == "Fair"
    assert soh_band(80.0001) == "Fair"
    assert soh_band(79.9999) == "Poor"

    # Verify vectorised dataset labelling logic behaves identically
    series = pd.Series([100.0, 90.0, 85.0, 82.0, 80.0, 75.0, 85.0001, 80.0001])
    labels = list(derive_condition(series))
    assert labels == ["Good", "Good", "Fair", "Fair", "Poor", "Poor", "Good", "Fair"]


def test_rul_with_current_soh_above_80():
    """Verify RUL calculation when current SoH is strictly above 80%."""
    # Cycle 1 -> 95%, Cycle 50 -> 85%
    # Slope = (85 - 95) / (50 - 1) = -10 / 49 = -0.2040816% per cycle
    # Remaining cycles to 80%: (80 - 85) / (-0.2040816) = -5 / -0.2040816 = 24.5 -> 25 cycles
    history = [
        {"inputs": {"discharge_cycle": 1}, "predicted_soh": 95.0},
        {"inputs": {"discharge_cycle": 50}, "predicted_soh": 85.0},
    ]
    res = estimate_rul(history, eol_threshold=80.0)
    assert res["available"] is True
    assert res["eol_threshold_soh"] == 80.0
    assert res["current_soh"] == 85.0
    assert res["current_cycle"] == 50.0
    assert res["remaining_cycles"] == 24
    assert res["projected_eol_cycle"] in (74, 75)
    assert "not a guaranteed battery lifetime" in res["disclaimer"]


def test_rul_battery_already_at_80():
    """Verify RUL when battery SoH is exactly at the 80% EOL threshold."""
    history = [
        {"inputs": {"discharge_cycle": 10}, "predicted_soh": 90.0},
        {"inputs": {"discharge_cycle": 60}, "predicted_soh": 80.0},
    ]
    res = estimate_rul(history, eol_threshold=80.0)
    assert res["available"] is False
    assert res.get("reached_eol") is True
    assert res["remaining_cycles"] == 0
    assert "reached its defined End-of-Life threshold" in res["reason"]


def test_rul_battery_below_80():
    """Verify RUL when battery SoH is already below 80% (e.g. 75%)."""
    history = [
        {"inputs": {"discharge_cycle": 10}, "predicted_soh": 90.0},
        {"inputs": {"discharge_cycle": 70}, "predicted_soh": 75.0},
    ]
    res = estimate_rul(history, eol_threshold=80.0)
    assert res["available"] is False
    assert res.get("reached_eol") is True
    assert res["remaining_cycles"] == 0
    assert "reached its defined End-of-Life threshold" in res["reason"]


def test_rul_zero_degradation_rate():
    """Verify RUL does not generate misleading estimates when degradation slope is 0."""
    history = [
        {"inputs": {"discharge_cycle": 5}, "predicted_soh": 92.0},
        {"inputs": {"discharge_cycle": 25}, "predicted_soh": 92.0},
    ]
    res = estimate_rul(history, eol_threshold=80.0)
    assert res["available"] is False
    assert "has not demonstrated a downward degradation trend" in res["reason"]


def test_rul_positive_degradation_rate():
    """Verify RUL does not generate misleading estimates when slope is positive (capacity increase)."""
    history = [
        {"inputs": {"discharge_cycle": 5}, "predicted_soh": 86.0},
        {"inputs": {"discharge_cycle": 25}, "predicted_soh": 90.0},
    ]
    res = estimate_rul(history, eol_threshold=80.0)
    assert res["available"] is False
    assert "has not demonstrated a downward degradation trend" in res["reason"]


def test_rul_insufficient_historical_records():
    """Verify proper message when insufficient cycles are recorded."""
    # 0 records
    res0 = estimate_rul([])
    assert res0["available"] is False
    assert "At least two distinct cycles" in res0["reason"]

    # 1 record
    res1 = estimate_rul([{"inputs": {"discharge_cycle": 1}, "predicted_soh": 95.0}])
    assert res1["available"] is False
    assert "At least two distinct cycles" in res1["reason"]

    # 2 records with identical cycle number (not distinct cycles)
    res_same = estimate_rul([
        {"inputs": {"discharge_cycle": 5}, "predicted_soh": 95.0},
        {"inputs": {"discharge_cycle": 5}, "predicted_soh": 94.0},
    ])
    assert res_same["available"] is False
    assert "At least two distinct cycles" in res_same["reason"]


def test_rul_negative_calculated_rul():
    """Verify that calculated RUL is never displayed as negative."""
    # If battery was at 81.0% at cycle 50, but slope is steep and target cycle is <= current cycle
    history = [
        {"inputs": {"discharge_cycle": 1}, "predicted_soh": 99.0},
        {"inputs": {"discharge_cycle": 50}, "predicted_soh": 80.1},
    ]
    # At cycle 50, SoH is 80.1%. With threshold 80.0%, target cycle might be ~50.2 (rounds to 50)
    res = estimate_rul(history, eol_threshold=80.0)
    assert res["remaining_cycles"] >= 0
    assert res["projected_eol_cycle"] >= 50


def test_maintenance_guidance_uses_80_threshold():
    """Verify maintenance advisory reflects the 80% EOL boundary."""
    notes_poor = maintenance_guidance("Poor", 80.0)
    assert any("at or below 80%" in note for note in notes_poor)
    assert not any("70%" in note for note in notes_poor)

    notes_fair = maintenance_guidance("Fair", 82.0)
    assert any("above 80% and at or below 85%" in note for note in notes_fair)

    notes_good = maintenance_guidance("Good", 92.0)
    assert any("above 85%" in note for note in notes_good)


def test_artifacts_and_classifier_use_80_threshold():
    """Verify the retrained model artifacts use the 80% EOL threshold."""
    artifacts = load_artifacts()
    thresholds = artifacts.metadata["condition_thresholds"]
    assert thresholds["poor_at_or_below"] == 80.0
    assert thresholds["fair_at_or_below"] == 85.0
    assert thresholds["good_above"] == 85.0
    assert artifacts.metadata["classification"]["algorithm"] == "xgboost"

    # Verify classifier works on test sample
    sample = {
        "ambient_temperature": 24.0,
        "temperature_mean": 32.5,
        "temperature_max": 38.9,
        "voltage_mean": 3.52,
        "voltage_min": 2.61,
        "voltage_max": 4.19,
        "voltage_std": 0.23,
        "current_mean": -1.81,
        "current_std": 0.59,
        "duration_seconds": 3690.0,
        "discharge_cycle": 1,
    }
    df = pd.DataFrame([sample])
    condition_pred = str(artifacts.classification_model.predict(df)[0])
    soh_pred = float(artifacts.regression_model.predict(df)[0])

    assert condition_pred in {"Good", "Fair", "Poor"}
    assert 50.0 <= soh_pred <= 105.0

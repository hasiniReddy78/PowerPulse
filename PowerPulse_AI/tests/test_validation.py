import pytest

from ml.predict import estimate_rul
from services.validation import validate_lithium_ion

RANGES = {
    "ambient_temperature": {"min": 4.0, "max": 44.0},
    "temperature_mean": {"min": 8.0, "max": 54.0},
    "temperature_max": {"min": 11.0, "max": 62.0},
    "voltage_mean": {"min": 3.0, "max": 3.7},
    "voltage_min": {"min": 1.5, "max": 2.7},
    "voltage_max": {"min": 3.9, "max": 4.3},
    "voltage_std": {"min": 0.2, "max": 0.5},
    "current_mean": {"min": -4.0, "max": -0.5},
    "current_std": {"min": 0.07, "max": 2.0},
    "duration_seconds": {"min": 1500.0, "max": 6500.0},
}
METADATA = {"raw_input_columns": list(RANGES), "raw_feature_ranges_training": RANGES}


def _valid():
    return {
        "ambient_temperature": 24.0,
        "temperature_mean": 32.0,
        "temperature_max": 39.0,
        "voltage_mean": 3.5,
        "voltage_min": 2.5,
        "voltage_max": 4.1,
        "voltage_std": 0.24,
        "current_mean": -1.8,
        "current_std": 0.5,
        "duration_seconds": 3600.0,
    }


def test_positive_current_is_converted_and_pack_voltage_is_rejected():
    cleaned, errors, warnings = validate_lithium_ion({**_valid(), "current_mean": 1.8}, METADATA)
    assert not errors
    assert cleaned["current_mean"] == -1.8
    assert warnings

    _, pack_errors, _ = validate_lithium_ion({**_valid(), "voltage_mean": 350.0, "voltage_max": 400.0}, METADATA)
    assert any("voltage_mean" in message for message in pack_errors)
    assert any("voltage_max" in message for message in pack_errors)


def test_inconsistent_voltage_and_missing_values_are_rejected():
    _, errors, _ = validate_lithium_ion({**_valid(), "voltage_min": 4.0, "voltage_mean": 3.2}, METADATA)
    assert errors
    _, missing_errors, _ = validate_lithium_ion({"ambient_temperature": 24}, METADATA)
    assert missing_errors


def test_thermal_anomaly_generates_warning():
    # Peak temperature 23 degrees above ambient
    payload = {**_valid(), "ambient_temperature": 20.0, "temperature_mean": 35.0, "temperature_max": 43.0}
    cleaned, errors, warnings = validate_lithium_ion(payload, METADATA)
    assert not errors
    assert any("Thermal anomaly" in w for w in warnings)


def test_rul_estimation_logic():
    # Insufficient data
    single = [{"inputs": {"discharge_cycle": 1}, "predicted_soh": 98.0}]
    res_single = estimate_rul(single)
    assert not res_single["available"]

    # Historical degradation: cycle 1 -> 98%, cycle 50 -> 88%
    # (slope = -10/49 = -0.20408% per cycle)
    # Remaining cycles to 70% threshold: (70 - 88) / (-0.20408) = 88.2 -> 88 cycles
    history = [
        {"inputs": {"discharge_cycle": 1}, "predicted_soh": 98.0},
        {"inputs": {"discharge_cycle": 50}, "predicted_soh": 88.0},
    ]
    res = estimate_rul(history)
    assert res["available"]
    assert res["remaining_cycles"] == 88
    assert res["projected_eol_cycle"] == 138
    assert pytest.approx(res["degradation_rate_per_cycle"], 0.01) == 0.204

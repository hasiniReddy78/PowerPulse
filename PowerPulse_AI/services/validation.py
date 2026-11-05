"""Reject measurements that do not match the trained feature definition and flag anomalies."""

from __future__ import annotations


def _as_float(name: str, value, errors: list[str]) -> float | None:
    if value is None or value == "":
        errors.append(f"{name} is required.")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a number.")
        return None
    if number != number or number in (float("inf"), float("-inf")):
        errors.append(f"{name} must be a finite number.")
        return None
    return number


def validate_lithium_ion(raw: dict, metadata: dict) -> tuple[dict, list[str], list[str]]:
    """Return a model-ready record, validation errors, and non-blocking behavioral warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    required = list(metadata["raw_input_columns"])
    ranges = metadata["raw_feature_ranges_training"]
    cleaned: dict[str, float] = {}

    for column in required:
        cleaned[column] = _as_float(column, raw.get(column), errors)

    if errors:
        return cleaned, errors, warnings

    if "current_mean" in cleaned and cleaned["current_mean"] > 0:
        cleaned["current_mean"] = -cleaned["current_mean"]
        warnings.append(
            "A positive average current was converted to a negative value to match "
            "the laboratory discharge sign convention."
        )

    for column in required:
        value = cleaned[column]
        bounds = ranges[column]
        if value < bounds["min"] or value > bounds["max"]:
            errors.append(
                f"{column} is {value:.6g}, outside the training range "
                f"{bounds['min']:.6g} to {bounds['max']:.6g}. "
                "No prediction was made because the measurement is not compatible "
                "with the laboratory feature definition."
            )

    # Physical consistency checks
    if {"voltage_min", "voltage_mean", "voltage_max"}.issubset(cleaned):
        if cleaned["voltage_min"] > cleaned["voltage_max"]:
            errors.append("Minimum voltage cannot exceed maximum voltage.")
        elif (
            cleaned["voltage_min"] > cleaned["voltage_mean"]
            or cleaned["voltage_mean"] > cleaned["voltage_max"]
        ):
            errors.append("Average voltage must lie between minimum and maximum voltage.")

        # Abnormal voltage drop / high internal resistance indicator
        v_drop = cleaned["voltage_max"] - cleaned["voltage_min"]
        if v_drop > 1.8:
            warnings.append(
                f"Abnormal voltage drop observed ({v_drop:.2f} V). This suggests high internal cell resistance or deep discharge stress."
            )

    if {"temperature_mean", "temperature_max"}.issubset(cleaned):
        if cleaned["temperature_mean"] > cleaned["temperature_max"]:
            errors.append("Average battery temperature cannot exceed maximum battery temperature.")

    if {"temperature_mean", "ambient_temperature"}.issubset(cleaned):
        if cleaned["temperature_mean"] < cleaned["ambient_temperature"]:
            errors.append(
                "Average battery temperature is below ambient temperature. "
                "That pattern does not occur in the training discharges."
            )

    if {"temperature_max", "ambient_temperature"}.issubset(cleaned):
        temp_rise = cleaned["temperature_max"] - cleaned["ambient_temperature"]
        if temp_rise > 20.0:
            warnings.append(
                f"Thermal anomaly: Peak temperature rise is {temp_rise:.1f} °C above ambient. Monitor for thermal management issues."
            )

    if "voltage_std" in cleaned and cleaned["voltage_std"] <= 0:
        errors.append("Voltage standard deviation must be positive.")
    if "current_std" in cleaned and cleaned["current_std"] <= 0:
        errors.append("Current standard deviation must be positive.")
    if "duration_seconds" in cleaned and cleaned["duration_seconds"] <= 0:
        errors.append("Discharge duration must be positive.")
    if "discharge_cycle" in cleaned and cleaned["discharge_cycle"] < 1:
        errors.append("Discharge cycle must be at least 1.")

    return cleaned, errors, warnings

"""Prediction and health assessment entry points used by the application and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import CONDITION_RULES, EOL_THRESHOLD
from ml.errors import InvalidMeasurementError
from ml.model_loader import Artifacts, load_artifacts
from services.guidance import maintenance_guidance
from services.validation import validate_lithium_ion


def soh_band(soh: float) -> str:
    if soh <= CONDITION_RULES["poor_max_inclusive"]:
        return "Poor"
    if soh <= CONDITION_RULES["fair_max_inclusive"]:
        return "Fair"
    return "Good"


EOL_DISCLAIMER = "RUL is an estimate based on the observed degradation trend, not a guaranteed battery lifetime."


def estimate_rul(
    history_records: list[dict],
    current_soh: float | None = None,
    current_cycle: int | float | None = None,
    eol_threshold: float = EOL_THRESHOLD,
) -> dict:
    """Estimate Remaining Useful Life (RUL) in discharge cycles based on historical degradation.

    Requires at least 2 distinct cycles with SoH estimates.
    Estimates the number of cycles remaining until the battery reaches the defined EOL threshold.
    """
    points: dict[float, float] = {}
    for r in history_records:
        inputs = r.get("inputs") or {}
        soh = r.get("predicted_soh")
        cycle = inputs.get("discharge_cycle")
        if soh is not None and cycle is not None:
            try:
                points[float(cycle)] = float(soh)
            except (ValueError, TypeError):
                continue

    if current_cycle is not None and current_soh is not None:
        try:
            points[float(current_cycle)] = float(current_soh)
        except (ValueError, TypeError):
            pass

    if len(points) < 2:
        return {
            "available": False,
            "reason": "At least two distinct cycles with SoH estimates are needed to compute degradation rate and project RUL.",
            "data_points": len(points),
            "eol_threshold_soh": float(eol_threshold),
            "disclaimer": EOL_DISCLAIMER,
        }

    sorted_cycles = np.array(sorted(points.keys()), dtype=float)
    sorted_sohs = np.array([points[c] for c in sorted_cycles], dtype=float)
    last_cycle = float(sorted_cycles[-1])
    last_soh = float(sorted_sohs[-1])

    # Case 1: Current SoH is already at or below the EOL threshold
    if last_soh <= eol_threshold:
        return {
            "available": False,
            "reached_eol": True,
            "remaining_cycles": 0,
            "reason": "Battery has reached its defined End-of-Life threshold.",
            "data_points": len(points),
            "current_cycle": last_cycle,
            "current_soh": last_soh,
            "eol_threshold_soh": float(eol_threshold),
            "disclaimer": EOL_DISCLAIMER,
        }

    # Linear trend fit: SoH = slope * cycle + intercept
    slope, intercept = np.polyfit(sorted_cycles, sorted_sohs, 1)

    # Case 2: Degradation slope is zero or positive (no downward trend, numerical precision tolerance)
    if slope >= 0 or np.isclose(slope, 0.0, atol=1e-7):
        return {
            "available": False,
            "reason": "Observed SoH has not demonstrated a downward degradation trend across the recorded cycles.",
            "data_points": len(points),
            "slope": float(slope),
            "current_cycle": last_cycle,
            "current_soh": last_soh,
            "eol_threshold_soh": float(eol_threshold),
            "disclaimer": EOL_DISCLAIMER,
        }

    # Solve for cycle where SoH = eol_threshold: target_cycle = (eol_threshold - intercept) / slope
    target_cycle = (eol_threshold - intercept) / slope
    remaining_raw = target_cycle - last_cycle

    # Case 3: Negative calculated RUL clamped to 0
    remaining = max(0, int(round(remaining_raw)))

    return {
        "available": True,
        "remaining_cycles": remaining,
        "projected_eol_cycle": max(int(round(last_cycle)), int(round(target_cycle))),
        "degradation_rate_per_cycle": float(abs(slope)),
        "eol_threshold_soh": float(eol_threshold),
        "data_points": len(points),
        "current_cycle": last_cycle,
        "current_soh": last_soh,
        "disclaimer": EOL_DISCLAIMER,
    }


def predict_lithium_ion(
    measurements: dict,
    model_dir: Path | None = None,
    artifacts: Artifacts | None = None,
) -> dict:
    loaded = artifacts if artifacts is not None else load_artifacts(model_dir)
    cleaned, errors, warnings = validate_lithium_ion(measurements, loaded.metadata)
    if errors:
        raise InvalidMeasurementError(errors)

    frame = pd.DataFrame([cleaned])
    soh = float(loaded.regression_model.predict(frame)[0])
    condition = str(loaded.classification_model.predict(frame)[0])
    band = soh_band(soh)
    if condition != band:
        warnings.append(
            f"The regression band is {band} and the classifier predicts {condition}. "
            "The two models can disagree. Neither result is a measured SoH."
        )

    # Early warning for severe degradation
    if soh <= EOL_THRESHOLD or condition == "Poor":
        warnings.append(
            f"CRITICAL ALERT: Estimated SoH has dropped into or below the {int(EOL_THRESHOLD)}% threshold. "
            "Replacement or detailed cell capacity inspection recommended."
        )

    test_metrics = loaded.metadata["regression"]["test_metrics"]
    limitations = list(loaded.metadata["limitations"])
    return {
        "status": "predicted",
        "category": "lithium_ion",
        "model_version": loaded.metadata["model_version"],
        "inputs_used": cleaned,
        "predicted_soh": soh,
        "predicted_condition": condition,
        "soh_band": band,
        "warnings": warnings,
        "guidance": maintenance_guidance(condition, soh),
        "feature_importance": loaded.metadata["regression"]["feature_importance"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "regression_algorithm": loaded.metadata["regression"]["algorithm"],
        "classification_algorithm": loaded.metadata["classification"]["algorithm"],
        "limitations": limitations,
        "train_batteries": loaded.metadata["split"]["train_batteries"],
        "test_batteries": loaded.metadata["split"]["test_batteries"],
    }


def assess(
    measurements: dict,
    category: str = "lithium_ion",
    model_dir: Path | None = None,
    battery_id: str | None = None,
    user_id: int | None = None,
    warranty_info: dict | None = None,
    as_of: Any = None,
    db_path: Path | None = None,
) -> dict:
    """Assess lithium-ion battery health from measurements after validating warranty policy.

    If battery_id or warranty_info is provided, validates warranty before ML execution.
    If warranty is EXPIRED, raises WarrantyExpiredError and executes NO ML models.
    """
    from utils.warranty import (
        WarrantyExpiredError,
        WarrantyNotRegisteredError,
        check_warranty_validity,
    )

    if battery_id is not None or warranty_info is not None:
        rec = warranty_info
        if rec is None and battery_id is not None:
            from database.db import get_battery_warranty

            rec = get_battery_warranty(battery_id, user_id=user_id, as_of=as_of, path=db_path)

        validity = check_warranty_validity(rec, as_of=as_of)
        if not validity["is_valid"]:
            if validity["status"] == "EXPIRED":
                raise WarrantyExpiredError(
                    "WARRANTY EXPIRED – Battery health assessment is unavailable under the current warranty policy."
                )
            elif validity["status"] == "NOT_REGISTERED":
                raise WarrantyNotRegisteredError(validity["message"])
            else:
                raise WarrantyExpiredError(validity["message"])

    result = predict_lithium_ion(measurements, model_dir=model_dir)
    if battery_id:
        result["battery_id"] = battery_id
    return result

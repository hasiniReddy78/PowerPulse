"""Prediction and health assessment entry points used by the application and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import CONDITION_RULES
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


def estimate_rul(
    history_records: list[dict],
    current_soh: float | None = None,
    current_cycle: int | float | None = None,
    eol_threshold: float = 70.0,
) -> dict:
    """Estimate Remaining Useful Life (RUL) in discharge cycles based on historical degradation.

    Requires at least 2 distinct cycles with SoH estimates.
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
        }

    sorted_cycles = np.array(sorted(points.keys()), dtype=float)
    sorted_sohs = np.array([points[c] for c in sorted_cycles], dtype=float)

    # Linear trend fit: SoH = slope * cycle + intercept
    slope, intercept = np.polyfit(sorted_cycles, sorted_sohs, 1)

    # For degradation, slope should be negative (loss of SoH over cycles)
    if slope >= 0:
        return {
            "available": False,
            "reason": "Observed SoH has not demonstrated a downward degradation trend across the recorded cycles.",
            "data_points": len(points),
            "slope": float(slope),
            "current_cycle": float(sorted_cycles[-1]),
            "current_soh": float(sorted_sohs[-1]),
        }

    # Solve for cycle where SoH = eol_threshold
    target_cycle = (eol_threshold - intercept) / slope
    remaining = max(0, int(round(target_cycle - sorted_cycles[-1])))

    return {
        "available": True,
        "remaining_cycles": remaining,
        "projected_eol_cycle": int(round(target_cycle)),
        "degradation_rate_per_cycle": float(abs(slope)),
        "eol_threshold_soh": float(eol_threshold),
        "data_points": len(points),
        "current_cycle": float(sorted_cycles[-1]),
        "current_soh": float(sorted_sohs[-1]),
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
    if soh <= 70.0 or condition == "Poor":
        warnings.append(
            "CRITICAL ALERT: Estimated SoH has dropped into or below the 70% threshold. "
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
) -> dict:
    """Assess lithium-ion battery health from measurements."""
    return predict_lithium_ion(measurements, model_dir=model_dir)

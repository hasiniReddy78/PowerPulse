"""SHAP explainability module for PowerPulse lithium-ion models.

Provides local (per-prediction) and global (feature-level) model interpretability
using Shapley Additive exPlanations (SHAP).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from ml.dataset import load_clean_dataset
from ml.model_loader import Artifacts, load_artifacts
from services.validation import validate_lithium_ion

_REG_EXPLAINER_CACHE: dict[str, shap.TreeExplainer] = {}
_CLF_EXPLAINER_CACHE: dict[str, shap.TreeExplainer] = {}
_GLOBAL_SHAP_CACHE: dict[int, dict] = {}


def _get_inner_tree_estimator(pipeline_model: Any) -> Any:
    """Extract the underlying fitted tree estimator from a Pipeline or wrapper."""
    if hasattr(pipeline_model, "named_steps"):
        return _get_inner_tree_estimator(pipeline_model.named_steps["model"])
    if hasattr(pipeline_model, "__class__") and pipeline_model.__class__.__name__ == "WrappedXGBClassifier":
        return getattr(pipeline_model, "estimator_", pipeline_model)
    return pipeline_model


def get_regression_explainer(artifacts: Artifacts) -> shap.TreeExplainer:
    """Return a cached TreeExplainer for the regression model."""
    version = artifacts.metadata.get("model_version", "1.0.0")
    if version not in _REG_EXPLAINER_CACHE:
        reg_model = artifacts.regression_model.named_steps["model"]
        inner_estimator = _get_inner_tree_estimator(reg_model)
        _REG_EXPLAINER_CACHE[version] = shap.TreeExplainer(inner_estimator)
    return _REG_EXPLAINER_CACHE[version]


def _feature_friendly_description(feature: str, val: float, shap_val: float) -> str:
    """Provide an intuitive automotive/battery engineering explanation."""
    direction = "raised" if shap_val > 0 else "lowered"
    abs_shap = abs(shap_val)

    if feature == "voltage_mean":
        return f"Mean voltage ({val:.3f} V) {direction} estimated SoH by {abs_shap:.2f} pp."
    if feature == "voltage_range":
        return f"Voltage span ({val:.3f} V) {direction} estimated SoH by {abs_shap:.2f} pp."
    if feature == "voltage_std":
        return f"Voltage variance during discharge ({val:.4f} V) {direction} estimated SoH by {abs_shap:.2f} pp."
    if feature == "temperature_mean":
        return f"Average battery temperature ({val:.1f} °C) {direction} estimated SoH by {abs_shap:.2f} pp."
    if feature == "temperature_peak_rise":
        return f"Peak temperature rise ({val:.1f} °C above ambient) {direction} estimated SoH by {abs_shap:.2f} pp."
    if feature == "temperature_rise":
        return f"Mean temperature rise ({val:.1f} °C above ambient) {direction} estimated SoH by {abs_shap:.2f} pp."
    if feature == "duration_seconds":
        return f"Discharge duration ({val:.0f} s / {val/60:.1f} min) {direction} estimated SoH by {abs_shap:.2f} pp."
    if feature == "current_mean":
        return f"Discharge current ({val:.2f} A) {direction} estimated SoH by {abs_shap:.2f} pp."
    if feature == "ambient_temperature":
        return f"Chamber ambient temperature ({val:.1f} °C) {direction} estimated SoH by {abs_shap:.2f} pp."
    return f"{feature} = {val:.4g} {direction} estimated SoH by {abs_shap:.2f} pp."


def explain_prediction(
    measurements: dict,
    model_dir: Path | None = None,
    artifacts: Artifacts | None = None,
) -> dict:
    """Compute local SHAP feature attributions for a single battery prediction.

    Decomposes the prediction into:
        predicted_soh = base_value + sum(shap_values)
    """
    loaded = artifacts if artifacts is not None else load_artifacts(model_dir)
    cleaned, errors, _ = validate_lithium_ion(measurements, loaded.metadata)
    if errors:
        raise ValueError(f"Cannot explain invalid measurements: {'; '.join(errors)}")

    features_step = loaded.regression_model.named_steps["features"]
    frame = pd.DataFrame([cleaned])
    X_trans = features_step.transform(frame)
    feature_names = list(features_step.feature_columns_)

    explainer = get_regression_explainer(loaded)
    shap_output = explainer(X_trans)

    # Base value (mean training prediction)
    raw_base = explainer.expected_value
    base_value = float(raw_base[0] if isinstance(raw_base, (list, np.ndarray)) else raw_base)

    raw_shap_vals = shap_output.values[0]
    predicted_soh = float(base_value + float(np.sum(raw_shap_vals)))

    contributions = []
    for name, input_val, s_val in zip(feature_names, X_trans[0], raw_shap_vals):
        s_float = float(s_val)
        contributions.append(
            {
                "feature": name,
                "input_value": float(input_val),
                "shap_value": s_float,
                "abs_shap": abs(s_float),
                "effect": "positive" if s_float >= 0 else "negative",
                "description": _feature_friendly_description(name, float(input_val), s_float),
            }
        )

    # Sort descending by absolute impact
    contributions.sort(key=lambda item: item["abs_shap"], reverse=True)

    positive_drivers = [c for c in contributions if c["shap_value"] > 0]
    negative_drivers = [c for c in contributions if c["shap_value"] < 0]

    return {
        "status": "explained",
        "model_version": loaded.metadata.get("model_version", "1.0.0"),
        "base_value": base_value,
        "predicted_soh": predicted_soh,
        "total_shap_sum": float(np.sum(raw_shap_vals)),
        "contributions": contributions,
        "top_positive_drivers": positive_drivers[:3],
        "top_negative_drivers": negative_drivers[:3],
    }


def create_shap_waterfall_chart(explanation: dict) -> go.Figure:
    """Create an interactive Plotly horizontal attribution chart."""
    contributions = explanation["contributions"]
    base_val = explanation["base_value"]
    pred_soh = explanation["predicted_soh"]

    # Select top 8 most influential features for visual clarity
    top_items = contributions[:8]
    # Reverse so largest impact appears at the top of the chart
    plot_items = list(reversed(top_items))

    names = [f"{item['feature']} ({item['input_value']:.2g})" for item in plot_items]
    shap_vals = [item["shap_value"] for item in plot_items]
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in shap_vals]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=names,
            x=shap_vals,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=1, color="rgba(0,0,0,0.2)"),
            ),
            text=[f"{v:+.2f} pp" for v in shap_vals],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>SHAP Impact: %{x:+.3f} SoH pp<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"SHAP Feature Attribution (Baseline: {base_val:.1f}% → Predicted: {pred_soh:.1f}%)",
            font=dict(size=15),
        ),
        xaxis=dict(
            title="Impact on Estimated State of Health (percentage points)",
            zeroline=True,
            zerolinecolor="rgba(100,100,100,0.5)",
            zerolinewidth=2,
        ),
        yaxis=dict(title="Key Telemetry Features"),
        margin=dict(l=20, r=20, t=50, b=40),
        height=380,
        plot_bgcolor="rgba(240, 244, 248, 0.4)",
    )
    return fig


def get_global_feature_importance(sample_size: int = 150) -> dict:
    """Compute global mean |SHAP| feature importance across the dataset."""
    if sample_size in _GLOBAL_SHAP_CACHE:
        return _GLOBAL_SHAP_CACHE[sample_size]

    artifacts = load_artifacts()
    frame = load_clean_dataset()
    if len(frame) > sample_size:
        sample_df = frame.sample(n=sample_size, random_state=42)
    else:
        sample_df = frame

    features_step = artifacts.regression_model.named_steps["features"]
    feature_names = list(features_step.feature_columns_)
    raw_columns = artifacts.metadata["raw_input_columns"]
    X_sample_raw = sample_df[raw_columns]

    X_trans = features_step.transform(X_sample_raw)
    explainer = get_regression_explainer(artifacts)
    shap_vals = explainer(X_trans).values

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    importance_list = [
        {"feature": name, "mean_abs_shap": float(val)}
        for name, val in zip(feature_names, mean_abs_shap)
    ]
    importance_list.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

    result = {
        "sample_size": len(sample_df),
        "rankings": importance_list,
    }
    _GLOBAL_SHAP_CACHE[sample_size] = result
    return result


def create_global_importance_chart(global_info: dict) -> go.Figure:
    """Create a bar chart of dataset-wide global SHAP importance."""
    rankings = global_info["rankings"]
    names = [item["feature"] for item in reversed(rankings)]
    values = [item["mean_abs_shap"] for item in reversed(rankings)]

    fig = go.Figure(
        go.Bar(
            y=names,
            x=values,
            orientation="h",
            marker=dict(color="#3b82f6"),
            text=[f"{v:.3f} pp" for v in values],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f} SoH pp<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Global SHAP Feature Importance (Mean |SHAP| across {global_info['sample_size']} cycles)",
            font=dict(size=15),
        ),
        xaxis=dict(title="Mean Absolute SHAP Value (Impact on SoH in pp)"),
        yaxis=dict(title="Feature"),
        margin=dict(l=20, r=20, t=50, b=40),
        height=420,
        plot_bgcolor="rgba(240, 244, 248, 0.4)",
    )
    return fig

import pytest

from ml.explainability import (
    create_global_importance_chart,
    create_shap_waterfall_chart,
    explain_prediction,
    get_global_feature_importance,
)
from ml.model_loader import load_artifacts


def _valid_inputs():
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


def test_explain_prediction_math_and_structure():
    artifacts = load_artifacts()
    inputs = _valid_inputs()
    explanation = explain_prediction(inputs, artifacts=artifacts)

    assert explanation["status"] == "explained"
    assert "base_value" in explanation
    assert "predicted_soh" in explanation
    assert "contributions" in explanation
    assert len(explanation["contributions"]) == len(artifacts.metadata["feature_columns"])

    # Verify additivity: base_value + sum(shap_values) == predicted_soh
    shap_sum = sum(c["shap_value"] for c in explanation["contributions"])
    assert explanation["base_value"] + shap_sum == pytest.approx(explanation["predicted_soh"], abs=1e-4)

    # Check that descriptions are generated
    for c in explanation["contributions"]:
        assert c["description"]
        assert c["effect"] in {"positive", "negative"}

    # Check top drivers
    assert isinstance(explanation["top_positive_drivers"], list)
    assert isinstance(explanation["top_negative_drivers"], list)


def test_create_shap_waterfall_chart_generates_figure():
    explanation = explain_prediction(_valid_inputs())
    fig = create_shap_waterfall_chart(explanation)
    assert fig is not None
    assert len(fig.data) > 0
    assert fig.data[0].orientation == "h"


def test_explain_prediction_rejects_invalid_inputs():
    invalid_inputs = {**_valid_inputs(), "voltage_mean": 350.0}
    with pytest.raises(ValueError):
        explain_prediction(invalid_inputs)


def test_global_feature_importance():
    global_info = get_global_feature_importance(sample_size=30)
    assert global_info["sample_size"] == 30
    assert len(global_info["rankings"]) > 0
    # Values should be non-negative mean absolute SHAP
    for item in global_info["rankings"]:
        assert item["mean_abs_shap"] >= 0

    fig = create_global_importance_chart(global_info)
    assert fig is not None
    assert len(fig.data) > 0


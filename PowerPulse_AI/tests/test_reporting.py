from services.reporting import assessment_csv, assessment_pdf


def test_report_contains_prediction_and_limitations():
    record = {
        "assessed_at": "2026-11-04T00:00:00+00:00",
        "battery_identifier": "CELL-9",
        "category": "lithium_ion",
        "status": "predicted",
        "model_version": "1.0.0",
        "predicted_soh": 76.5,
        "predicted_condition": "Fair",
        "soh_band": "Fair",
        "inputs": {"voltage_mean": 3.4},
        "warnings": ["Models disagreed."],
        "limitations": ["Laboratory NASA results do not validate real-world vehicle performance."],
    }
    csv_text = assessment_csv(record).decode("utf-8")
    assert "CELL-9" in csv_text
    assert "76.5" in csv_text
    assert "real-world vehicle performance" in csv_text
    pdf = assessment_pdf(record)
    assert pdf.startswith(b"%PDF-1.4")
    assert b"CELL-9" in pdf
    assert b"%%EOF" in pdf


def test_report_contains_shap_explanations():
    record = {
        "assessed_at": "2026-11-04T00:00:00+00:00",
        "battery_identifier": "CELL-10",
        "category": "lithium_ion",
        "status": "predicted",
        "model_version": "1.0.0",
        "predicted_soh": 85.0,
        "predicted_condition": "Good",
        "soh_band": "Good",
        "inputs": {"voltage_mean": 3.6},
        "shap_explanation": {
            "base_value": 72.5,
            "top_positive_drivers": [{"feature": "voltage_mean", "input_value": 3.6, "shap_value": 4.5}],
            "top_negative_drivers": [{"feature": "temperature_rise", "input_value": 15.0, "shap_value": -1.2}],
        },
        "warnings": [],
        "limitations": [],
    }
    csv_text = assessment_csv(record).decode("utf-8")
    assert "shap.base_value" in csv_text
    assert "shap.positive.voltage_mean" in csv_text
    pdf = assessment_pdf(record)
    assert b"SHAP EXPLAINABILITY" in pdf

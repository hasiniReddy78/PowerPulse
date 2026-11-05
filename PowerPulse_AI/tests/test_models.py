import json
import shutil

import pandas as pd
import pytest

from ml.config import LEAKAGE_COLUMNS, MODELS_DIR
from ml.dataset import load_clean_dataset
from ml.errors import InvalidMeasurementError, ModelNotFoundError
from ml.model_loader import load_artifacts
from ml.predict import predict_lithium_ion


def _first_training_row(artifacts):
    frame = load_clean_dataset()
    battery_id = artifacts.metadata["split"]["train_batteries"][0]
    row = frame.loc[frame["battery_id"] == battery_id].iloc[0]
    return {column: float(row[column]) for column in artifacts.metadata["raw_input_columns"]}


def test_saved_models_exclude_leakage_and_share_feature_order():
    artifacts = load_artifacts(MODELS_DIR)
    features = artifacts.metadata["feature_columns"]
    assert artifacts.preprocessing.feature_columns_ == features
    assert artifacts.regression_model.named_steps["features"].feature_columns_ == features
    assert artifacts.classification_model.named_steps["features"].feature_columns_ == features
    for banned in LEAKAGE_COLUMNS + ["battery_id"]:
        assert banned not in features
    overlap = set(artifacts.metadata["split"]["train_batteries"]) & set(
        artifacts.metadata["split"]["test_batteries"]
    )
    assert not overlap


def test_predictions_are_finite_and_invalid_measurements_are_rejected():
    artifacts = load_artifacts(MODELS_DIR)
    result = predict_lithium_ion(_first_training_row(artifacts), artifacts=artifacts)
    assert result["predicted_condition"] in {"Good", "Fair", "Poor"}
    assert result["predicted_soh"] == result["predicted_soh"]
    assert "calibrated confidence" not in json.dumps(result).lower()
    broken = _first_training_row(artifacts)
    broken["voltage_max"] = 400.0
    broken["voltage_mean"] = 350.0
    with pytest.raises(InvalidMeasurementError):
        predict_lithium_ion(broken, artifacts=artifacts)


def test_missing_model_directory_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_artifacts(tmp_path)


def test_replacing_artifacts_changes_the_loaded_version(tmp_path):
    for name in (
        "regression_model.pkl",
        "classification_model.pkl",
        "preprocessing.pkl",
        "model_metadata.json",
        "evaluation_metrics.json",
    ):
        shutil.copy2(MODELS_DIR / name, tmp_path / name)
    metadata_path = tmp_path / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_version"] = "9.9.9-test"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    replaced = load_artifacts(tmp_path)
    assert replaced.metadata["model_version"] == "9.9.9-test"
    original = load_artifacts(MODELS_DIR)
    assert original.metadata["model_version"] != "9.9.9-test"
    frame = pd.DataFrame([_first_training_row(replaced)])
    predicted = replaced.regression_model.predict(frame)
    label = replaced.classification_model.predict(frame)
    assert len(predicted) == 1
    assert len(label) == 1

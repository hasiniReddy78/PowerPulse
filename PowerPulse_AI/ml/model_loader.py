"""Load saved artifacts. The application calls this instead of training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from ml.config import MODELS_DIR
from ml.errors import ModelNotFoundError

REQUIRED_FILES = (
    "regression_model.pkl",
    "classification_model.pkl",
    "preprocessing.pkl",
    "model_metadata.json",
    "evaluation_metrics.json",
)


@dataclass
class Artifacts:
    model_dir: Path
    regression_model: object
    classification_model: object
    preprocessing: object
    metadata: dict
    evaluation: dict


def load_artifacts(model_dir: Path | None = None) -> Artifacts:
    directory = Path(model_dir) if model_dir is not None else MODELS_DIR
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise ModelNotFoundError(
            "Saved model artifacts are missing: "
            + ", ".join(missing)
            + f". Train them with `python -m ml.train_models` in {directory}."
        )
    try:
        metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
        evaluation = json.loads((directory / "evaluation_metrics.json").read_text(encoding="utf-8"))
        regression_model = joblib.load(directory / "regression_model.pkl")
        classification_model = joblib.load(directory / "classification_model.pkl")
        preprocessing = joblib.load(directory / "preprocessing.pkl")
    except Exception as exc:  # noqa: BLE001 - surface a single load error to the UI
        raise ModelNotFoundError(f"Model artifacts in {directory} could not be read: {exc}") from exc

    return Artifacts(
        model_dir=directory,
        regression_model=regression_model,
        classification_model=classification_model,
        preprocessing=preprocessing,
        metadata=metadata,
        evaluation=evaluation,
    )

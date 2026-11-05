"""Streamlit cache around the on-disk artifacts.

The cache key is the metadata version and its modification time, so promoting
a replacement model is picked up on the next rerun.
"""

import json

import streamlit as st

from ml.config import MODELS_DIR
from ml.model_loader import load_artifacts


@st.cache_resource
def _load(version: str, mtime_ns: int):
    del version, mtime_ns
    return load_artifacts(MODELS_DIR)


def get_artifacts():
    metadata_path = MODELS_DIR / "model_metadata.json"
    if not metadata_path.is_file():
        return load_artifacts(MODELS_DIR)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return _load(str(metadata.get("model_version")), metadata_path.stat().st_mtime_ns)

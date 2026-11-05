"""POWERPULSE home page."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from ml.dataset import load_clean_dataset
from ml.errors import ModelNotFoundError
from ui.common import hero, note, prepare_page
from ui.models import get_artifacts

prepare_page("Overview")
hero(
    "PowerPulse: Lithium-Ion Battery Health Monitoring & Predictive Maintenance",
    "Estimate State of Health (SoH), classify degradation condition, identify anomalies, and project Remaining Useful Life (RUL).",
)

left, right = st.columns(2)
with left:
    st.subheader("Project Purpose")
    st.markdown(
        """
**PowerPulse** is an intelligent diagnostic and predictive maintenance platform built specifically for **lithium-ion battery systems**.

It enables engineers, fleet operators, and technicians to assess cell degradation, monitor operating parameters, classify health condition, and anticipate capacity fade before critical failure.
        """
    )
with right:
    st.subheader("Platform Focus & Scope")
    st.markdown(
        """
- **Battery Chemistry**: Lithium-ion (single-cell discharge characteristics).
- **Core ML Capabilities**: Continuous State of Health (SoH) regression & multi-class condition classification (`Good`, `Fair`, `Poor`).
- **Prognostics**: Historical degradation trend analysis & Remaining Useful Life (RUL) estimation.
- **Data Foundation**: Trained on cleaned NASA laboratory discharge data; architecturally extensible to electric vehicle BMS telemetry.
        """
    )

st.subheader("How Battery Health Assessment Works")
st.markdown(
    """
1. **Input Measurements**: Provide operating discharge parameters (temperatures, voltage profile, discharge current, duration, and cycle count).
2. **Physical Validation**: The system validates values against physical laws and training bounds, rejecting implausible inputs (e.g. inverted voltages or pack-level voltages > 5 V).
3. **ML Inference**: Pre-trained machine learning pipelines predict continuous State of Health (SoH %) and classify battery condition.
4. **Degradation & RUL**: When repeated assessments are stored for a battery identifier, the system evaluates capacity fade trajectory and projects remaining cycles to End-of-Life (70% SoH).
5. **Reporting**: Downloadable CSV and PDF reports document the complete assessment with actionable maintenance recommendations.

*Target Leakage Prevention*: Measured capacity (`capacity_ah`) is never requested as an input, as it directly defines the target SoH label.
    """
)

try:
    frame = load_clean_dataset()
    artifacts = get_artifacts()
    version = artifacts.metadata["model_version"]
    regression = artifacts.metadata["regression"]
    classification = artifacts.metadata["classification"]
    st.subheader("Active Model Benchmark & Baseline")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Laboratory Records", f"{len(frame):,}")
    metric_cols[1].metric("Unique Cells", f"{frame['battery_id'].nunique()}")
    metric_cols[2].metric("Holdout SoH MAE", f"{regression['test_metrics']['mae']:.2f} pp")
    metric_cols[3].metric("Classifier Accuracy", f"{classification['test_metrics']['accuracy']*100:.1f}%")
    note(
        f"Baseline model version {version} was trained on laboratory discharge cycles from the cleaned NASA 18650 dataset. "
        "These metrics represent performance on strictly held-out test batteries under controlled chamber conditions, "
        "not real-world electric vehicle pack trials."
    )
except (ModelNotFoundError, FileNotFoundError) as exc:
    note(str(exc))

st.subheader("Available Modules")
st.markdown(
    """
- **Dashboard**: Track overall assessment counts, condition distribution, active alerts, and laboratory degradation benchmarks.
- **Battery Assessment**: Run real-time SoH estimation, condition classification, anomaly detection, and RUL estimation.
- **Battery History**: Inspect longitudinal degradation curves, per-cell records, and prognostic trajectories.
- **Reports**: Generate and download structured CSV and PDF assessment reports.
    """
)

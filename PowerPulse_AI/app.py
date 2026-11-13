"""POWERPULSE home and authentication portal."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from ml.config import EOL_THRESHOLD
from ml.dataset import load_clean_dataset
from ml.errors import ModelNotFoundError
from ui.common import current_user, hero, note, prepare_page, render_auth_interface
from ui.models import get_artifacts

# Allow landing page view, but show clean auth portal if not logged in
prepare_page("Overview", require_login=False)

user = current_user()

hero(
    "PowerPulse: Lithium-Ion Battery Health Monitoring & Predictive Maintenance",
    "Estimate State of Health (SoH), classify degradation condition, enforce warranty policies, and project Remaining Useful Life (RUL).",
)

if not user:
    st.markdown(
        """
        <div style='background: #FFF0F3; border: 1.5px solid #F8BBD0; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem;'>
            <h3 style='color: #7B112D; margin: 0 0 0.3rem 0;'>Welcome to PowerPulse</h3>
            <p style='color: #4A0B1A; margin: 0;'>
                Please log in to your authorized Service Centre account or create a new account below to register battery warranties,
                conduct ML-based health assessments, inspect degradation trends, and export diagnostic reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_auth_interface()
    st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Project Purpose")
    st.markdown(
        """
**PowerPulse** is an intelligent diagnostic and predictive maintenance platform built specifically for **lithium-ion battery systems**.

It enables engineers, fleet operators, and service technicians to validate warranties, assess cell degradation, monitor operating parameters, classify health condition, and anticipate capacity fade before critical failure.
        """
    )
with right:
    st.subheader("Platform Focus & Scope")
    st.markdown(
        """
- **Battery Chemistry**: Lithium-ion (single-cell discharge characteristics).
- **Warranty Enforcement**: Real-time policy validation ensuring health assessments are performed exclusively on active warranties.
- **Core ML Capabilities**: Continuous State of Health (SoH) regression & multi-class condition classification (`Good`, `Fair`, `Poor`).
- **Prognostics & XAI**: Historical degradation trajectory analysis, RUL estimation, and SHAP explainability.
- **Data Foundation**: Trained on cleaned NASA laboratory discharge data; architecturally extensible to EV BMS telemetry.
        """
    )

st.subheader("How Battery Health Assessment Works")
st.markdown(
    f"""
1. **Warranty Validation**: The service centre enters or verifies the battery's warranty credentials (purchase date, duration). Only batteries with an **ACTIVE** warranty are authorized for ML assessment.
2. **Input Telemetry**: Enter compatible discharge parameters (voltages, temperatures, current, discharge duration, and cycle count).
3. **Physical Validation**: The system validates values against physical limits and training bounds, rejecting implausible inputs.
4. **ML Inference**: Pre-trained machine learning models estimate continuous State of Health (SoH %) and classify battery condition.
5. **Degradation & RUL**: Multi-cycle records project remaining cycles to the {int(EOL_THRESHOLD)}% End-of-Life threshold.
6. **Reporting**: Downloadable CSV and PDF reports document the complete assessment with actionable maintenance guidance.
    """
)

if user:
    st.markdown(
        f"""
        <div class='pp-ok'>
            <strong>Logged in as {user['username']}</strong>. You have full access to all diagnostic and assessment modules.
        </div>
        """,
        unsafe_allow_html=True,
    )
    cta_col1, cta_col2, cta_col3 = st.columns(3)
    with cta_col1:
        if st.button("📊 Open Dashboard", use_container_width=True, type="primary"):
            st.switch_page("pages/dashboard.py")
    with cta_col2:
        if st.button("⚡ Run Battery Assessment", use_container_width=True):
            st.switch_page("pages/battery_assessment.py")
    with cta_col3:
        if st.button("📈 View Battery History", use_container_width=True):
            st.switch_page("pages/history.py")

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

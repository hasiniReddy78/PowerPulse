"""Lithium-ion battery health assessment, predictive maintenance, and SHAP explainability."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from database.db import add_assessment, list_assessments
from ml.errors import InvalidMeasurementError, ModelNotFoundError
from ml.explainability import create_shap_waterfall_chart, explain_prediction
from ml.predict import assess, estimate_rul
from ui.common import current_user, hero, note, prepare_page
from ui.models import get_artifacts

prepare_page("Battery Assessment")
hero(
    "Lithium-Ion Battery Assessment & SHAP Explainability",
    "Estimate single-cell State of Health (SoH), classify condition, project RUL, and explain predictions using SHAP attribution.",
)

LABELS = {
    "ambient_temperature": "Ambient temperature (°C)",
    "temperature_mean": "Average battery temperature (°C)",
    "temperature_max": "Maximum battery temperature (°C)",
    "voltage_mean": "Average cell voltage (V)",
    "voltage_min": "Minimum cell voltage (V)",
    "voltage_max": "Maximum cell voltage (V)",
    "voltage_std": "Voltage standard deviation (V)",
    "current_mean": "Average current (A)",
    "current_std": "Current standard deviation (A)",
    "duration_seconds": "Discharge duration (seconds)",
    "discharge_cycle": "Discharge cycle number",
}

identifier = st.text_input(
    "Battery Identifier",
    max_chars=64,
    placeholder="e.g. CELL-101, PACK1-MODULE3-CELL4",
    help="Identifier for the cell or battery under inspection."
)

note(
    "The initial model was trained on laboratory discharge-cycle summaries from the cleaned NASA lithium-ion dataset. "
    "Enter cell-level measurements compatible with that domain: cell voltages in the 2.0–4.3 V range, discharge current "
    "(recorded as negative), and chamber/operating temperatures. Pack voltages (> 5 V) and uncalibrated readings are rejected. "
    "A prediction is an engineering estimate, not a guaranteed diagnostic."
)

try:
    artifacts = get_artifacts()
except ModelNotFoundError as exc:
    st.error(str(exc))
    st.stop()

metadata = artifacts.metadata
with st.expander("Model Architecture & Holdout Benchmark"):
    st.write(
        f"**Model version**: {metadata['model_version']} | "
        f"**Regressor**: {metadata['regression']['algorithm']} | "
        f"**Classifier**: {metadata['classification']['algorithm']}"
    )
    test = metadata["regression"]["test_metrics"]
    st.write(
        f"On 8 held-out laboratory test batteries, the regressor achieved **MAE {test['mae']:.2f} pp** "
        f"(RMSE {test['rmse']:.2f}, R² {test['r2']:.3f}). "
        f"The classifier achieved **{metadata['classification']['test_metrics']['accuracy']*100:.1f}% accuracy**."
    )
    st.write("**Training batteries**: " + ", ".join(metadata["split"]["train_batteries"]))
    st.write("**Held-out test batteries**: " + ", ".join(metadata["split"]["test_batteries"]))
    st.markdown("**Known Model Scope & Limitations:**")
    for item in metadata["limitations"]:
        st.markdown(f"- {item}")

entries = {}
columns = st.columns(2)
for index, name in enumerate(metadata["raw_input_columns"]):
    bounds = metadata["raw_feature_ranges_training"][name]
    with columns[index % 2]:
        entries[name] = st.number_input(
            LABELS.get(name, name),
            value=None,
            help=(
                f"Valid training range: {bounds['min']:.4g} to {bounds['max']:.4g}. "
                "Discharge current is stored negative; positive values will be converted."
                if name == "current_mean"
                else f"Valid training range: {bounds['min']:.4g} to {bounds['max']:.4g}."
            ),
        )

submitted = st.button("Estimate Battery Health & Compute SHAP Explanations", type="primary")
if submitted:
    if not identifier.strip():
        st.error("Please enter a battery identifier before running the assessment.")
    else:
        try:
            result = assess(entries)
            shap_info = explain_prediction(result["inputs_used"], artifacts=artifacts)
        except InvalidMeasurementError as exc:
            for message in exc.errors:
                st.error(message)
        except Exception as exc:
            st.error(f"Assessment error: {exc}")
        else:
            user = current_user()
            user_id = user["id"] if user else None
            # Check historical records for this battery to evaluate RUL
            all_history = list_assessments(user_id)
            battery_history = [
                h for h in all_history
                if h["battery_identifier"].strip().lower() == identifier.strip().lower()
                and h["status"] == "predicted"
            ]
            rul_info = estimate_rul(
                battery_history,
                current_soh=result["predicted_soh"],
                current_cycle=result["inputs_used"].get("discharge_cycle"),
            )

            st.session_state["pending_assessment"] = {
                "battery_identifier": identifier.strip(),
                "category": "lithium_ion",
                "status": "predicted",
                "model_version": result["model_version"],
                "predicted_soh": result["predicted_soh"],
                "predicted_condition": result["predicted_condition"],
                "soh_band": result["soh_band"],
                "inputs": result["inputs_used"],
                "warnings": result["warnings"],
                "guidance": result["guidance"],
                "limitations": result["limitations"],
                "importance": result["feature_importance"],
                "test_mae": result["test_mae"],
                "test_rmse": result["test_rmse"],
                "regression_algorithm": result["regression_algorithm"],
                "classification_algorithm": result["classification_algorithm"],
                "rul": rul_info,
                "shap_explanation": shap_info,
            }

pending = st.session_state.get("pending_assessment")
if pending:
    st.subheader("Health Assessment Results")
    band = pending["predicted_condition"]
    css = "pp-ok" if band == "Good" else "pp-note" if band == "Fair" else "pp-bad"
    st.markdown(
        f"<div class='{css}'>"
        f"<h3>Estimated State of Health: {pending['predicted_soh']:.1f}%</h3>"
        f"<strong>Condition Classification</strong>: {band} (Implied Band: {pending['soh_band']})<br>"
        "<em>This is a machine learning prediction based on laboratory discharge features, not a physical teardown diagnosis.</em>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(max(pending["predicted_soh"] / 100.0, 0.0), 1.0))

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.caption(
            f"**Model**: {pending['regression_algorithm']} regressor + {pending['classification_algorithm']} classifier (v{pending['model_version']})."
        )
    with col_meta2:
        st.caption(
            f"**Holdout Benchmark MAE**: {pending['test_mae']:.2f} pp (RMSE {pending['test_rmse']:.2f} pp)."
        )

    # Remaining Useful Life (RUL) Section
    rul = pending.get("rul") or {}
    if rul.get("available"):
        st.markdown(
            f"<div class='pp-note'>"
            f"<strong>Remaining Useful Life (RUL) Projection:</strong><br>"
            f"Based on {rul['data_points']} recorded cycles, the estimated degradation rate is "
            f"<strong>{rul['degradation_rate_per_cycle']:.3f}% SoH per cycle</strong>. "
            f"Projected remaining life before reaching 70% End-of-Life: "
            f"<strong>~{rul['remaining_cycles']} cycles</strong> (Cycle {rul['projected_eol_cycle']})."
            f"</div>",
            unsafe_allow_html=True,
        )
    elif "reason" in rul:
        st.info(f"**RUL Prognostics**: {rul['reason']}")

    # SHAP Explainability Section
    shap_info = pending.get("shap_explanation")
    if shap_info:
        st.markdown("---")
        st.subheader("SHAP Feature Attribution & Model Explainability")
        st.caption(
            "Shapley Additive exPlanations (SHAP) allocate the exact contribution of each telemetry parameter "
            f"relative to the dataset baseline SoH ({shap_info['base_value']:.1f}%). Green bars indicate features "
            "that boosted the health estimate; red bars indicate degradation drivers that reduced estimated health."
        )

        fig_shap = create_shap_waterfall_chart(shap_info)
        st.plotly_chart(fig_shap, use_container_width=True)

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("**Top Positive Health Drivers**")
            pos_drivers = shap_info.get("top_positive_drivers", [])
            if pos_drivers:
                for d in pos_drivers:
                    st.markdown(f"- **{d['feature']}** ({d['input_value']:.3g}): **+{d['shap_value']:.2f} pp**")
                    st.caption(f"  {d['description']}")
            else:
                st.caption("No positive health drivers observed for this discharge record.")

        with col_neg:
            st.markdown("**Top Degradation Drivers**")
            neg_drivers = shap_info.get("top_negative_drivers", [])
            if neg_drivers:
                for d in neg_drivers:
                    st.markdown(f"- **{d['feature']}** ({d['input_value']:.3g}): **{d['shap_value']:.2f} pp**")
                    st.caption(f"  {d['description']}")
            else:
                st.caption("No negative degradation drivers observed for this discharge record.")

    if pending["warnings"]:
        st.markdown("**Alerts & Observations**")
        for message in pending["warnings"]:
            st.warning(message)

    if pending["guidance"]:
        st.markdown("**Recommended Maintenance Guidance**")
        for message in pending["guidance"]:
            st.write(f"• {message}")

    if st.button("Save Assessment to Database", type="secondary"):
        user = current_user()
        add_assessment({**pending, "user_id": user["id"] if user else None})
        st.success(f"Assessment for {pending['battery_identifier']} saved. View in History or export from Reports.")
        st.session_state.pop("pending_assessment", None)

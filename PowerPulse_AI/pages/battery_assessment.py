"""Lithium-ion battery health assessment, warranty policy enforcement, and SHAP explainability."""

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from database.db import (
    add_assessment,
    get_battery_warranty,
    list_assessments,
    list_user_batteries,
    register_battery_warranty,
)
from ml.config import EOL_THRESHOLD
from ml.errors import (
    InvalidMeasurementError,
    ModelNotFoundError,
    WarrantyError,
    WarrantyExpiredError,
    WarrantyNotRegisteredError,
    WarrantyValidationError,
)
from ml.explainability import create_shap_waterfall_chart, explain_prediction
from ml.predict import assess, estimate_rul
from ui.common import current_user, hero, note, prepare_page
from ui.models import get_artifacts
from utils.warranty import check_warranty_validity, parse_date

prepare_page("Battery Assessment", require_login=True)

user = current_user()
user_id = user["id"] if user else None

hero(
    "Battery Assessment & Warranty Policy Enforcement",
    "Validate warranty status before executing ML health estimation, condition classification, and SHAP feature attribution.",
)

st.markdown("### 1. Battery Warranty Verification")
st.caption(
    "In accordance with service centre policy, machine learning health assessments are strictly available "
    "for batteries with an **ACTIVE** warranty. Enter or select the battery warranty credentials below."
)

# Fetch batteries registered by this service centre
registered_batteries = list_user_batteries(user_id)
battery_options = ["+ Register New Battery"] + [b["battery_id"] for b in registered_batteries]

selected_option = st.selectbox(
    "Select Registered Battery or Register New",
    battery_options,
    key="select_battery_warranty",
    help="Select an existing registered battery from your service centre or register a new unit.",
)

# Populate form values based on selection
default_id = ""
default_mfg = ""
default_model = ""
default_purchase = date.today()
default_years = 3
default_months = 0

if selected_option != "+ Register New Battery":
    found = next((b for b in registered_batteries if b["battery_id"] == selected_option), None)
    if found:
        default_id = found["battery_id"]
        default_mfg = found.get("manufacturer") or ""
        default_model = found.get("battery_model") or ""
        parsed_p = parse_date(found.get("purchase_date"))
        if parsed_p:
            default_purchase = parsed_p
        default_years = int(found.get("warranty_years") or 0)
        default_months = int(found.get("warranty_months") or 0)

with st.expander("📝 Battery Warranty Registration & Details", expanded=(selected_option == "+ Register New Battery")):
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        inp_id = st.text_input("Battery ID *", value=default_id, placeholder="e.g. BAT-001, CELL-A9", key="inp_w_id")
        inp_purchase = st.date_input("Purchase Date *", value=default_purchase, max_value=date.today(), key="inp_w_date")
    with w_col2:
        inp_mfg = st.text_input("Manufacturer *", value=default_mfg, placeholder="e.g. Panasonic, CATL, LG", key="inp_w_mfg")
        inp_years = st.number_input("Warranty Duration (Years) *", min_value=0, max_value=20, value=default_years, key="inp_w_years")
    with w_col3:
        inp_model = st.text_input("Battery Model *", value=default_model, placeholder="e.g. Model-X, 18650-NCR", key="inp_w_model")
        inp_months = st.number_input("Additional Months", min_value=0, max_value=11, value=default_months, key="inp_w_months")

    if st.button("💾 Save & Validate Battery Warranty", type="primary", key="btn_save_warranty"):
        try:
            saved_battery = register_battery_warranty(
                battery_id=inp_id,
                manufacturer=inp_mfg,
                battery_model=inp_model,
                purchase_date=inp_purchase,
                warranty_years=inp_years,
                warranty_months=inp_months,
                user_id=user_id,
            )
            st.success(f"Warranty credentials for battery '{inp_id}' saved successfully.")
            st.session_state["active_battery_id"] = inp_id
            st.rerun()
        except WarrantyValidationError as exc:
            st.error(f"Warranty validation error: {exc}")
        except Exception as exc:
            st.error(f"Error saving warranty: {exc}")

active_battery_id = st.session_state.get("active_battery_id") or (selected_option if selected_option != "+ Register New Battery" else default_id)
active_battery = get_battery_warranty(active_battery_id, user_id=user_id) if active_battery_id else None
warranty_check = check_warranty_validity(active_battery)

# Display real-time Warranty Status
if not active_battery:
    st.info("ℹ️ Please enter and save the battery warranty information above to proceed with health assessment.")
    st.stop()

if warranty_check["status"] == "ACTIVE":
    st.markdown(
        f"""
        <div class='pp-ok'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span class='pp-badge-active'>WARRANTY ACTIVE</span>
                <span style='color: #155724; font-size: 0.9rem; font-weight: 600;'>Expires: {warranty_check['expiry_date']}</span>
            </div>
            <h4 style='color: #155724; margin: 0.4rem 0 0.2rem 0;'>Battery: {active_battery['battery_id']}</h4>
            <p style='margin: 0; font-size: 0.93rem;'>
                <strong>Manufacturer</strong>: {active_battery.get('manufacturer', 'Generic')} |
                <strong>Model</strong>: {active_battery.get('battery_model', 'Standard')} |
                <strong>Purchase Date</strong>: {active_battery.get('purchase_date')}<br/>
                <strong>Remaining Coverage</strong>: {warranty_check['days_remaining']} days remaining.
                <em>Battery health assessment is authorized under the current warranty policy.</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif warranty_check["status"] == "EXPIRED":
    st.markdown(
        f"""
        <div class='pp-bad'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span class='pp-badge-expired'>WARRANTY EXPIRED</span>
                <span style='color: #721C24; font-size: 0.9rem; font-weight: 600;'>Expired: {warranty_check['expiry_date']}</span>
            </div>
            <h3 style='color: #721C24; margin: 0.4rem 0 0.2rem 0;'>WARRANTY EXPIRED – Battery health assessment is unavailable under the current warranty policy.</h3>
            <p style='margin: 0; font-size: 0.95rem;'>
                This battery's warranty expired on <strong>{warranty_check['expiry_date']}</strong>
                ({abs(warranty_check['days_remaining'])} days ago).<br/>
                <strong>Battery ID</strong>: {active_battery['battery_id']} |
                <strong>Manufacturer</strong>: {active_battery.get('manufacturer')} |
                <strong>Model</strong>: {active_battery.get('battery_model')}<br/>
                <em>In accordance with policy, machine learning models (Random Forest, XGBoost), SHAP explainability, and RUL prognostics are locked.
                Please contact the manufacturer or service provider to review service options.</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # HARD BLOCK: Stop execution here. Do NOT display telemetry fields or run ML models!
    st.stop()
else:
    st.warning(f"Warranty status: {warranty_check['message']}")
    st.stop()

# ---------------------------------------------------------------------------
# 2. Battery Telemetry & Machine Learning Assessment (ONLY ACCESSIBLE IF ACTIVE)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 2. Battery Telemetry & ML Assessment")
note(
    "Enter laboratory or single-cell discharge cycle telemetry. Pack-level voltages (> 5 V) are rejected. "
    "A prediction is an engineering estimate based on calibrated electro-thermal parameters, not a teardown diagnostic."
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

try:
    artifacts = get_artifacts()
except ModelNotFoundError as exc:
    st.error(str(exc))
    st.stop()

metadata = artifacts.metadata
entries = {}
columns = st.columns(2)
for index, name in enumerate(metadata["raw_input_columns"]):
    bounds = metadata["raw_feature_ranges_training"][name]
    with columns[index % 2]:
        entries[name] = st.number_input(
            LABELS.get(name, name),
            value=None,
            help=f"Valid training range: {bounds['min']:.4g} to {bounds['max']:.4g}.",
            key=f"telem_{name}",
        )

submitted = st.button("⚡ Estimate Battery Health & Compute SHAP Explanations", type="primary", key="btn_run_ml")
if submitted:
    try:
        # Backend Warranty Validation: Enforced in assess() before any ML execution!
        result = assess(
            measurements=entries,
            battery_id=active_battery["battery_id"],
            user_id=user_id,
        )
        shap_info = explain_prediction(result["inputs_used"], artifacts=artifacts)
    except WarrantyExpiredError as exc:
        st.error(f"🚨 {exc}")
    except WarrantyNotRegisteredError as exc:
        st.error(f"⚠️ {exc}")
    except InvalidMeasurementError as exc:
        for message in exc.errors:
            st.error(message)
    except Exception as exc:
        st.error(f"Assessment error: {exc}")
    else:
        # Check historical records for this battery to evaluate RUL
        all_history = list_assessments(user_id)
        battery_history = [
            h for h in all_history
            if h["battery_identifier"].strip().lower() == active_battery["battery_id"].strip().lower()
            and h["status"] == "predicted"
        ]
        rul_info = estimate_rul(
            battery_history,
            current_soh=result["predicted_soh"],
            current_cycle=result["inputs_used"].get("discharge_cycle"),
        )

        st.session_state["pending_assessment"] = {
            "battery_identifier": active_battery["battery_id"],
            "battery_id": active_battery["battery_id"],
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
            "manufacturer": active_battery.get("manufacturer"),
            "battery_model": active_battery.get("battery_model"),
            "purchase_date": active_battery.get("purchase_date"),
            "warranty_years": active_battery.get("warranty_years"),
            "warranty_months": active_battery.get("warranty_months"),
        }

pending = st.session_state.get("pending_assessment")
if pending and pending.get("battery_identifier") == active_battery["battery_id"]:
    st.markdown("---")
    st.subheader(f"Health Assessment Results: {pending['battery_identifier']}")
    band = pending["predicted_condition"]
    css = "pp-ok" if band == "Good" else "pp-note" if band == "Fair" else "pp-bad"
    st.markdown(
        f"<div class='{css}'>"
        f"<h3 style='margin: 0;'>Estimated State of Health: {pending['predicted_soh']:.1f}%</h3>"
        f"<strong>Condition Classification</strong>: {band} (Implied Band: {pending['soh_band']})<br/>"
        "<em>Machine learning prediction derived from calibrated Random Forest and XGBoost ensembles.</em>"
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
    eol_val = int(rul.get("eol_threshold_soh", EOL_THRESHOLD))
    if rul.get("available"):
        st.markdown(
            f"<div class='pp-note'>"
            f"<strong>Remaining Useful Life (RUL) Projection:</strong><br/>"
            f"Based on {rul['data_points']} recorded cycles, the estimated degradation rate is "
            f"<strong>{rul['degradation_rate_per_cycle']:.3f}% SoH per cycle</strong>.<br/>"
            f"Projected remaining life before reaching {eol_val}% End-of-Life: "
            f"<strong>~{rul['remaining_cycles']} cycles</strong> (Cycle {rul['projected_eol_cycle']}).<br/>"
            f"<small><em>{rul.get('disclaimer', 'RUL is an estimate based on the observed degradation trend, not a guaranteed battery lifetime.')}</em></small><br/>"
            f"<span style='font-size: 0.85rem; color: #7B112D; font-weight: 600;'>Defined End-of-Life Threshold: {eol_val}% SoH</span>"
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
            "Shapley Additive exPlanations (SHAP) decompose the exact contribution of each telemetry parameter "
            f"relative to the dataset baseline SoH ({shap_info['base_value']:.1f}%). Green bars indicate features "
            "that supported battery health; red bars indicate degradation drivers that dragged estimated health down."
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

    if st.button("💾 Save Assessment Record to Database", type="secondary", key="btn_save_assessment"):
        add_assessment({**pending, "user_id": user_id})
        st.success(f"Assessment for battery '{pending['battery_identifier']}' saved. View in History or export from Reports.")
        st.session_state.pop("pending_assessment", None)

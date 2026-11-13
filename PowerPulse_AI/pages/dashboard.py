"""Laboratory reference charts, saved assessments, warranty tracking, and global SHAP insights."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from database.db import (
    acknowledge_battery_alert,
    dashboard_summary,
    get_lifecycle_monitoring_summary,
    list_assessments,
)
from ml.config import EOL_THRESHOLD
from ml.dataset import load_clean_dataset
from ml.explainability import create_global_importance_chart, get_global_feature_importance
from ui.common import current_user, hero, note, prepare_page

prepare_page("Dashboard", require_login=True)
hero("Dashboard", "Battery lifecycle monitoring, warranty tracking, predictive maintenance, and SHAP explainability.")

user = current_user()
user_id = user["id"] if user else None
summary = dashboard_summary(user_id)
lifecycle = get_lifecycle_monitoring_summary(user_id=user_id)
scope = f"Service Centre '{user['username']}'" if user else "guest records"

# ---------------------------------------------------------------------------
# 1. Battery Lifecycle Monitoring Section
# ---------------------------------------------------------------------------
st.subheader("🔋 Battery Lifecycle Monitoring")
st.caption(
    f"Active fleet status for {scope}. Battery ages and warranty countdowns are calculated dynamically against the current date."
)

# 5 Lifecycle Metrics
kpi_cols = st.columns(5)
kpi_cols[0].metric("Total Batteries", lifecycle["total_batteries"])
kpi_cols[1].metric("Active Warranties", lifecycle["active_warranties"])
kpi_cols[2].metric("Expiring within 90d", lifecycle["expiring_90_days"])
kpi_cols[3].metric("Expired Warranties", lifecycle["expired_warranties"])
kpi_cols[4].metric("Ageing Attention (≥24m)", lifecycle["ageing_attention_count"])

# Battery Information Table
st.markdown("#### Battery Fleet Lifecycle & Warranty Status")
if not lifecycle["table_rows"]:
    st.info("No batteries have been registered yet. Navigate to **Battery Assessment** to register a battery.")
else:
    # Filter columns for display table
    display_cols = [
        "Battery ID",
        "Manufacturer",
        "Purchase Date",
        "Battery Age",
        "Warranty Expiry",
        "Warranty Status",
        "Alert Status",
    ]
    df_table = pd.DataFrame(lifecycle["table_rows"])[display_cols]
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    st.caption(
        "**Status Legend:** "
        "🟢 **Normal / Active**: Warranty active & age < 24 months | "
        "🟡 **Expiring / Info**: Warranty expires within 90 days or age 24–36 months | "
        "🟠 **Ageing Warning**: In service 36–48 months | "
        "🔴 **Expired / Advanced**: Warranty expired or age > 48 months."
    )

# ---------------------------------------------------------------------------
# 2. Smart Alerts Section (Warranty & Ageing Reminders)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔔 Smart Lifecycle Alerts & Notifications")
st.caption("Real-time alerts stored in MongoDB. Acknowledging an alert marks it as reviewed in the database.")

unread = lifecycle["unread_alerts"]
if not unread:
    st.markdown(
        """
        <div class='pp-ok'>
            <strong>All Clear</strong>: No active unread warranty or ageing alerts for your registered batteries.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(f"**Unread Alerts ({len(unread)}):**")
    for a in unread:
        b_id = a["battery_id"]
        a_type = a["alert_type"].replace("_", " ").title()
        msg = a["alert_message"]
        gen_time = a.get("generated_at")[:19].replace("T", " ") if a.get("generated_at") else ""
        sev = a.get("severity", "yellow")
        css_box = "pp-bad" if sev in ("red", "orange") else "pp-note"

        st.markdown(
            f"""
            <div class='{css_box}' style='margin-bottom: 0.5rem;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <strong>Battery: {b_id} &nbsp;|&nbsp; {a_type}</strong>
                    <span style='font-size: 0.8rem; color: #6C757D;'>{gen_time} UTC</span>
                </div>
                <p style='margin: 0.3rem 0 0 0; font-size: 0.95rem;'>{msg}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ack_col1, ack_col2 = st.columns([6, 1])
        with ack_col2:
            if st.button("Acknowledge", key=f"btn_ack_{a['id']}", use_container_width=True):
                acknowledge_battery_alert(a["id"], user_id=user_id)
                st.success(f"Alert acknowledged for battery {b_id}.")
                st.rerun()

# Acknowledged alerts audit log
if lifecycle["acknowledged_alerts"]:
    with st.expander(f"📋 Acknowledged Alerts History ({len(lifecycle['acknowledged_alerts'])})"):
        ack_data = [
            {
                "Battery ID": a["battery_id"],
                "Alert Type": a["alert_type"],
                "Message": a["alert_message"],
                "Generated At": a.get("generated_at", "")[:19].replace("T", " "),
                "Acknowledged At": (a.get("acknowledged_at") or "")[:19].replace("T", " "),
                "Status": a["alert_status"],
            }
            for a in lifecycle["acknowledged_alerts"]
        ]
        st.dataframe(pd.DataFrame(ack_data), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 3. ML Assessment Analytics & Health Degradation
# ---------------------------------------------------------------------------
st.markdown("---")
left, right = st.columns(2)
with left:
    st.subheader("Condition Distribution")
    if summary["predicted_count"] == 0:
        st.info("No assessments recorded yet. Condition counts will appear once health assessments are saved.")
    else:
        condition_frame = pd.DataFrame(
            {"condition": list(summary["conditions"]), "assessments": list(summary["conditions"].values())}
        )
        st.plotly_chart(
            px.bar(
                condition_frame,
                x="condition",
                y="assessments",
                color="condition",
                color_discrete_map={"Good": "#1E7E34", "Fair": "#D39E00", "Poor": "#BD2130"},
                title="Condition Distribution of Evaluated Batteries",
            ),
            use_container_width=True,
        )
with right:
    st.subheader("Critical Alerts & Degraded Cells")
    if not summary["alerts"]:
        st.info(f"No Poor condition or sub-{int(EOL_THRESHOLD)}% SoH batteries detected in recent assessments.")
    else:
        for alert in summary["alerts"]:
            st.markdown(
                f"<div class='pp-bad'><strong>{alert['battery_identifier']}</strong> · "
                f"{alert['predicted_condition']} · {alert['predicted_soh']:.1f}% SoH · {alert['assessed_at']}</div>",
                unsafe_allow_html=True,
            )

st.subheader("Recent Health Assessments")
if not summary["recent"]:
    st.info("No health assessments have been saved yet.")
else:
    recent = pd.DataFrame(
        [
            {
                "Date (UTC)": row["assessed_at"],
                "Battery ID": row["battery_identifier"],
                "Category": "Lithium-Ion",
                "Status": row["status"],
                "Predicted SoH": None if row["predicted_soh"] is None else f"{row['predicted_soh']:.2f}%",
                "Condition": row["predicted_condition"],
                "Warranty Status": row.get("warranty_status") or "ACTIVE",
            }
            for row in summary["recent"]
        ]
    )
    st.dataframe(recent, use_container_width=True, hide_index=True)

st.subheader("Fleet Degradation Trajectories")
all_rows = [row for row in list_assessments(user_id) if row["status"] == "predicted"]
if len(all_rows) < 2:
    st.info("At least two saved lithium-ion assessments are required to render historical fleet degradation trajectories.")
else:
    trend = pd.DataFrame(
        [
            {
                "assessed_at": row["assessed_at"],
                "battery": row["battery_identifier"],
                "predicted_soh": row["predicted_soh"],
                "discharge_cycle": row["inputs"].get("discharge_cycle"),
            }
            for row in all_rows
        ]
    )
    st.plotly_chart(
        px.line(
            trend.sort_values("assessed_at"),
            x="assessed_at",
            y="predicted_soh",
            color="battery",
            markers=True,
            title="Saved SoH Estimates Over Time",
            labels={"predicted_soh": "Predicted SoH (%)", "assessed_at": "Assessment Time (UTC)"},
        ),
        use_container_width=True,
    )

st.divider()
st.subheader("Global SHAP Model Explainability")
st.caption(
    "Shapley Additive exPlanations (SHAP) across representative cycles quantify which telemetry parameters "
    "most heavily drive State of Health estimations across the lithium-ion fleet."
)
with st.spinner("Loading global SHAP feature importance..."):
    global_shap = get_global_feature_importance(sample_size=100)
    fig_global = create_global_importance_chart(global_shap)
    st.plotly_chart(fig_global, use_container_width=True)

st.divider()
st.subheader("Laboratory Reference Baseline")
note(
    "The charts below display the cleaned NASA lithium-ion baseline dataset used to calibrate the machine learning models. "
    "These single-cell laboratory cycles are kept distinct from your recorded fleet assessments."
)
lab = load_clean_dataset()
lab_left, lab_right = st.columns(2)
with lab_left:
    st.plotly_chart(
        px.histogram(lab, x="soh_percent", nbins=40, title="Laboratory SoH Distribution", labels={"soh_percent": "SoH (%)"}),
        use_container_width=True,
    )
with lab_right:
    counts = lab["condition"].value_counts().rename_axis("condition").reset_index(name="records")
    st.plotly_chart(px.bar(counts, x="condition", y="records", title="Laboratory Condition Counts"), use_container_width=True)

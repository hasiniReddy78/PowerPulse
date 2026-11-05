"""Stored lithium-ion battery assessments and historical degradation tracking."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from database.db import list_assessments
from ml.predict import estimate_rul
from ui.common import current_user, hero, prepare_page

prepare_page("Battery History")
hero("Battery History", "Review stored lithium-ion assessments, degradation trends, and RUL projections.")

user = current_user()
user_id = user["id"] if user else None
rows = list_assessments(user_id)
if user:
    st.caption(f"Showing assessments for logged-in user: **{user['username']}**.")
else:
    st.caption("Showing local guest assessments stored on this system.")

if not rows:
    st.info("No assessments have been recorded yet. Perform an assessment to start tracking battery degradation.")
    st.stop()

table_data = []
for row in rows:
    cycle = row["inputs"].get("discharge_cycle")
    table_data.append(
        {
            "ID": row["id"],
            "Date (UTC)": row["assessed_at"],
            "Battery": row["battery_identifier"],
            "Cycle": cycle if cycle is not None else "—",
            "Predicted SoH": f"{row['predicted_soh']:.2f}%" if row["predicted_soh"] is not None else "—",
            "Condition": row["predicted_condition"] or "—",
            "Model": row["model_version"] or "—",
        }
    )

table = pd.DataFrame(table_data)
st.dataframe(table, use_container_width=True, hide_index=True)

# Battery-specific degradation tracking and RUL
unique_batteries = sorted(list({row["battery_identifier"] for row in rows}))
selected_battery = st.selectbox("Inspect Battery Degradation & History", unique_batteries)

battery_rows = [r for r in rows if r["battery_identifier"] == selected_battery]
st.subheader(f"History: {selected_battery}")

if len(battery_rows) >= 2:
    trend_data = []
    for r in battery_rows:
        if r["predicted_soh"] is not None:
            trend_data.append(
                {
                    "Date": r["assessed_at"],
                    "Cycle": r["inputs"].get("discharge_cycle"),
                    "Predicted SoH (%)": r["predicted_soh"],
                }
            )
    trend_df = pd.DataFrame(trend_data)
    has_cycles = trend_df["Cycle"].notna().sum() >= 2
    x_axis = "Cycle" if has_cycles else "Date"

    fig = px.line(
        trend_df.sort_values(x_axis),
        x=x_axis,
        y="Predicted SoH (%)",
        markers=True,
        title=f"Degradation Trajectory for {selected_battery}",
    )
    fig.add_hline(y=70.0, line_dash="dash", line_color="red", annotation_text="EoL Threshold (70% SoH)")
    fig.add_hline(y=85.0, line_dash="dash", line_color="orange", annotation_text="Fair Threshold (85% SoH)")
    st.plotly_chart(fig, use_container_width=True)

    # Calculate RUL
    rul_analysis = estimate_rul(battery_rows)
    if rul_analysis.get("available"):
        st.markdown(
            f"<div class='pp-note'>"
            f"<strong>Remaining Useful Life (RUL) Analysis:</strong><br>"
            f"Based on {rul_analysis['data_points']} recorded data points, the degradation rate is "
            f"<strong>{rul_analysis['degradation_rate_per_cycle']:.3f}% SoH per cycle</strong>.<br>"
            f"Estimated remaining cycles before reaching 70% End-of-Life: "
            f"<strong>~{rul_analysis['remaining_cycles']} cycles</strong> (Projected Cycle {rul_analysis['projected_eol_cycle']})."
            f"</div>",
            unsafe_allow_html=True,
        )
    elif "reason" in rul_analysis:
        st.info(f"**RUL Prognostics**: {rul_analysis['reason']}")
else:
    st.info("Additional assessments are needed for this battery identifier to render a multi-point degradation trajectory.")

# Detailed view of individual assessment
st.markdown("---")
st.subheader("Assessment Details")
selected_id = st.selectbox(
    "Select assessment ID to inspect inputs",
    [r["id"] for r in battery_rows],
    format_func=lambda x: f"Record #{x} - {next(r['assessed_at'] for r in battery_rows if r['id'] == x)}"
)
record = next(r for r in battery_rows if r["id"] == selected_id)

col_left, col_right = st.columns(2)
with col_left:
    st.write(f"**Date (UTC)**: {record['assessed_at']}")
    st.write(f"**Predicted SoH**: {record['predicted_soh']:.2f}%")
    st.write(f"**Condition**: {record['predicted_condition']}")
    st.write(f"**SoH Band**: {record['soh_band']}")
    st.write(f"**Model Version**: {record['model_version']}")
with col_right:
    st.markdown("**Recorded Measurements**")
    st.json(record["inputs"])

if record["warnings"]:
    st.markdown("**Warnings & Diagnostics**")
    for warning in record["warnings"]:
        st.warning(warning)

if record["limitations"]:
    with st.expander("Model Limitations"):
        for item in record["limitations"]:
            st.markdown(f"- {item}")

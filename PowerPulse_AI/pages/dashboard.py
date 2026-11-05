"""Laboratory reference charts, saved assessments, and global SHAP insights."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from database.db import dashboard_summary, list_assessments
from ml.dataset import load_clean_dataset
from ml.explainability import create_global_importance_chart, get_global_feature_importance
from ui.common import current_user, hero, note, prepare_page

prepare_page("Dashboard")
hero("Dashboard", "Saved assessments, battery health trends, and global SHAP feature explainability.")

user = current_user()
user_id = user["id"] if user else None
summary = dashboard_summary(user_id)
scope = f"account {user['username']}" if user else "guest records on this computer"

columns = st.columns(4)
columns[0].metric("Registered batteries", summary["battery_count"])
columns[1].metric("Assessments", summary["assessment_count"])
if summary["average_predicted_soh"] is None:
    columns[2].metric("Average predicted SoH", "—")
else:
    columns[2].metric("Average predicted SoH", f"{summary['average_predicted_soh']:.1f}%")
columns[3].metric("Predicted records", summary["predicted_count"])
st.caption(f"Figures above count {scope}. Empty values stay empty when nothing has been saved.")

left, right = st.columns(2)
with left:
    st.subheader("Condition distribution")
    if summary["predicted_count"] == 0:
        st.info("No predicted assessments yet. Condition counts will appear after a lithium-ion prediction is saved.")
    else:
        condition_frame = pd.DataFrame(
            {"condition": list(summary["conditions"]), "assessments": list(summary["conditions"].values())}
        )
        st.plotly_chart(
            px.bar(condition_frame, x="condition", y="assessments", title="Saved lithium-ion predictions"),
            use_container_width=True,
        )
with right:
    st.subheader("Recent alerts")
    if not summary["alerts"]:
        st.info("No Poor or sub-70% predictions are stored in this view.")
    else:
        for alert in summary["alerts"]:
            st.markdown(
                f"<div class='pp-bad'><strong>{alert['battery_identifier']}</strong> · "
                f"{alert['predicted_condition']} · {alert['predicted_soh']:.1f}% SoH · {alert['assessed_at']}</div>",
                unsafe_allow_html=True,
            )

st.subheader("Recent assessments")
if not summary["recent"]:
    st.info("No assessments have been saved yet.")
else:
    recent = pd.DataFrame(
        [
            {
                "Date (UTC)": row["assessed_at"],
                "Battery": row["battery_identifier"],
                "Category": "Lithium-Ion",
                "Status": row["status"],
                "Predicted SoH": None if row["predicted_soh"] is None else round(row["predicted_soh"], 2),
                "Condition": row["predicted_condition"],
            }
            for row in summary["recent"]
        ]
    )
    st.dataframe(recent, use_container_width=True, hide_index=True)

st.subheader("Your degradation records")
all_rows = [row for row in list_assessments(user_id) if row["status"] == "predicted"]
if len(all_rows) < 2:
    st.info("At least two saved lithium-ion predictions are needed before a personal trend is drawn.")
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
            title="Saved SoH estimates over time",
            labels={"predicted_soh": "Predicted SoH (%)", "assessed_at": "Assessment time (UTC)"},
        ),
        use_container_width=True,
    )

st.divider()
st.subheader("Global SHAP Model Explainability")
st.caption(
    "Shapley Additive exPlanations (SHAP) across the dataset quantify which telemetry parameters "
    "most heavily drive State of Health estimations. Higher mean absolute SHAP indicates greater overall predictive impact."
)
with st.spinner("Loading global SHAP feature importance..."):
    global_shap = get_global_feature_importance(sample_size=100)
    fig_global = create_global_importance_chart(global_shap)
    st.plotly_chart(fig_global, use_container_width=True)

st.divider()
st.subheader("Laboratory Reference Dataset")
note(
    "The charts below are the cleaned NASA lithium-ion file used for training. "
    "They are single-cell chamber cycling data, kept separate from your recorded vehicle inspections."
)
lab = load_clean_dataset()
lab_left, lab_right = st.columns(2)
with lab_left:
    st.plotly_chart(
        px.histogram(lab, x="soh_percent", nbins=40, title="Laboratory SoH distribution", labels={"soh_percent": "SoH (%)"}),
        use_container_width=True,
    )
with lab_right:
    counts = lab["condition"].value_counts().rename_axis("condition").reset_index(name="records")
    st.plotly_chart(px.bar(counts, x="condition", y="records", title="Laboratory condition counts"), use_container_width=True)

battery_id = st.selectbox("Laboratory battery", sorted(lab["battery_id"].unique()))
subset = lab.loc[lab["battery_id"] == battery_id].sort_values("discharge_cycle")
st.plotly_chart(
    px.line(
        subset,
        x="discharge_cycle",
        y=["capacity_ah", "soh_percent"],
        title=f"{battery_id} laboratory capacity and SoH",
        labels={"discharge_cycle": "Discharge cycle", "value": "Value", "variable": "Series"},
    ),
    use_container_width=True,
)
st.caption(
    f"{battery_id}: {len(subset)} cycles, ambient {subset['ambient_temperature'].iloc[0]:.0f} °C, "
    f"SoH {subset['soh_percent'].min():.1f}% to {subset['soh_percent'].max():.1f}%."
)

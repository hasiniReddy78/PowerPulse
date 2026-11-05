"""Download a CSV or PDF for one stored assessment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from database.db import list_assessments
from services.reporting import assessment_csv, assessment_lines, assessment_pdf
from ui.common import current_user, hero, prepare_page

prepare_page("Reports")
hero("Reports", "Download the measurements, prediction, warnings, and model limitations for one assessment.")

user = current_user()
rows = list_assessments(user["id"] if user else None)
if not rows:
    st.info("Save an assessment before generating a report.")
    st.stop()

labels = {
    row["id"]: f"{row['id']} · {row['battery_identifier']} · {row['assessed_at']}"
    for row in rows
}
selected = st.selectbox("Assessment", list(labels), format_func=lambda key: labels[key])
record = next(row for row in rows if row["id"] == selected)
st.text("\n".join(assessment_lines(record)))
stem = f"powerpulse_{record['battery_identifier']}_{record['id']}"
csv_col, pdf_col = st.columns(2)
with csv_col:
    st.download_button(
        "Download CSV",
        data=assessment_csv(record),
        file_name=f"{stem}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with pdf_col:
    st.download_button(
        "Download PDF",
        data=assessment_pdf(record),
        file_name=f"{stem}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

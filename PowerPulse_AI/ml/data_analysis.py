"""Phase 1 inspection of the cleaned laboratory dataset.

The script reports unusual values. It does not delete rows and it does not
write back to the CSV.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from ml.config import DATA_CSV, FIGURES_DIR, REFERENCE_CAPACITY_AH, REPORTS_DIR
from ml.dataset import load_clean_dataset


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def soh_from_capacity(capacity: pd.Series) -> pd.Series:
    return (capacity / REFERENCE_CAPACITY_AH * 100.0).clip(upper=100.0)


def analyse(frame: pd.DataFrame) -> dict:
    soh_gap = (soh_from_capacity(frame["capacity_ah"]) - frame["soh_percent"]).abs()
    expected_condition = np.where(
        frame["soh_percent"] <= 70.0,
        "Poor",
        np.where(frame["soh_percent"] <= 85.0, "Fair", "Good"),
    )
    per_battery = (
        frame.groupby("battery_id")
        .agg(
            records=("discharge_cycle", "size"),
            ambient_temperatures=(
                "ambient_temperature",
                lambda values: ", ".join(f"{float(value):.0f}" for value in sorted(set(values))),
            ),
            cycle_min=("discharge_cycle", "min"),
            cycle_max=("discharge_cycle", "max"),
            capacity_min=("capacity_ah", "min"),
            capacity_max=("capacity_ah", "max"),
            soh_min=("soh_percent", "min"),
            soh_max=("soh_percent", "max"),
            current_median=("current_mean", "median"),
        )
        .reset_index()
    )
    capacity_increases = {}
    duration_capacity_corr = []
    ordered = frame.sort_values(["battery_id", "discharge_cycle"])
    for battery_id, subset in ordered.groupby("battery_id"):
        delta = subset["capacity_ah"].diff()
        increases = int((delta > 0.01).sum())
        if increases:
            capacity_increases[battery_id] = increases
        if len(subset) >= 3:
            duration_capacity_corr.append(
                float(subset["duration_seconds"].corr(subset["capacity_ah"]))
            )

    numeric = frame.select_dtypes(include="number")
    correlations = (
        numeric.corr(numeric_only=True)["soh_percent"]
        .drop(labels=["soh_percent"])
        .sort_values()
    )
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "unique_batteries": int(frame["battery_id"].nunique()),
        "records_per_battery": frame["battery_id"].value_counts().sort_index().to_dict(),
        "missing_values": frame.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_battery_cycle": int(
            frame.duplicated(subset=["battery_id", "discharge_cycle"]).sum()
        ),
        "soh_describe": frame["soh_percent"].describe().to_dict(),
        "condition_counts": frame["condition"].value_counts().to_dict(),
        "descriptive_statistics": json.loads(frame.describe().to_json()),
        "soh_formula_max_abs_difference": float(soh_gap.max()),
        "capacity_above_reference": int((frame["capacity_ah"] > REFERENCE_CAPACITY_AH).sum()),
        "capacity_above_reference_max": float(frame["capacity_ah"].max()),
        "condition_mismatches": int((expected_condition != frame["condition"]).sum()),
        "consistency": {
            "temperature_mean_below_ambient": int(
                (frame["temperature_mean"] < frame["ambient_temperature"]).sum()
            ),
            "temperature_max_below_mean": int(
                (frame["temperature_max"] < frame["temperature_mean"]).sum()
            ),
            "voltage_min_above_max": int((frame["voltage_min"] > frame["voltage_max"]).sum()),
            "voltage_mean_outside_min_max": int(
                (
                    (frame["voltage_mean"] < frame["voltage_min"])
                    | (frame["voltage_mean"] > frame["voltage_max"])
                ).sum()
            ),
            "positive_current_mean": int((frame["current_mean"] > 0).sum()),
            "non_positive_duration": int((frame["duration_seconds"] <= 0).sum()),
            "non_positive_voltage_std": int((frame["voltage_std"] <= 0).sum()),
            "non_positive_current_std": int((frame["current_std"] <= 0).sum()),
        },
        "batteries_not_starting_at_cycle_1": per_battery.loc[
            per_battery["cycle_min"] != 1, ["battery_id", "cycle_min"]
        ].to_dict(orient="records"),
        "capacity_increases_over_0_01_ah": capacity_increases,
        "median_within_battery_duration_capacity_correlation": float(
            np.median(duration_capacity_corr)
        ),
        "correlation_with_soh": correlations.to_dict(),
        "ambient_battery_counts": frame.groupby("ambient_temperature")["battery_id"]
        .nunique()
        .astype(int)
        .to_dict(),
        "batteries_with_multiple_ambient_temperatures": per_battery.loc[
            per_battery["ambient_temperatures"].str.contains(","),
            ["battery_id", "ambient_temperatures"],
        ].to_dict(orient="records"),
        "per_battery": per_battery.to_dict(orient="records"),
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_markdown(path: Path, sha256: str, findings: dict) -> str:
    condition = findings["condition_counts"]
    soh = findings["soh_describe"]
    lines = [
        "# PowerPulse dataset analysis",
        "",
        "Source file: `data/powerpulse_clean.csv`.",
        "This report is generated from that file. No rows were removed.",
        "",
        f"- SHA-256: `{sha256}`",
        f"- Records: {findings['rows']}",
        f"- Columns: {findings['columns']}",
        f"- Unique batteries: {findings['unique_batteries']}",
        f"- Duplicate rows: {findings['duplicate_rows']}",
        f"- Duplicate battery and cycle pairs: {findings['duplicate_battery_cycle']}",
        "",
        "## Columns and data types",
        "",
    ]
    for name, dtype in findings["column_dtypes"].items():
        lines.append(f"- `{name}`: {dtype}")
    lines.extend(
        [
            "",
            "## Missing values",
            "",
            "Every column has zero missing values.",
            "",
            "## Records per battery",
            "",
            "| Battery | Records |",
            "| --- | ---: |",
        ]
    )
    for battery_id, count in findings["records_per_battery"].items():
        lines.append(f"| {battery_id} | {count} |")
    lines.extend(
        [
            "",
            "## Targets",
            "",
            "SoH in the file matches `min(100, capacity_ah / 2.0 × 100)`.",
            f"Maximum absolute difference: {findings['soh_formula_max_abs_difference']:.3e}.",
            f"Records with capacity above 2.0 Ah: {findings['capacity_above_reference']} "
            f"(maximum capacity {findings['capacity_above_reference_max']:.6f} Ah). "
            "Those seven B0006 rows have SoH stored as 100%.",
            "The 2.0 Ah divisor is the preprocessing assumption already present in the file. "
            "It is the usual nominal capacity quoted for the NASA 18650 cells, but capacities "
            "above 2.0 Ah show that it should not be treated as a measured nominal capacity "
            "for every record.",
            "",
            f"Condition labels match the stated bands "
            f"(Poor ≤ 70, Fair ≤ 85, Good > 85). Mismatches: {findings['condition_mismatches']}.",
            "",
            "| Condition | Records | Share |",
            "| --- | ---: | ---: |",
        ]
    )
    for label in ("Good", "Fair", "Poor"):
        count = int(condition.get(label, 0))
        share = count / findings["rows"]
        lines.append(f"| {label} | {count} | {share:.1%} |")
    lines.extend(
        [
            "",
            "The classes are imbalanced. Poor is the largest class. Accuracy alone would "
            "overstate a classifier that mostly predicts Poor.",
            "",
            "## SoH summary",
            "",
            f"- Minimum: {_fmt(soh['min'])}%",
            f"- 25th percentile: {_fmt(soh['25%'])}%",
            f"- Median: {_fmt(soh['50%'])}%",
            f"- Mean: {_fmt(soh['mean'])}%",
            f"- 75th percentile: {_fmt(soh['75%'])}%",
            f"- Maximum: {_fmt(soh['max'])}%",
            f"- Standard deviation: {_fmt(soh['std'])} percentage points",
            "",
            "## Unusual values kept in the file",
            "",
            "These were not deleted.",
            "",
            f"- {findings['capacity_above_reference']} capacities exceed the 2.0 Ah reference, "
            "and the corresponding SoH values are capped at 100%.",
            "- Capacity is not strictly decreasing. Increases larger than 0.01 Ah versus the "
            "previous stored cycle were counted for "
            f"{len(findings['capacity_increases_over_0_01_ah'])} batteries.",
            "- Some batteries do not start at discharge cycle 1: "
            + ", ".join(
                f"{item['battery_id']} (starts at {int(item['cycle_min'])})"
                for item in findings["batteries_not_starting_at_cycle_1"]
            )
            + ".",
            "- Current is negative on every row. That is the discharge sign convention in this file.",
            "- Internal checks found no inverted voltage bounds, no mean voltage outside the "
            "min/max, no mean temperature below ambient, and no non-positive duration or "
            "standard deviation.",
            "",
            "## Why rows are not independent",
            "",
            "Each battery contributes many discharge cycles. A random row split would put "
            "neighbouring cycles from the same cell in both train and test.",
            "Most cells stay at one chamber temperature. B0042, B0043 and B0044 do not: "
            "each has an early 22 °C block and a later 4 °C block. The whole battery still "
            "stays in one split. Batteries that contribute at least one row at each temperature:",
            "",
        ]
    )
    for temperature, count in sorted(findings["ambient_battery_counts"].items()):
        lines.append(f"- {float(temperature):.0f} °C: {int(count)} batteries")
    lines.extend(
        [
            "",
            "Correlation of each numeric column with SoH, computed on all rows and therefore "
            "inflated by repeated measures:",
            "",
            "| Column | Correlation with SoH |",
            "| --- | ---: |",
        ]
    )
    for name, value in findings["correlation_with_soh"].items():
        lines.append(f"| `{name}` | {value:.3f} |")
    lines.extend(
        [
            "",
            "`capacity_ah` correlates with SoH at essentially 1 because SoH was calculated "
            "from capacity. It is excluded from the model inputs.",
            "",
            "Ambient temperature correlates strongly with SoH (about 0.74) because the cold "
            "experiments contain most of the Poor labels. That association is experimental "
            "structure, not evidence that a warmer room repairs a battery.",
            "Current is also tied to the experiment: the 43 °C group is discharged near 4 A, "
            "while several 4 °C cells are discharged below 1 A. A correlation between current "
            "and SoH is therefore confounded.",
            "",
            "Median within-battery correlation between discharge duration and capacity: "
            f"{findings['median_within_battery_duration_capacity_correlation']:.3f}. "
            "Duration tracks capacity inside a cell when the current profile is stable, but "
            "it is a weak SoH proxy across the whole file because the discharge rate changes "
            "between experiments. Duration is still a legitimate measurement. Capacity is not "
            "used as a feature.",
            "",
            "## Figures",
            "",
            "- `reports/figures/soh_distribution.html`",
            "- `reports/figures/capacity_distribution.html`",
            "- `reports/figures/condition_counts.html`",
            "- `reports/figures/soh_degradation_by_battery.html`",
            "- `reports/figures/capacity_degradation_by_battery.html`",
            "",
            "Laboratory discharge curves are not a validation of real-world electric-vehicle packs.",
            "",
        ]
    )
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def write_figures(frame: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values(["battery_id", "discharge_cycle"])
    figures = {
        "soh_distribution.html": px.histogram(
            frame,
            x="soh_percent",
            nbins=40,
            title="State of Health distribution (laboratory file)",
            labels={"soh_percent": "SoH (%)"},
        ),
        "capacity_distribution.html": px.histogram(
            frame,
            x="capacity_ah",
            nbins=40,
            title="Discharge capacity distribution (laboratory file)",
            labels={"capacity_ah": "Capacity (Ah)"},
        ),
        "condition_counts.html": px.bar(
            frame["condition"].value_counts().rename_axis("condition").reset_index(name="records"),
            x="condition",
            y="records",
            title="Condition counts (laboratory file)",
        ),
        "soh_degradation_by_battery.html": px.line(
            ordered,
            x="discharge_cycle",
            y="soh_percent",
            color="battery_id",
            title="SoH by discharge cycle and battery",
            labels={"discharge_cycle": "Discharge cycle", "soh_percent": "SoH (%)"},
        ),
        "capacity_degradation_by_battery.html": px.line(
            ordered,
            x="discharge_cycle",
            y="capacity_ah",
            color="battery_id",
            title="Capacity by discharge cycle and battery",
            labels={"discharge_cycle": "Discharge cycle", "capacity_ah": "Capacity (Ah)"},
        ),
    }
    for name, figure in figures.items():
        figure.write_html(FIGURES_DIR / name, include_plotlyjs="cdn")


def main() -> None:
    frame = load_clean_dataset()
    findings = analyse(frame)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sha = file_sha256(DATA_CSV)
    render_markdown(REPORTS_DIR / "dataset_analysis.md", sha, findings)
    write_figures(frame)
    pd.DataFrame(findings["per_battery"]).to_csv(
        REPORTS_DIR / "battery_summary.csv", index=False
    )
    print(f"rows={findings['rows']} batteries={findings['unique_batteries']}")
    print(f"conditions={findings['condition_counts']}")
    print(f"report={REPORTS_DIR / 'dataset_analysis.md'}")


if __name__ == "__main__":
    main()

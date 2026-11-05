"""CSV and plain-text PDF exports for stored lithium-ion assessments."""

from __future__ import annotations

import csv
import io
from textwrap import wrap


def assessment_lines(record: dict) -> list[str]:
    lines = [
        "POWERPULSE ASSESSMENT REPORT",
        "Intelligent Lithium-Ion Battery Health Monitoring & Predictive Maintenance",
        "=" * 72,
        "",
        f"Assessment Date (UTC): {record.get('assessed_at', '')}",
        f"Battery Identifier: {record.get('battery_identifier', '')}",
        "Battery Chemistry: Lithium-Ion",
        f"Record Status: {record.get('status', 'predicted')}",
        f"Model Version: {record.get('model_version') or '1.0.0'}",
        "",
        "ENTERED MEASUREMENTS",
        "-" * 40,
    ]
    inputs = record.get("inputs") or {}
    for key, value in inputs.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.extend(
        [
            "HEALTH ESTIMATION RESULTS",
            "-" * 40,
            f"Estimated State of Health (SoH): {record.get('predicted_soh')}%",
            f"Predicted Condition: {record.get('predicted_condition')}",
            f"Condition Band: {record.get('soh_band')}",
            "",
            "The SoH figure is a machine-learning estimate based on laboratory discharge features.",
            "It is not a direct capacity measurement or guaranteed physical diagnosis.",
            "",
        ]
    )

    # SHAP Explainability Section
    shap_info = record.get("shap_explanation")
    if not shap_info and inputs:
        try:
            from ml.explainability import explain_prediction
            shap_info = explain_prediction(inputs)
        except Exception:
            shap_info = None

    if shap_info:
        lines.extend(["SHAP EXPLAINABILITY ATTRIBUTION", "-" * 40])
        lines.append(f"  Dataset Baseline SoH: {shap_info['base_value']:.2f}%")
        lines.append("  Top Positive Health Contributors:")
        for d in shap_info.get("top_positive_drivers", []):
            lines.append(f"    + {d['feature']} ({d['input_value']:.3g}): +{d['shap_value']:.2f} pp")
        lines.append("  Top Degradation Drivers:")
        for d in shap_info.get("top_negative_drivers", []):
            lines.append(f"    - {d['feature']} ({d['input_value']:.3g}): {d['shap_value']:.2f} pp")
        lines.append("")

    lines.extend(["WARNINGS & ANOMALIES", "-" * 40])
    warnings = record.get("warnings") or []
    if warnings:
        lines.extend(f"  - {item}" for item in warnings)
    else:
        lines.append("  None recorded.")

    guidance = record.get("guidance") or []
    if guidance:
        lines.extend(["", "MAINTENANCE GUIDANCE", "-" * 40])
        lines.extend(f"  • {item}" for item in guidance)

    lines.extend(["", "MODEL LIMITATIONS", "-" * 40])
    limitations = record.get("limitations") or []
    if isinstance(limitations, str):
        limitations = [limitations] if limitations else []
    if limitations:
        lines.extend(f"  - {item}" for item in limitations)
    else:
        lines.append("  None recorded.")

    lines.extend(
        [
            "",
            "=" * 72,
            "Notice: Laboratory NASA test results do not validate real-world EV pack performance.",
        ]
    )
    return lines


def assessment_csv(record: dict) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["field", "value"])
    writer.writerow(["assessed_at_utc", record.get("assessed_at", "")])
    writer.writerow(["battery_identifier", record.get("battery_identifier", "")])
    writer.writerow(["category", "lithium_ion"])
    writer.writerow(["status", record.get("status", "predicted")])
    writer.writerow(["model_version", record.get("model_version") or ""])
    writer.writerow(["predicted_soh", "" if record.get("predicted_soh") is None else record["predicted_soh"]])
    writer.writerow(["predicted_condition", record.get("predicted_condition") or ""])
    writer.writerow(["soh_band", record.get("soh_band") or ""])
    inputs = record.get("inputs") or {}
    for key, value in inputs.items():
        writer.writerow([f"input.{key}", value])

    shap_info = record.get("shap_explanation")
    if not shap_info and inputs:
        try:
            from ml.explainability import explain_prediction
            shap_info = explain_prediction(inputs)
        except Exception:
            shap_info = None

    if shap_info:
        writer.writerow(["shap.base_value", shap_info["base_value"]])
        for d in shap_info.get("top_positive_drivers", []):
            writer.writerow([f"shap.positive.{d['feature']}", d["shap_value"]])
        for d in shap_info.get("top_negative_drivers", []):
            writer.writerow([f"shap.negative.{d['feature']}", d["shap_value"]])

    for index, warning in enumerate(record.get("warnings") or [], start=1):
        writer.writerow([f"warning.{index}", warning])
    limitations = record.get("limitations") or []
    if isinstance(limitations, str):
        limitations = [limitations]
    for index, item in enumerate(limitations, start=1):
        writer.writerow([f"limitation.{index}", item])
    return buffer.getvalue().encode("utf-8")


def assessment_pdf(record: dict) -> bytes:
    wrapped: list[str] = []
    for line in assessment_lines(record):
        pieces = wrap(line, width=95) or [""]
        wrapped.extend(pieces)
    return _text_pdf(wrapped)


def _text_pdf(lines: list[str]) -> bytes:
    lines_per_page = 48
    pages = [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)] or [[""]]
    objects: list[bytes] = []

    catalog_id = 1
    pages_id = 2
    font_id = 3
    base_object_count = 3

    page_ids: list[int] = []
    content_ids: list[int] = []
    current_id = base_object_count + 1

    for _ in pages:
        page_ids.append(current_id)
        content_ids.append(current_id + 1)
        current_id += 2

    # Object 1: Catalog
    objects.append(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))

    # Object 2: Pages
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [ {kids} ] /Count {len(page_ids)} >>".encode("ascii"))

    # Object 3: Font
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    # Page and Content objects
    for page_idx, page_lines in enumerate(pages):
        pid = page_ids[page_idx]
        cid = content_ids[page_idx]
        objects.append(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {cid} 0 R >>".encode("ascii")
        )

        stream_lines = ["BT", "/F1 10 Tf", "50 742 Td", "14 TL"]
        for idx, line in enumerate(page_lines):
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            escaped = "".join(ch if 32 <= ord(ch) <= 126 else "?" for ch in safe)
            if idx == 0:
                stream_lines.append(f"({escaped}) Tj")
            else:
                stream_lines.append(f"T* ({escaped}) Tj")
        stream_lines.append("ET")
        stream_content = "\n".join(stream_lines).encode("ascii")
        stream_obj = (
            f"<< /Length {len(stream_content)} >>\nstream\n".encode("ascii")
            + stream_content
            + b"\nendstream"
        )
        objects.append(stream_obj)

    # PDF Assembly
    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{idx} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)

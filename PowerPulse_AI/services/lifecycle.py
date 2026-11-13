"""Battery lifecycle, dynamic age calculation, and smart alert definitions."""

from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Any

# Configurable battery ageing thresholds in months
DEFAULT_AGEING_THRESHOLDS = {
    "info_months": 24,       # 2 years
    "warning_months": 36,    # 3 years
    "advanced_months": 48,   # 4 years
}


def parse_date(value: Any) -> date | None:
    """Safely parse various date representations into a datetime.date object."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if "T" in cleaned:
            cleaned = cleaned.split("T")[0]
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
    return None


def add_months(source_date: date, months: int) -> date:
    """Accurately add months to a date, respecting leap years and month ends."""
    year = source_date.year + (source_date.month + months - 1) // 12
    month = (source_date.month + months - 1) % 12 + 1
    max_day = calendar.monthrange(year, month)[1]
    day = min(source_date.day, max_day)
    return date(year, month, day)


def calculate_dynamic_battery_age(
    purchase_date: date | str | None,
    as_of: date | None = None,
    thresholds: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Calculate battery age dynamically in months and years/months.

    Does not store permanently fixed age; calculates dynamically from current evaluation date.
    """
    parsed = parse_date(purchase_date)
    cfg = thresholds or DEFAULT_AGEING_THRESHOLDS
    info_th = cfg.get("info_months", 24)
    warn_th = cfg.get("warning_months", 36)
    adv_th = cfg.get("advanced_months", 48)

    if parsed is None:
        return {
            "available": False,
            "months": 0,
            "years": 0,
            "remaining_months": 0,
            "formatted": "Unknown",
            "tier": "NORMAL",
            "badge_color": "green",
            "tier_label": "Normal",
            "requires_attention": False,
        }

    current = as_of or date.today()
    if parsed > current:
        return {
            "available": True,
            "months": 0,
            "years": 0,
            "remaining_months": 0,
            "formatted": "0 months (Future purchase date)",
            "tier": "NORMAL",
            "badge_color": "green",
            "tier_label": "Normal",
            "requires_attention": False,
        }

    # Accurate calendar month calculation
    total_months = (current.year - parsed.year) * 12 + (current.month - parsed.month)
    if current.day < parsed.day:
        total_months -= 1
    total_months = max(0, total_months)

    years = total_months // 12
    rem_months = total_months % 12

    # Human-friendly formatted age
    parts = []
    if years > 0:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if rem_months > 0 or not parts:
        parts.append(f"{rem_months} month{'s' if rem_months != 1 else ''}")
    formatted = f"{total_months} months ({', '.join(parts)})"

    # Ageing tier classification based on configurable thresholds
    if total_months < info_th:
        tier = "NORMAL"
        badge_color = "green"
        tier_label = f"Normal (< {info_th} mo)"
        requires_attention = False
    elif info_th <= total_months < warn_th:
        tier = "AGEING_INFO"
        badge_color = "yellow"
        tier_label = f"Ageing Info ({info_th}–{warn_th} mo)"
        requires_attention = True
    elif warn_th <= total_months <= adv_th:
        tier = "AGEING_WARNING"
        badge_color = "orange"
        tier_label = f"Ageing Warning ({warn_th}–{adv_th} mo)"
        requires_attention = True
    else:
        tier = "ADVANCED_AGEING"
        badge_color = "red"
        tier_label = f"Advanced Ageing (> {adv_th} mo)"
        requires_attention = True

    return {
        "available": True,
        "months": total_months,
        "years": years,
        "remaining_months": rem_months,
        "formatted": formatted,
        "tier": tier,
        "badge_color": badge_color,
        "tier_label": tier_label,
        "requires_attention": requires_attention,
    }


def get_ageing_alert_message(tier: str) -> str | None:
    """Return standard educational message for a battery ageing tier."""
    if tier == "AGEING_INFO":
        return "This battery has been in service for 2 years. Consider periodic health monitoring."
    elif tier == "AGEING_WARNING":
        return "Your battery is approaching 3 years of usage. A health assessment is recommended."
    elif tier == "ADVANCED_AGEING":
        return "This battery has been in service for more than 4 years. Consider a detailed battery inspection."
    return None


def evaluate_warranty_milestones(days_remaining: int) -> list[dict[str, Any]]:
    """Determine applicable warranty milestone alerts for a given days-remaining value.

    Conditions:
      - 90 days before expiry: "Your battery warranty expires in 3 months."
      - 30 days before expiry: "Your battery warranty expires in 30 days."
      - 7 days before expiry: "Your battery warranty expires in 7 days."
      - After expiry: "Your battery warranty has expired."
    """
    alerts: list[dict[str, Any]] = []

    if days_remaining <= 0:
        alerts.append(
            {
                "alert_type": "WARRANTY_EXPIRED",
                "alert_message": "Your battery warranty has expired.",
                "category": "warranty",
                "severity": "red",
            }
        )
    else:
        if days_remaining <= 7:
            alerts.append(
                {
                    "alert_type": "WARRANTY_EXPIRING_7",
                    "alert_message": "Your battery warranty expires in 7 days.",
                    "category": "warranty",
                    "severity": "yellow",
                }
            )
        if days_remaining <= 30:
            alerts.append(
                {
                    "alert_type": "WARRANTY_EXPIRING_30",
                    "alert_message": "Your battery warranty expires in 30 days.",
                    "category": "warranty",
                    "severity": "yellow",
                }
            )
        if days_remaining <= 90:
            alerts.append(
                {
                    "alert_type": "WARRANTY_EXPIRING_90",
                    "alert_message": "Your battery warranty expires in 3 months.",
                    "category": "warranty",
                    "severity": "yellow",
                }
            )

    return alerts


def compute_overall_alert_status(
    warranty_status: str,
    days_remaining: int | None,
    age_months: int,
) -> dict[str, Any]:
    """Compute consolidated visual indicator and label for the battery table.

    Indicators:
      - Green: Normal / Active
      - Yellow: Approaching expiry (<= 90 days) / Ageing information (24-36 mo)
      - Orange: Ageing warning (36-48 mo)
      - Red: Expired / Advanced ageing (> 48 mo)
    """
    if warranty_status == "EXPIRED":
        return {
            "color": "red",
            "badge": "🔴 Expired",
            "text": "Warranty Expired",
            "severity_rank": 4,
        }
    if age_months > 48:
        return {
            "color": "red",
            "badge": "🔴 Advanced Ageing",
            "text": "Advanced Ageing (> 48 mo)",
            "severity_rank": 4,
        }
    if age_months >= 36:
        return {
            "color": "orange",
            "badge": "🟠 Ageing Warning",
            "text": "Ageing Warning (36–48 mo)",
            "severity_rank": 3,
        }
    if days_remaining is not None and days_remaining <= 90:
        return {
            "color": "yellow",
            "badge": "🟡 Approaching Expiry",
            "text": f"Warranty Expiring ({days_remaining}d)",
            "severity_rank": 2,
        }
    if age_months >= 24:
        return {
            "color": "yellow",
            "badge": "🟡 Ageing Info",
            "text": "Ageing Info (24–36 mo)",
            "severity_rank": 2,
        }
    return {
        "color": "green",
        "badge": "🟢 Normal / Active",
        "text": "Normal / Active",
        "severity_rank": 1,
    }

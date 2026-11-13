"""Battery warranty calculation, validation, and policy enforcement module."""

from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Any

from ml.errors import (
    PowerPulseError,
    WarrantyError,
    WarrantyExpiredError,
    WarrantyNotRegisteredError,
    WarrantyValidationError,
)


def parse_date(value: Any) -> date | None:
    """Safely parse diverse date representations into a standard datetime.date."""
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


def add_months_to_date(source_date: date, months: int) -> date:
    """Accurately add months to a date, respecting month lengths and leap years."""
    year = source_date.year + (source_date.month + months - 1) // 12
    month = (source_date.month + months - 1) % 12 + 1
    max_day = calendar.monthrange(year, month)[1]
    day = min(source_date.day, max_day)
    return date(year, month, day)


def calculate_warranty_expiry(
    purchase_date: date | str,
    warranty_years: int,
    warranty_months: int = 0,
) -> date:
    """Calculate the exact warranty expiry date using purchase date and warranty duration."""
    parsed_date = parse_date(purchase_date)
    if parsed_date is None:
        raise WarrantyValidationError("A valid battery purchase date is required.")

    total_months = int(warranty_years) * 12 + int(warranty_months)
    if total_months <= 0:
        raise WarrantyValidationError("Total warranty duration must be greater than zero.")

    return add_months_to_date(parsed_date, total_months)


def validate_warranty_inputs(
    battery_id: str,
    manufacturer: str,
    battery_model: str,
    purchase_date: date | str,
    warranty_years: int,
    warranty_months: int = 0,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Validate user inputs for registering or updating a battery warranty."""
    clean_id = (battery_id or "").strip()
    if not clean_id:
        raise WarrantyValidationError("Battery ID is required.")
    if len(clean_id) > 64:
        raise WarrantyValidationError("Battery ID must be 64 characters or fewer.")

    clean_mfg = (manufacturer or "").strip()
    if not clean_mfg:
        raise WarrantyValidationError("Battery manufacturer is required.")

    clean_model = (battery_model or "").strip()
    if not clean_model:
        raise WarrantyValidationError("Battery model is required.")

    parsed_purchase = parse_date(purchase_date)
    if parsed_purchase is None:
        raise WarrantyValidationError("A valid battery purchase date is required.")

    current_date = as_of or date.today()
    if parsed_purchase > current_date:
        raise WarrantyValidationError("Battery purchase date cannot be in the future.")

    try:
        w_years = int(warranty_years)
        w_months = int(warranty_months)
    except (TypeError, ValueError) as exc:
        raise WarrantyValidationError("Warranty duration values must be valid integers.") from exc

    if w_years < 0 or w_months < 0:
        raise WarrantyValidationError("Warranty duration values cannot be negative.")

    total_months = w_years * 12 + w_months
    if total_months <= 0:
        raise WarrantyValidationError("Total warranty duration must be at least 1 month.")

    expiry_date = calculate_warranty_expiry(parsed_purchase, w_years, w_months)
    status, days_remaining = evaluate_status(expiry_date, as_of=current_date)

    return {
        "battery_id": clean_id,
        "manufacturer": clean_mfg,
        "battery_model": clean_model,
        "purchase_date": parsed_purchase.isoformat(),
        "warranty_years": w_years,
        "warranty_months": w_months,
        "warranty_expiry_date": expiry_date.isoformat(),
        "warranty_status": status,
        "days_remaining": days_remaining,
    }


def evaluate_status(expiry_date: date, as_of: date | None = None) -> tuple[str, int]:
    """Determine ACTIVE or EXPIRED status based on the current evaluation date.

    - ACTIVE: Current date is on or before the warranty expiry date.
    - EXPIRED: Current date is after the warranty expiry date.
    """
    current_date = as_of or date.today()
    days_remaining = (expiry_date - current_date).days
    status = "ACTIVE" if current_date <= expiry_date else "EXPIRED"
    return status, days_remaining


def get_warranty_status(
    purchase_date: date | str,
    warranty_years: int,
    warranty_months: int = 0,
    as_of: date | None = None,
) -> tuple[str, date, int]:
    """Calculate expiry date and current warranty status."""
    expiry_date = calculate_warranty_expiry(purchase_date, warranty_years, warranty_months)
    status, days_remaining = evaluate_status(expiry_date, as_of=as_of)
    return status, expiry_date, days_remaining


def check_warranty_validity(
    battery_record: dict[str, Any] | None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Evaluate whether a battery record is eligible for ML health assessment.

    Revalidates the expiry date dynamically against the current date.
    """
    if not battery_record:
        return {
            "is_valid": False,
            "status": "NOT_REGISTERED",
            "message": "Battery is not registered. Please register warranty details before performing a health assessment.",
            "expiry_date": None,
            "days_remaining": None,
        }

    purchase_date = battery_record.get("purchase_date")
    warranty_years = battery_record.get("warranty_years")
    warranty_months = battery_record.get("warranty_months")

    # If warranty_years is missing, check total warranty_months fallback
    if warranty_years is None and warranty_months is not None:
        warranty_years = int(warranty_months) // 12
        warranty_months = int(warranty_months) % 12

    if not purchase_date or (warranty_years is None and warranty_months is None):
        return {
            "is_valid": False,
            "status": "INCOMPLETE",
            "message": "Battery warranty details are incomplete. Please register purchase date and warranty duration.",
            "expiry_date": None,
            "days_remaining": None,
        }

    parsed_purchase = parse_date(purchase_date)
    if parsed_purchase is None:
        return {
            "is_valid": False,
            "status": "INVALID_DATE",
            "message": "Stored battery purchase date is invalid.",
            "expiry_date": None,
            "days_remaining": None,
        }

    w_years = int(warranty_years or 0)
    w_months = int(warranty_months or 0)
    if (w_years * 12 + w_months) <= 0:
        return {
            "is_valid": False,
            "status": "INVALID_DURATION",
            "message": "Stored warranty duration must be greater than zero.",
            "expiry_date": None,
            "days_remaining": None,
        }

    expiry_date = calculate_warranty_expiry(parsed_purchase, w_years, w_months)
    status, days_remaining = evaluate_status(expiry_date, as_of=as_of)

    if status == "ACTIVE":
        return {
            "is_valid": True,
            "status": "ACTIVE",
            "expiry_date": expiry_date.isoformat(),
            "days_remaining": days_remaining,
            "message": (
                f"WARRANTY ACTIVE — Valid until {expiry_date.isoformat()} "
                f"({days_remaining} day{'s' if days_remaining != 1 else ''} remaining). "
                "Battery health assessment is authorized under the current warranty policy."
            ),
        }
    else:
        return {
            "is_valid": False,
            "status": "EXPIRED",
            "expiry_date": expiry_date.isoformat(),
            "days_remaining": days_remaining,
            "message": "WARRANTY EXPIRED – Battery health assessment is unavailable under the current warranty policy.",
        }


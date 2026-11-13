"""Automated tests for battery warranty management, validation, and policy enforcement."""

from datetime import date, timedelta
from pathlib import Path

import pytest

from database.db import (
    add_assessment,
    create_user,
    get_battery_warranty,
    list_assessments,
    list_user_batteries,
    register_battery_warranty,
)
from ml.errors import WarrantyExpiredError, WarrantyNotRegisteredError, WarrantyValidationError
from ml.predict import assess
from utils.warranty import (
    calculate_warranty_expiry,
    check_warranty_validity,
    get_warranty_status,
    validate_warranty_inputs,
)

SAMPLE_TELEMETRY = {
    "ambient_temperature": 24.0,
    "temperature_mean": 32.0,
    "temperature_max": 39.0,
    "voltage_mean": 3.5,
    "voltage_min": 2.5,
    "voltage_max": 4.1,
    "voltage_std": 0.24,
    "current_mean": -1.8,
    "current_std": 0.5,
    "duration_seconds": 3600.0,
}


def test_scenario_1_past_purchase_active_warranty():
    """Scenario 1: Purchase date in the past + active warranty duration -> ACTIVE."""
    today = date(2026, 11, 13)
    purchase = date(2025, 11, 13)  # 1 year ago
    status, expiry_date, days_remaining = get_warranty_status(
        purchase_date=purchase,
        warranty_years=3,  # 3 years total -> expires 2028-11-13
        warranty_months=0,
        as_of=today,
    )
    assert status == "ACTIVE"
    assert expiry_date == date(2028, 11, 13)
    assert days_remaining > 0


def test_scenario_2_past_purchase_expired_warranty():
    """Scenario 2: Purchase date in the past + expired warranty duration -> EXPIRED."""
    today = date(2026, 11, 13)
    purchase = date(2023, 1, 15)  # 3+ years ago
    status, expiry_date, days_remaining = get_warranty_status(
        purchase_date=purchase,
        warranty_years=2,  # 2 years warranty -> expired 2025-01-15
        warranty_months=0,
        as_of=today,
    )
    assert status == "EXPIRED"
    assert expiry_date == date(2025, 1, 15)
    assert days_remaining < 0


def test_scenario_3_future_purchase_date_validation_error():
    """Scenario 3: Purchase date in the future -> validation error."""
    today = date(2026, 11, 13)
    future_date = date(2026, 12, 1)
    with pytest.raises(WarrantyValidationError, match="future"):
        validate_warranty_inputs(
            battery_id="BAT-FUTURE",
            manufacturer="Panasonic",
            battery_model="NCR-18650B",
            purchase_date=future_date,
            warranty_years=2,
            warranty_months=0,
            as_of=today,
        )


def test_scenario_4_zero_or_negative_warranty_duration_error():
    """Scenario 4: Zero or negative warranty duration -> validation error."""
    today = date(2026, 11, 13)
    purchase = date(2025, 5, 1)

    # Total duration = 0
    with pytest.raises(WarrantyValidationError, match="at least 1 month"):
        validate_warranty_inputs(
            battery_id="BAT-ZERO",
            manufacturer="LG",
            battery_model="HG2",
            purchase_date=purchase,
            warranty_years=0,
            warranty_months=0,
            as_of=today,
        )

    # Negative duration
    with pytest.raises(WarrantyValidationError, match="negative"):
        validate_warranty_inputs(
            battery_id="BAT-NEG",
            manufacturer="LG",
            battery_model="HG2",
            purchase_date=purchase,
            warranty_years=-1,
            warranty_months=0,
            as_of=today,
        )


def test_scenario_5_active_warranty_ml_assessment_allowed(tmp_path):
    """Scenario 5: Battery with active warranty -> ML assessment allowed."""
    db = tmp_path / "warranty_test.db"
    user_id = create_user("service_sc5", "ValidPass123", path=db)
    today = date(2026, 11, 13)

    register_battery_warranty(
        battery_id="BAT-ACTIVE-01",
        manufacturer="Samsung SDI",
        battery_model="30Q",
        purchase_date="2025-06-01",
        warranty_years=3,
        warranty_months=0,
        user_id=user_id,
        path=db,
        as_of=today,
    )

    result = assess(
        measurements=SAMPLE_TELEMETRY,
        battery_id="BAT-ACTIVE-01",
        user_id=user_id,
        as_of=today,
        db_path=db,
    )
    assert result["status"] == "predicted"
    assert "predicted_soh" in result
    assert result["predicted_condition"] in ("Good", "Fair", "Poor")
    assert result["battery_id"] == "BAT-ACTIVE-01"


def test_scenario_6_and_7_expired_warranty_assessment_blocked(tmp_path):
    """Scenario 6 & 7: Battery with expired warranty -> ML assessment blocked; direct call blocked."""
    db = tmp_path / "warranty_test.db"
    user_id = create_user("service_sc6", "ValidPass123", path=db)
    today = date(2026, 11, 13)

    register_battery_warranty(
        battery_id="BAT-EXPIRED-01",
        manufacturer="Samsung SDI",
        battery_model="25R",
        purchase_date="2023-01-01",
        warranty_years=2,  # Expired 2025-01-01
        warranty_months=0,
        user_id=user_id,
        path=db,
        as_of=today,
    )

    # Direct call to assess() must be blocked with WarrantyExpiredError
    with pytest.raises(WarrantyExpiredError, match="WARRANTY EXPIRED"):
        assess(
            measurements=SAMPLE_TELEMETRY,
            battery_id="BAT-EXPIRED-01",
            user_id=user_id,
            as_of=today,
            db_path=db,
        )


def test_scenario_8_no_assessment_record_created_on_expired_warranty(tmp_path):
    """Scenario 8: Verify that expired warranty attempts do not create ML assessment records."""
    db = tmp_path / "warranty_test.db"
    user_id = create_user("service_sc8", "ValidPass123", path=db)
    today = date(2026, 11, 13)

    register_battery_warranty(
        battery_id="BAT-EXPIRED-RECORD",
        manufacturer="Panasonic",
        battery_model="NCR",
        purchase_date="2022-01-01",
        warranty_years=1,
        warranty_months=0,
        user_id=user_id,
        path=db,
        as_of=today,
    )

    initial_records = list_assessments(user_id, path=db)
    assert len(initial_records) == 0

    try:
        assess(
            measurements=SAMPLE_TELEMETRY,
            battery_id="BAT-EXPIRED-RECORD",
            user_id=user_id,
            as_of=today,
            db_path=db,
        )
    except WarrantyExpiredError:
        pass  # Expected block

    # Verify database still has 0 assessments
    subsequent_records = list_assessments(user_id, path=db)
    assert len(subsequent_records) == 0


def test_scenario_9_database_persistence_and_retrieval(tmp_path):
    """Scenario 9: Verify warranty information is correctly stored and retrieved from the database."""
    db = tmp_path / "warranty_test.db"
    user_id = create_user("service_sc9", "ValidPass123", path=db)
    today = date(2026, 11, 13)

    register_battery_warranty(
        battery_id="BAT-STORE-TEST",
        manufacturer="CATL",
        battery_model="Blade-Cell-01",
        purchase_date="2024-05-20",
        warranty_years=4,
        warranty_months=6,
        user_id=user_id,
        path=db,
        as_of=today,
    )

    loaded = get_battery_warranty("BAT-STORE-TEST", user_id=user_id, as_of=today, path=db)
    assert loaded is not None
    assert loaded["battery_id"] == "BAT-STORE-TEST"
    assert loaded["manufacturer"] == "CATL"
    assert loaded["battery_model"] == "Blade-Cell-01"
    assert loaded["purchase_date"] == "2024-05-20"
    assert loaded["warranty_years"] == 4
    assert loaded["warranty_months"] == 6
    assert loaded["warranty_expiry_date"] == "2028-11-20"
    assert loaded["warranty_status"] == "ACTIVE"


def test_scenario_10_account_level_isolation(tmp_path):
    """Scenario 10: Verify one service centre cannot access another service centre's battery records."""
    db = tmp_path / "warranty_test.db"
    user1 = create_user("centre_alpha", "ValidAlphaPass1", path=db)
    user2 = create_user("centre_beta", "ValidBetaPass2", path=db)

    # Service Centre 1 registers a battery
    register_battery_warranty(
        battery_id="BAT-CONFIDENTIAL-01",
        manufacturer="Tesla",
        battery_model="4680",
        purchase_date="2025-01-01",
        warranty_years=8,
        user_id=user1,
        path=db,
    )

    # Service Centre 1 can access it
    assert get_battery_warranty("BAT-CONFIDENTIAL-01", user_id=user1, path=db) is not None

    # Service Centre 2 CANNOT access Service Centre 1's battery
    assert get_battery_warranty("BAT-CONFIDENTIAL-01", user_id=user2, path=db) is None

    # Service Centre 2 cannot see it in user battery list
    sc2_batteries = list_user_batteries(user_id=user2, path=db)
    assert not any(b["battery_id"] == "BAT-CONFIDENTIAL-01" for b in sc2_batteries)

    # Service Centre 2 cannot run assessment on Service Centre 1's battery
    with pytest.raises(WarrantyNotRegisteredError):
        assess(
            measurements=SAMPLE_TELEMETRY,
            battery_id="BAT-CONFIDENTIAL-01",
            user_id=user2,
            db_path=db,
        )


def test_leap_year_handling():
    """Verify leap year calculations (e.g. Feb 29)."""
    # 2024 is leap year, purchased Feb 29, 2024
    purchase = date(2024, 2, 29)
    # 1 year warranty expires on Feb 28, 2025 (non-leap year)
    expiry_1yr = calculate_warranty_expiry(purchase, warranty_years=1, warranty_months=0)
    assert expiry_1yr == date(2025, 2, 28)

    # 4 years warranty expires on Feb 29, 2028 (leap year)
    expiry_4yr = calculate_warranty_expiry(purchase, warranty_years=4, warranty_months=0)
    assert expiry_4yr == date(2028, 2, 29)

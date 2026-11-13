"""Automated tests for Battery Lifecycle Monitoring and Smart Alerts in PowerPulse."""

from datetime import date, timedelta
from pathlib import Path

import pytest

from database.db import (
    acknowledge_battery_alert,
    create_user,
    get_lifecycle_monitoring_summary,
    list_battery_alerts,
    register_battery_warranty,
)
from ml.errors import WarrantyExpiredError
from ml.predict import assess
from services.lifecycle import (
    add_months,
    calculate_dynamic_battery_age,
    compute_overall_alert_status,
    evaluate_warranty_milestones,
    get_ageing_alert_message,
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


def test_scenario_1_newly_registered_battery_with_active_warranty():
    """Scenario 1: Newly registered battery with active warranty."""
    today = date(2026, 11, 13)
    purchase = date(2026, 11, 1)  # 12 days ago
    age_info = calculate_dynamic_battery_age(purchase, as_of=today)
    assert age_info["available"] is True
    assert age_info["months"] == 0
    assert age_info["tier"] == "NORMAL"
    assert age_info["requires_attention"] is False

    # Warranty expires in 3 years (1095 days)
    status_info = compute_overall_alert_status(
        warranty_status="ACTIVE",
        days_remaining=1095,
        age_months=0,
    )
    assert status_info["color"] == "green"
    assert "Active" in status_info["text"]


def test_scenario_2_warranty_expiring_in_90_days():
    """Scenario 2: Warranty expiring in 90 days."""
    alerts = evaluate_warranty_milestones(days_remaining=90)
    alert_types = [a["alert_type"] for a in alerts]
    assert "WARRANTY_EXPIRING_90" in alert_types
    match = next(a for a in alerts if a["alert_type"] == "WARRANTY_EXPIRING_90")
    assert match["alert_message"] == "Your battery warranty expires in 3 months."


def test_scenario_3_warranty_expiring_in_30_days():
    """Scenario 3: Warranty expiring in 30 days."""
    alerts = evaluate_warranty_milestones(days_remaining=30)
    alert_types = [a["alert_type"] for a in alerts]
    assert "WARRANTY_EXPIRING_30" in alert_types
    match = next(a for a in alerts if a["alert_type"] == "WARRANTY_EXPIRING_30")
    assert match["alert_message"] == "Your battery warranty expires in 30 days."


def test_scenario_4_warranty_expiring_in_7_days():
    """Scenario 4: Warranty expiring in 7 days."""
    alerts = evaluate_warranty_milestones(days_remaining=7)
    alert_types = [a["alert_type"] for a in alerts]
    assert "WARRANTY_EXPIRING_7" in alert_types
    match = next(a for a in alerts if a["alert_type"] == "WARRANTY_EXPIRING_7")
    assert match["alert_message"] == "Your battery warranty expires in 7 days."


def test_scenario_5_already_expired_warranty():
    """Scenario 5: Already expired warranty."""
    alerts = evaluate_warranty_milestones(days_remaining=-5)
    alert_types = [a["alert_type"] for a in alerts]
    assert "WARRANTY_EXPIRED" in alert_types
    match = next(a for a in alerts if a["alert_type"] == "WARRANTY_EXPIRED")
    assert match["alert_message"] == "Your battery warranty has expired."


def test_scenario_6_battery_age_below_24_months():
    """Scenario 6: Battery age below 24 months (< 2 years)."""
    today = date(2026, 11, 13)
    purchase = date(2025, 5, 13)  # 18 months ago
    age_info = calculate_dynamic_battery_age(purchase, as_of=today)
    assert age_info["months"] == 18
    assert age_info["tier"] == "NORMAL"
    assert age_info["badge_color"] == "green"
    assert age_info["requires_attention"] is False
    assert get_ageing_alert_message(age_info["tier"]) is None


def test_scenario_7_battery_age_between_24_and_36_months():
    """Scenario 7: Battery age between 24 and 36 months."""
    today = date(2026, 11, 13)
    purchase = date(2024, 5, 13)  # 30 months ago
    age_info = calculate_dynamic_battery_age(purchase, as_of=today)
    assert age_info["months"] == 30
    assert age_info["tier"] == "AGEING_INFO"
    assert age_info["badge_color"] == "yellow"
    assert age_info["requires_attention"] is True
    msg = get_ageing_alert_message(age_info["tier"])
    assert msg == "This battery has been in service for 2 years. Consider periodic health monitoring."


def test_scenario_8_battery_age_between_36_and_48_months():
    """Scenario 8: Battery age between 36 and 48 months."""
    today = date(2026, 11, 13)
    purchase = date(2023, 7, 13)  # 40 months ago
    age_info = calculate_dynamic_battery_age(purchase, as_of=today)
    assert age_info["months"] == 40
    assert age_info["tier"] == "AGEING_WARNING"
    assert age_info["badge_color"] == "orange"
    assert age_info["requires_attention"] is True
    msg = get_ageing_alert_message(age_info["tier"])
    assert msg == "Your battery is approaching 3 years of usage. A health assessment is recommended."


def test_scenario_9_battery_age_above_48_months():
    """Scenario 9: Battery age above 48 months (> 4 years)."""
    today = date(2026, 11, 13)
    purchase = date(2022, 1, 13)  # 58 months ago
    age_info = calculate_dynamic_battery_age(purchase, as_of=today)
    assert age_info["months"] == 58
    assert age_info["tier"] == "ADVANCED_AGEING"
    assert age_info["badge_color"] == "red"
    assert age_info["requires_attention"] is True
    msg = get_ageing_alert_message(age_info["tier"])
    assert msg == "This battery has been in service for more than 4 years. Consider a detailed battery inspection."


def test_scenario_10_dashboard_refresh_idempotent_no_duplicate_alerts(tmp_path):
    """Scenario 10: Dashboard refresh does not generate duplicate alerts."""
    db = tmp_path / "lifecycle_test.db"
    user_id = create_user("service_sc_idempotent", "Password123", path=db)
    today = date(2026, 11, 13)

    # Register battery with warranty expiring in 25 days (triggers 30d & 90d alerts)
    register_battery_warranty(
        battery_id="BAT-IDEM-01",
        manufacturer="Panasonic",
        battery_model="NCR",
        purchase_date="2024-12-08",
        warranty_years=2,  # Expires 2026-12-08 -> exactly 25 days from 2026-11-13
        warranty_months=0,
        user_id=user_id,
        path=db,
        as_of=today,
    )

    # First load
    summary1 = get_lifecycle_monitoring_summary(user_id=user_id, as_of=today, path=db)
    alerts1_count = len(summary1["unread_alerts"])
    assert alerts1_count > 0

    # Second load (simulate page refresh)
    summary2 = get_lifecycle_monitoring_summary(user_id=user_id, as_of=today, path=db)
    alerts2_count = len(summary2["unread_alerts"])
    assert alerts2_count == alerts1_count, "Refreshing dashboard must not create duplicate alerts"

    # Third load
    summary3 = get_lifecycle_monitoring_summary(user_id=user_id, as_of=today, path=db)
    assert len(summary3["unread_alerts"]) == alerts1_count


def test_scenario_11_acknowledging_alert_updates_database(tmp_path):
    """Scenario 11: Acknowledging an alert updates database status to Acknowledged."""
    db = tmp_path / "lifecycle_test.db"
    user_id = create_user("service_sc_ack", "Password123", path=db)
    today = date(2026, 11, 13)

    register_battery_warranty(
        battery_id="BAT-ACK-01",
        manufacturer="CATL",
        battery_model="Blade",
        purchase_date="2023-11-13",  # 36 months old -> ageing warning alert
        warranty_years=5,
        warranty_months=0,
        user_id=user_id,
        path=db,
        as_of=today,
    )

    summary = get_lifecycle_monitoring_summary(user_id=user_id, as_of=today, path=db)
    assert len(summary["unread_alerts"]) >= 1

    alert_to_ack = summary["unread_alerts"][0]
    alert_id = alert_to_ack["id"]

    success = acknowledge_battery_alert(alert_id, user_id=user_id, path=db)
    assert success is True

    # Re-fetch summary
    updated_summary = get_lifecycle_monitoring_summary(user_id=user_id, as_of=today, path=db)
    assert not any(a["id"] == alert_id for a in updated_summary["unread_alerts"])
    assert any(a["id"] == alert_id for a in updated_summary["acknowledged_alerts"])


def test_scenario_12_and_13_ml_eligibility_and_expired_restriction(tmp_path):
    """Scenario 12 & 13: Eligible batteries proceed with ML; expired batteries are locked."""
    db = tmp_path / "lifecycle_test.db"
    user_id = create_user("service_sc_ml", "Password123", path=db)
    today = date(2026, 11, 13)

    # 1. Eligible battery
    register_battery_warranty(
        battery_id="BAT-ELIGIBLE",
        manufacturer="LG",
        battery_model="HG2",
        purchase_date="2025-01-01",
        warranty_years=3,
        user_id=user_id,
        path=db,
        as_of=today,
    )
    result = assess(
        measurements=SAMPLE_TELEMETRY,
        battery_id="BAT-ELIGIBLE",
        user_id=user_id,
        as_of=today,
        db_path=db,
    )
    assert result["status"] == "predicted"
    assert "predicted_soh" in result

    # 2. Expired battery
    register_battery_warranty(
        battery_id="BAT-EXPIRED-LOCKED",
        manufacturer="LG",
        battery_model="HG2",
        purchase_date="2022-01-01",
        warranty_years=2,  # Expired 2024-01-01
        user_id=user_id,
        path=db,
        as_of=today,
    )
    with pytest.raises(WarrantyExpiredError, match="WARRANTY EXPIRED"):
        assess(
            measurements=SAMPLE_TELEMETRY,
            battery_id="BAT-EXPIRED-LOCKED",
            user_id=user_id,
            as_of=today,
            db_path=db,
        )


def test_date_calculations_month_end_and_leap_year():
    """Verify leap year and month-end date calculations."""
    # Leap year purchase: Feb 29, 2024
    leap_purchase = date(2024, 2, 29)
    # Add 12 months -> Feb 28, 2025 (non-leap year)
    res_1yr = add_months(leap_purchase, 12)
    assert res_1yr == date(2025, 2, 28)

    # Month end: Aug 31 + 1 month -> Sep 30
    aug_31 = date(2025, 8, 31)
    sep_end = add_months(aug_31, 1)
    assert sep_end == date(2025, 9, 30)


import pytest

from database.db import (
    add_assessment,
    authenticate,
    create_user,
    dashboard_summary,
    get_assessment,
    hash_password,
    list_assessments,
)


def test_password_hash_is_not_the_password_and_login_round_trips(tmp_path):
    database = tmp_path / "powerpulse.db"
    stored = hash_password("a-reasonable-password")
    assert "a-reasonable-password" not in stored
    user_id = create_user("hasini", "a-reasonable-password", database)
    assert authenticate("hasini", "a-reasonable-password", database)["id"] == user_id
    assert authenticate("hasini", "wrong-password", database) is None
    with pytest.raises(ValueError):
        create_user("hasini", "a-reasonable-password", database)


def test_assessment_insert_and_scoped_retrieval(tmp_path):
    database = tmp_path / "powerpulse.db"
    guest = add_assessment(
        {
            "battery_identifier": "CELL-1",
            "category": "lithium_ion",
            "user_id": None,
            "status": "predicted",
            "model_version": "1.0.0",
            "predicted_soh": 88.2,
            "predicted_condition": "Good",
            "soh_band": "Good",
            "inputs": {"voltage_mean": 3.5, "discharge_cycle": 10},
            "warnings": [],
            "guidance": ["monitor"],
            "limitations": ["laboratory model"],
        },
        database,
    )
    owner = create_user("tech1", "workshop1", database)
    owned = add_assessment(
        {
            "battery_identifier": "CELL-2",
            "category": "lithium_ion",
            "user_id": owner,
            "status": "predicted",
            "predicted_soh": 68.5,
            "predicted_condition": "Poor",
            "soh_band": "Poor",
            "inputs": {"voltage_mean": 3.2, "discharge_cycle": 120},
            "warnings": ["CRITICAL ALERT"],
            "guidance": ["inspect"],
            "limitations": ["laboratory model"],
        },
        database,
    )
    guest_rows = list_assessments(None, database)
    owner_rows = list_assessments(owner, database)
    assert [row["id"] for row in guest_rows] == [guest]
    assert [row["id"] for row in owner_rows] == [owned]
    assert get_assessment(owned, None, database) is None
    loaded = get_assessment(guest, None, database)
    assert loaded["inputs"]["voltage_mean"] == 3.5
    assert loaded["predicted_soh"] == 88.2
    summary = dashboard_summary(None, database)
    assert summary["battery_count"] == 1
    assert summary["average_predicted_soh"] == 88.2
    hidden = dashboard_summary(owner, database)
    assert hidden["average_predicted_soh"] == 68.5
    assert hidden["assessment_count"] == 1
    assert len(hidden["alerts"]) == 1


def test_env_config_defaults_and_parsing(monkeypatch):
    from database.db import get_active_db_type, get_db_config

    monkeypatch.setenv("DB_TYPE", "mongodb")
    monkeypatch.setenv("MONGODB_URI", "")
    cfg = get_db_config()
    assert cfg["db_type"] == "mongodb"
    assert cfg["mongo_db_name"] == "powerpulse_db"
    assert cfg["users_collection"] == "users"
    assert cfg["batteries_collection"] == "battery_records"
    assert cfg["assessments_collection"] == "assessments"

    # With empty URI, active backend safely falls back to sqlite
    assert get_active_db_type() == "sqlite"


def test_explicit_path_always_routes_to_sqlite(tmp_path):
    from database.db import get_active_db_type

    test_db = tmp_path / "custom.db"
    assert get_active_db_type(test_db) == "sqlite"


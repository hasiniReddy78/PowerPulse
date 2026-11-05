"""SQLite access layer for lithium-ion battery records and assessment history."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from database.schema import INDEXES, STATEMENTS
from ml.config import DATABASE_PATH

PBKDF2_ROUNDS = 200_000


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect(path: Path | None = None) -> sqlite3.Connection:
    database = Path(path) if path is not None else DATABASE_PATH
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(path: Path | None = None) -> None:
    with connect(path) as connection:
        for statement in STATEMENTS:
            connection.execute(statement)
        for statement in INDEXES:
            connection.execute(statement)
        connection.commit()


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    candidate = hash_password(password, salt)
    return secrets.compare_digest(candidate, f"{salt.hex()}:{digest_hex}")


def create_user(username: str, password: str, path: Path | None = None) -> int:
    username = username.strip()
    if len(username) < 3 or len(username) > 32 or not username.replace("_", "").isalnum():
        raise ValueError("Username must be 3 to 32 characters and use letters, numbers, or underscore.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    init_db(path)
    with connect(path) as connection:
        try:
            cursor = connection.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, hash_password(password), utc_now()),
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError("That username is already registered.") from exc
        return int(cursor.lastrowid)


def authenticate(username: str, password: str, path: Path | None = None) -> dict | None:
    init_db(path)
    with connect(path) as connection:
        row = connection.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return {"id": int(row["id"]), "username": row["username"]}


def add_battery(identifier: str, category: str = "lithium_ion", user_id: int | None = None, path: Path | None = None) -> int:
    identifier = identifier.strip()
    if not identifier or len(identifier) > 64:
        raise ValueError("Battery identifier is required and must be 64 characters or fewer.")
    category = "lithium_ion"
    init_db(path)
    with connect(path) as connection:
        existing = connection.execute(
            """
            SELECT id FROM battery_records
            WHERE battery_identifier = ? AND category = ?
              AND ((user_id IS NULL AND ? IS NULL) OR user_id = ?)
            """,
            (identifier, category, user_id, user_id),
        ).fetchone()
        if existing is not None:
            return int(existing["id"])
        cursor = connection.execute(
            """
            INSERT INTO battery_records (user_id, battery_identifier, category, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, identifier, category, utc_now()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def add_assessment(payload: dict, path: Path | None = None) -> int:
    init_db(path)
    category = payload.get("category", "lithium_ion")
    battery_id = add_battery(
        payload["battery_identifier"],
        category,
        payload.get("user_id"),
        path,
    )
    limitations = payload.get("limitations") or []
    if isinstance(limitations, str):
        limitations = [limitations]
    with connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO assessments (
                battery_record_id, user_id, assessed_at, category, model_version,
                predicted_soh, predicted_condition, soh_band, status,
                inputs_json, warnings_json, guidance_json, limitations_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                battery_id,
                payload.get("user_id"),
                payload.get("assessed_at") or utc_now(),
                category,
                payload.get("model_version"),
                payload.get("predicted_soh"),
                payload.get("predicted_condition"),
                payload.get("soh_band"),
                payload.get("status", "predicted"),
                json.dumps(payload.get("inputs") or {}),
                json.dumps(payload.get("warnings") or []),
                json.dumps(payload.get("guidance") or []),
                json.dumps(limitations),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def _row_to_assessment(row: sqlite3.Row) -> dict:
    return {
        "id": int(row["id"]),
        "battery_record_id": int(row["battery_record_id"]),
        "user_id": row["user_id"],
        "battery_identifier": row["battery_identifier"],
        "assessed_at": row["assessed_at"],
        "category": row["category"],
        "model_version": row["model_version"],
        "predicted_soh": row["predicted_soh"],
        "predicted_condition": row["predicted_condition"],
        "soh_band": row["soh_band"],
        "status": row["status"],
        "inputs": json.loads(row["inputs_json"]),
        "warnings": json.loads(row["warnings_json"]),
        "guidance": json.loads(row["guidance_json"]),
        "limitations": json.loads(row["limitations_json"]),
    }


def list_assessments(user_id: int | None, path: Path | None = None) -> list[dict]:
    init_db(path)
    query = """
        SELECT a.*, b.battery_identifier
        FROM assessments AS a
        JOIN battery_records AS b ON b.id = a.battery_record_id
        WHERE ((? IS NULL AND a.user_id IS NULL) OR a.user_id = ?)
        ORDER BY a.assessed_at DESC, a.id DESC
    """
    with connect(path) as connection:
        rows = connection.execute(query, (user_id, user_id)).fetchall()
    return [_row_to_assessment(row) for row in rows]


def get_assessment(assessment_id: int, user_id: int | None, path: Path | None = None) -> dict | None:
    init_db(path)
    query = """
        SELECT a.*, b.battery_identifier
        FROM assessments AS a
        JOIN battery_records AS b ON b.id = a.battery_record_id
        WHERE a.id = ?
          AND ((? IS NULL AND a.user_id IS NULL) OR a.user_id = ?)
    """
    with connect(path) as connection:
        row = connection.execute(query, (assessment_id, user_id, user_id)).fetchone()
    if row is None:
        return None
    return _row_to_assessment(row)


def dashboard_summary(user_id: int | None, path: Path | None = None) -> dict:
    records = list_assessments(user_id, path)
    predicted = [row for row in records if row["status"] == "predicted" and row["predicted_soh"] is not None]
    conditions = {"Good": 0, "Fair": 0, "Poor": 0}
    for row in predicted:
        label = row["predicted_condition"]
        if label in conditions:
            conditions[label] += 1
    alerts = [
        row
        for row in predicted
        if row["predicted_condition"] == "Poor" or (row["predicted_soh"] is not None and row["predicted_soh"] <= 70)
    ]
    battery_ids = {row["battery_record_id"] for row in records}
    average = None
    if predicted:
        average = float(sum(row["predicted_soh"] for row in predicted) / len(predicted))
    return {
        "battery_count": len(battery_ids),
        "assessment_count": len(records),
        "predicted_count": len(predicted),
        "average_predicted_soh": average,
        "conditions": conditions,
        "recent": records[:8],
        "alerts": alerts[:8],
    }

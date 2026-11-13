"""Database access layer supporting SQLite and MongoDB for lithium-ion battery records.

Reads configuration from .env:
- DB_TYPE: 'mongodb' or 'sqlite' (defaults to 'mongodb' when configured, falling back to 'sqlite')
- MONGODB_URI / MONGO_URI: MongoDB connection URI
- MONGODB_DB_NAME / MONGO_DB_NAME: Database name (default: 'powerpulse_db')
- MONGODB_USERS_COLLECTION: Users collection name (default: 'users')
- MONGODB_BATTERIES_COLLECTION: Batteries collection name (default: 'batteries')
- MONGODB_ASSESSMENTS_COLLECTION: Assessments collection name (default: 'assessments')
- MONGODB_TIMEOUT_MS: Server selection timeout in milliseconds (default: 5000)
- SQLITE_DB_PATH: Path to SQLite fallback database (default: 'database/powerpulse.db')
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from database.schema import INDEXES, STATEMENTS
from ml.config import DATABASE_PATH, EOL_THRESHOLD

logger = logging.getLogger("powerpulse.database")
PBKDF2_ROUNDS = 200_000

# Cache for MongoDB client and database instance
_MONGO_CLIENT = None
_MONGO_DB = None
_MONGO_INITIALIZED = False
_MONGO_WARNING_LOGGED = False


def _load_env_file() -> None:
    """Load variables from .env if present, supporting python-dotenv or fallback parser."""
    try:
        from dotenv import load_dotenv  # type: ignore

        env_candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[1] / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ]
        for candidate in env_candidates:
            if candidate.is_file():
                load_dotenv(candidate, override=False)
                return
    except ImportError:
        pass

    # Native fallback .env parser when python-dotenv is not installed
    env_candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for candidate in env_candidates:
        if candidate.is_file():
            try:
                with open(candidate, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip("'\"")
                            if key and key not in os.environ:
                                os.environ[key] = val
                return
            except Exception:
                pass


# Load environment variables on module import
_load_env_file()


def get_db_config() -> dict[str, Any]:
    """Retrieve database configuration settings from environment variables."""
    _load_env_file()
    db_type = os.environ.get("DB_TYPE", "mongodb").strip().lower()
    mongo_uri = (os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI") or "").strip()
    db_name = (os.environ.get("MONGODB_DB_NAME") or os.environ.get("MONGO_DB_NAME") or "powerpulse_db").strip()
    users_col = os.environ.get("MONGODB_USERS_COLLECTION", "users").strip()
    batt_col = os.environ.get("MONGODB_BATTERIES_COLLECTION", "batteries").strip()
    assess_col = os.environ.get("MONGODB_ASSESSMENTS_COLLECTION", "assessments").strip()
    try:
        timeout_ms = int(os.environ.get("MONGODB_TIMEOUT_MS", "5000"))
    except ValueError:
        timeout_ms = 5000
    sqlite_path_str = os.environ.get("SQLITE_DB_PATH", "").strip()
    sqlite_path = Path(sqlite_path_str) if sqlite_path_str else DATABASE_PATH

    return {
        "db_type": db_type,
        "mongo_uri": mongo_uri,
        "mongo_db_name": db_name,
        "users_collection": users_col,
        "batteries_collection": batt_col,
        "assessments_collection": assess_col,
        "timeout_ms": timeout_ms,
        "sqlite_path": sqlite_path,
    }


def utc_now() -> str:
    """Return the current UTC timestamp formatted as ISO 8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# MongoDB Backend Connection & Helpers
# ---------------------------------------------------------------------------

def _get_mongodb():
    """Attempt to establish or return an active MongoDB database instance.

    Returns None if MongoDB is disabled, pymongo is missing, or connection fails.
    """
    global _MONGO_CLIENT, _MONGO_DB, _MONGO_WARNING_LOGGED
    cfg = get_db_config()

    if cfg["db_type"] != "mongodb":
        return None

    if not cfg["mongo_uri"]:
        if not _MONGO_WARNING_LOGGED:
            logger.info("DB_TYPE is 'mongodb' but MONGODB_URI is not set. Operating with local SQLite fallback.")
            _MONGO_WARNING_LOGGED = True
        return None

    if _MONGO_DB is not None:
        return _MONGO_DB

    try:
        import pymongo  # type: ignore
    except ImportError:
        if not _MONGO_WARNING_LOGGED:
            logger.warning("pymongo package not installed. Operating with local SQLite fallback.")
            _MONGO_WARNING_LOGGED = True
        return None

    try:
        client = pymongo.MongoClient(
            cfg["mongo_uri"],
            serverSelectionTimeoutMS=cfg["timeout_ms"],
            connectTimeoutMS=cfg["timeout_ms"],
        )
        # Fast ping check to verify connectivity
        client.admin.command("ping")
        _MONGO_CLIENT = client
        _MONGO_DB = client[cfg["mongo_db_name"]]
        _init_mongodb_indexes(_MONGO_DB, cfg)
        return _MONGO_DB
    except Exception as exc:
        if not _MONGO_WARNING_LOGGED:
            logger.warning("Could not connect to MongoDB (%s). Operating with local SQLite fallback.", exc)
            _MONGO_WARNING_LOGGED = True
        return None


def _init_mongodb_indexes(db, cfg: dict[str, Any]) -> None:
    """Create indexes in MongoDB collections for optimal query performance."""
    global _MONGO_INITIALIZED
    if _MONGO_INITIALIZED:
        return
    try:
        import pymongo  # type: ignore

        users_col = db[cfg["users_collection"]]
        users_col.create_index([("username", pymongo.ASCENDING)], unique=True)
        users_col.create_index([("id", pymongo.ASCENDING)], unique=True)

        batt_col = db[cfg["batteries_collection"]]
        batt_col.create_index(
            [
                ("battery_id", pymongo.ASCENDING),
                ("user_id", pymongo.ASCENDING),
            ],
            name="idx_battery_id_user",
        )
        batt_col.create_index(
            [
                ("battery_identifier", pymongo.ASCENDING),
                ("user_id", pymongo.ASCENDING),
            ],
            name="idx_battery_identity",
        )
        batt_col.create_index([("id", pymongo.ASCENDING)], unique=True)

        assess_col = db[cfg["assessments_collection"]]
        assess_col.create_index([("id", pymongo.ASCENDING)], unique=True)
        assess_col.create_index([("battery_record_id", pymongo.ASCENDING)])
        assess_col.create_index([("user_id", pymongo.ASCENDING)])
        assess_col.create_index([("assessed_at", pymongo.DESCENDING)])

        alerts_col = db["battery_alerts"]
        alerts_col.create_index(
            [
                ("battery_id", pymongo.ASCENDING),
                ("alert_type", pymongo.ASCENDING),
                ("relevant_expiry_date", pymongo.ASCENDING),
                ("user_id", pymongo.ASCENDING),
            ],
            name="idx_battery_alerts_lookup",
        )
        alerts_col.create_index(
            [("user_id", pymongo.ASCENDING), ("alert_status", pymongo.ASCENDING)],
            name="idx_alerts_user_status",
        )
        alerts_col.create_index([("generated_at", pymongo.DESCENDING)], name="idx_alerts_time")
        _MONGO_INITIALIZED = True
    except Exception as exc:
        logger.warning("Failed to initialize MongoDB indexes: %s", exc)


def _get_next_sequence(db, sequence_name: str) -> int:
    """Generate sequential auto-incrementing integer IDs in MongoDB."""
    try:
        import pymongo  # type: ignore

        ret = db["counters"].find_one_and_update(
            {"_id": sequence_name},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=pymongo.ReturnDocument.AFTER,
        )
        return int(ret["seq"])
    except Exception:
        col_name = sequence_name
        doc = db[col_name].find_one(sort=[("id", -1)])
        return 1 if not doc or "id" not in doc else int(doc["id"]) + 1


def _is_mongodb_active(path: Path | None) -> bool:
    """Determine whether MongoDB should serve the current request.

    If an explicit SQLite path is passed (e.g. by unit tests or CLI), SQLite is used.
    """
    if path is not None:
        return False
    return _get_mongodb() is not None


def get_active_db_type(path: Path | None = None) -> str:
    """Return the name of the currently active database backend ('mongodb' or 'sqlite')."""
    return "mongodb" if _is_mongodb_active(path) else "sqlite"


# ---------------------------------------------------------------------------
# SQLite Backend Connection & Helpers
# ---------------------------------------------------------------------------

def connect(path: Path | None = None) -> sqlite3.Connection:
    """Create a connection to the SQLite database."""
    cfg = get_db_config()
    database = Path(path) if path is not None else cfg["sqlite_path"]
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _migrate_sqlite_schema(connection: sqlite3.Connection) -> None:
    """Add new lifecycle and warranty columns to battery_records if migrating an existing DB."""
    try:
        cursor = connection.execute("PRAGMA table_info(battery_records)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        new_columns = [
            ("battery_id", "TEXT"),
            ("manufacturer", "TEXT"),
            ("battery_model", "TEXT"),
            ("purchase_date", "TEXT"),
            ("warranty_years", "INTEGER DEFAULT 0"),
            ("warranty_months", "INTEGER DEFAULT 0"),
            ("warranty_expiry_date", "TEXT"),
            ("warranty_status", "TEXT"),
            ("expected_lifespan_years", "REAL"),
            ("warranty_alert_days", "INTEGER DEFAULT 30"),
        ]
        for col_name, col_type in new_columns:
            if col_name not in existing_cols:
                connection.execute(f"ALTER TABLE battery_records ADD COLUMN {col_name} {col_type}")
        connection.commit()
    except Exception as exc:
        logger.warning("Could not auto-migrate battery_records table: %s", exc)


def _init_sqlite(path: Path | None = None) -> None:
    with connect(path) as connection:
        for statement in STATEMENTS:
            connection.execute(statement)
        for statement in INDEXES:
            connection.execute(statement)
        _migrate_sqlite_schema(connection)
        connection.commit()


# ---------------------------------------------------------------------------
# Universal DB Initialization
# ---------------------------------------------------------------------------

def init_db(path: Path | None = None) -> None:
    """Initialize database schemas, indexes, and tables for SQLite or MongoDB."""
    if _is_mongodb_active(path):
        db = _get_mongodb()
        cfg = get_db_config()
        _init_mongodb_indexes(db, cfg)
    else:
        _init_sqlite(path)


# ---------------------------------------------------------------------------
# Security & Password Utilities
# ---------------------------------------------------------------------------

def hash_password(password: str, salt: bytes | None = None) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256 with 200,000 rounds."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Verify a plain password against the stored salt:digest string."""
    try:
        salt_hex, digest_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    candidate = hash_password(password, salt)
    return secrets.compare_digest(candidate, f"{salt.hex()}:{digest_hex}")


# ---------------------------------------------------------------------------
# User Authentication & Management
# ---------------------------------------------------------------------------

def create_user(username: str, password: str, path: Path | None = None) -> int:
    """Register a new user, returning the generated numeric user ID."""
    username = username.strip()
    if len(username) < 3 or len(username) > 32 or not username.replace("_", "").isalnum():
        raise ValueError("Username must be 3 to 32 characters and use letters, numbers, or underscore.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")

    init_db(path)

    if _is_mongodb_active(path):
        db = _get_mongodb()
        cfg = get_db_config()
        users_col = db[cfg["users_collection"]]
        if users_col.find_one({"username": username}) is not None:
            raise ValueError("That username is already registered.")
        user_id = _get_next_sequence(db, "users")
        doc = {
            "id": user_id,
            "username": username,
            "password_hash": hash_password(password),
            "created_at": utc_now(),
        }
        users_col.insert_one(doc)
        return user_id

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
    """Authenticate a user by username and password. Returns user dict or None."""
    init_db(path)

    if _is_mongodb_active(path):
        db = _get_mongodb()
        cfg = get_db_config()
        doc = db[cfg["users_collection"]].find_one({"username": username.strip()})
        if doc is None or not verify_password(password, doc.get("password_hash", "")):
            return None
        return {"id": int(doc["id"]), "username": doc["username"]}

    with connect(path) as connection:
        row = connection.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return {"id": int(row["id"]), "username": row["username"]}


# ---------------------------------------------------------------------------
# Battery & Warranty Record Operations
# ---------------------------------------------------------------------------

def _row_to_battery(row: sqlite3.Row) -> dict[str, Any]:
    """Convert an SQLite Row into a battery record dictionary."""
    keys = row.keys()
    b_id = row["battery_id"] if "battery_id" in keys and row["battery_id"] else row["battery_identifier"]
    return {
        "id": int(row["id"]),
        "user_id": row["user_id"],
        "battery_id": b_id,
        "battery_identifier": row["battery_identifier"],
        "category": row["category"],
        "created_at": row["created_at"],
        "manufacturer": row["manufacturer"] if "manufacturer" in keys else None,
        "battery_model": row["battery_model"] if "battery_model" in keys else None,
        "purchase_date": row["purchase_date"] if "purchase_date" in keys else None,
        "warranty_years": row["warranty_years"] if "warranty_years" in keys else 0,
        "warranty_months": row["warranty_months"] if "warranty_months" in keys else 0,
        "warranty_expiry_date": row["warranty_expiry_date"] if "warranty_expiry_date" in keys else None,
        "warranty_status": row["warranty_status"] if "warranty_status" in keys else "UNKNOWN",
        "expected_lifespan_years": row["expected_lifespan_years"] if "expected_lifespan_years" in keys else None,
        "warranty_alert_days": row["warranty_alert_days"] if "warranty_alert_days" in keys and row["warranty_alert_days"] is not None else 30,
    }


def _mongo_doc_to_battery(doc: dict[str, Any]) -> dict[str, Any]:
    """Normalize a MongoDB document into a battery record dictionary."""
    b_id = doc.get("battery_id") or doc.get("battery_identifier", "")
    return {
        "id": int(doc.get("id", 1)),
        "user_id": doc.get("user_id"),
        "battery_id": b_id,
        "battery_identifier": doc.get("battery_identifier") or b_id,
        "category": doc.get("category", "lithium_ion"),
        "created_at": doc.get("created_at", ""),
        "manufacturer": doc.get("manufacturer"),
        "battery_model": doc.get("battery_model"),
        "purchase_date": doc.get("purchase_date"),
        "warranty_years": doc.get("warranty_years", 0),
        "warranty_months": doc.get("warranty_months", 0),
        "warranty_expiry_date": doc.get("warranty_expiry_date"),
        "warranty_status": doc.get("warranty_status", "UNKNOWN"),
        "expected_lifespan_years": doc.get("expected_lifespan_years"),
        "warranty_alert_days": doc.get("warranty_alert_days", 30),
    }


def add_battery(
    identifier: str,
    category: str = "lithium_ion",
    user_id: int | None = None,
    purchase_date: str | None = None,
    warranty_months: int | None = None,
    expected_lifespan_years: float | None = None,
    manufacturer: str | None = None,
    warranty_alert_days: int | None = None,
    battery_model: str | None = None,
    warranty_years: int | None = None,
    path: Path | None = None,
) -> int:
    """Register or look up a battery record, updating metadata if provided."""
    clean_id = identifier.strip()
    if not clean_id or len(clean_id) > 64:
        raise ValueError("Battery identifier is required and must be 64 characters or fewer.")
    category = "lithium_ion"
    init_db(path)

    w_years = warranty_years or 0
    w_months = warranty_months or 0

    if _is_mongodb_active(path):
        db = _get_mongodb()
        cfg = get_db_config()
        batt_col = db[cfg["batteries_collection"]]
        query: dict[str, Any] = {
            "$or": [{"battery_id": clean_id}, {"battery_identifier": clean_id}],
            "user_id": user_id,
        }
        existing = batt_col.find_one(query)
        if existing is not None:
            update_fields: dict[str, Any] = {}
            if purchase_date is not None:
                update_fields["purchase_date"] = purchase_date
            if warranty_months is not None:
                update_fields["warranty_months"] = warranty_months
            if warranty_years is not None:
                update_fields["warranty_years"] = warranty_years
            if expected_lifespan_years is not None:
                update_fields["expected_lifespan_years"] = expected_lifespan_years
            if manufacturer is not None:
                update_fields["manufacturer"] = manufacturer
            if battery_model is not None:
                update_fields["battery_model"] = battery_model
            if warranty_alert_days is not None:
                update_fields["warranty_alert_days"] = warranty_alert_days
            if update_fields:
                update_fields["updated_at"] = utc_now()
                batt_col.update_one({"_id": existing["_id"]}, {"$set": update_fields})
            return int(existing.get("id", 1))

        battery_id = _get_next_sequence(db, "battery_records")
        doc = {
            "id": battery_id,
            "user_id": user_id,
            "battery_id": clean_id,
            "battery_identifier": clean_id,
            "category": category,
            "created_at": utc_now(),
            "manufacturer": manufacturer,
            "battery_model": battery_model,
            "purchase_date": purchase_date,
            "warranty_years": w_years,
            "warranty_months": w_months,
            "expected_lifespan_years": expected_lifespan_years,
            "warranty_alert_days": warranty_alert_days or 30,
        }
        batt_col.insert_one(doc)
        return battery_id

    with connect(path) as connection:
        existing = connection.execute(
            """
            SELECT id FROM battery_records
            WHERE (battery_identifier = ? OR battery_id = ?) AND category = ?
              AND ((user_id IS NULL AND ? IS NULL) OR user_id = ?)
            """,
            (clean_id, clean_id, category, user_id, user_id),
        ).fetchone()

        if existing is not None:
            battery_id = int(existing["id"])
            updates = []
            params = []
            if purchase_date is not None:
                updates.append("purchase_date = ?")
                params.append(purchase_date)
            if warranty_months is not None:
                updates.append("warranty_months = ?")
                params.append(warranty_months)
            if warranty_years is not None:
                updates.append("warranty_years = ?")
                params.append(warranty_years)
            if expected_lifespan_years is not None:
                updates.append("expected_lifespan_years = ?")
                params.append(expected_lifespan_years)
            if manufacturer is not None:
                updates.append("manufacturer = ?")
                params.append(manufacturer)
            if battery_model is not None:
                updates.append("battery_model = ?")
                params.append(battery_model)
            if warranty_alert_days is not None:
                updates.append("warranty_alert_days = ?")
                params.append(warranty_alert_days)

            if updates:
                params.append(battery_id)
                connection.execute(
                    f"UPDATE battery_records SET {', '.join(updates)} WHERE id = ?",
                    params,
                )
                connection.commit()
            return battery_id

        cursor = connection.execute(
            """
            INSERT INTO battery_records (
                user_id, battery_identifier, battery_id, category, created_at,
                manufacturer, battery_model, purchase_date, warranty_years,
                warranty_months, expected_lifespan_years, warranty_alert_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                clean_id,
                clean_id,
                category,
                utc_now(),
                manufacturer,
                battery_model,
                purchase_date,
                w_years,
                w_months,
                expected_lifespan_years,
                warranty_alert_days or 30,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def register_battery_warranty(
    battery_id: str,
    manufacturer: str,
    battery_model: str,
    purchase_date: str | date,
    warranty_years: int,
    warranty_months: int = 0,
    expected_lifespan_years: float | None = None,
    user_id: int | None = None,
    path: Path | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Register or update battery warranty details, enforcing strict validation."""
    from utils.warranty import validate_warranty_inputs

    validated = validate_warranty_inputs(
        battery_id=battery_id,
        manufacturer=manufacturer,
        battery_model=battery_model,
        purchase_date=purchase_date,
        warranty_years=warranty_years,
        warranty_months=warranty_months,
        as_of=as_of,
    )

    clean_id = validated["battery_id"]
    init_db(path)

    if _is_mongodb_active(path):
        db = _get_mongodb()
        cfg = get_db_config()
        batt_col = db[cfg["batteries_collection"]]
        query = {
            "$or": [{"battery_id": clean_id}, {"battery_identifier": clean_id}],
            "user_id": user_id,
        }
        existing = batt_col.find_one(query)

        doc_data = {
            "battery_id": clean_id,
            "battery_identifier": clean_id,
            "manufacturer": validated["manufacturer"],
            "battery_model": validated["battery_model"],
            "purchase_date": validated["purchase_date"],
            "warranty_years": validated["warranty_years"],
            "warranty_months": validated["warranty_months"],
            "warranty_expiry_date": validated["warranty_expiry_date"],
            "warranty_status": validated["warranty_status"],
            "expected_lifespan_years": expected_lifespan_years,
            "user_id": user_id,
            "category": "lithium_ion",
            "updated_at": utc_now(),
        }

        if existing:
            batt_col.update_one({"_id": existing["_id"]}, {"$set": doc_data})
        else:
            doc_data["id"] = _get_next_sequence(db, "battery_records")
            doc_data["created_at"] = utc_now()
            batt_col.insert_one(doc_data)

        res = get_battery_warranty(clean_id, user_id=user_id, as_of=as_of, path=path)
        return res if res is not None else validated

    with connect(path) as connection:
        existing = connection.execute(
            """
            SELECT id FROM battery_records
            WHERE (battery_identifier = ? OR battery_id = ?)
              AND ((user_id IS NULL AND ? IS NULL) OR user_id = ?)
            """,
            (clean_id, clean_id, user_id, user_id),
        ).fetchone()

        if existing:
            rec_id = int(existing["id"])
            connection.execute(
                """
                UPDATE battery_records SET
                    battery_id = ?,
                    battery_identifier = ?,
                    manufacturer = ?,
                    battery_model = ?,
                    purchase_date = ?,
                    warranty_years = ?,
                    warranty_months = ?,
                    warranty_expiry_date = ?,
                    warranty_status = ?,
                    expected_lifespan_years = ?
                WHERE id = ?
                """,
                (
                    clean_id,
                    clean_id,
                    validated["manufacturer"],
                    validated["battery_model"],
                    validated["purchase_date"],
                    validated["warranty_years"],
                    validated["warranty_months"],
                    validated["warranty_expiry_date"],
                    validated["warranty_status"],
                    expected_lifespan_years,
                    rec_id,
                ),
            )
        else:
            connection.execute(
                """
                INSERT INTO battery_records (
                    user_id, battery_identifier, battery_id, category, created_at,
                    manufacturer, battery_model, purchase_date, warranty_years,
                    warranty_months, warranty_expiry_date, warranty_status, expected_lifespan_years
                ) VALUES (?, ?, ?, 'lithium_ion', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    clean_id,
                    clean_id,
                    utc_now(),
                    validated["manufacturer"],
                    validated["battery_model"],
                    validated["purchase_date"],
                    validated["warranty_years"],
                    validated["warranty_months"],
                    validated["warranty_expiry_date"],
                    validated["warranty_status"],
                    expected_lifespan_years,
                ),
            )
        connection.commit()

    res = get_battery_warranty(clean_id, user_id=user_id, as_of=as_of, path=path)
    return res if res is not None else validated


def get_battery_warranty(
    battery_id: str,
    user_id: int | None = None,
    as_of: date | None = None,
    path: Path | None = None,
) -> dict[str, Any] | None:
    """Retrieve battery warranty details with freshly re-evaluated status."""
    from utils.warranty import get_warranty_status

    clean_id = (battery_id or "").strip()
    if not clean_id:
        return None
    init_db(path)

    record = None
    if _is_mongodb_active(path):
        db = _get_mongodb()
        cfg = get_db_config()
        query = {
            "$or": [{"battery_id": clean_id}, {"battery_identifier": clean_id}],
            "user_id": user_id,
        }
        doc = db[cfg["batteries_collection"]].find_one(query)
        if doc is None:
            return None
        record = _mongo_doc_to_battery(doc)
    else:
        with connect(path) as connection:
            row = connection.execute(
                """
                SELECT * FROM battery_records
                WHERE (battery_identifier = ? OR battery_id = ?)
                  AND ((user_id IS NULL AND ? IS NULL) OR user_id = ?)
                """,
                (clean_id, clean_id, user_id, user_id),
            ).fetchone()
            if row is None:
                return None
            record = _row_to_battery(row)

    if record is None:
        return None

    # Dynamically re-evaluate warranty status against target evaluation date
    p_date = record.get("purchase_date")
    w_years = record.get("warranty_years") or 0
    w_months = record.get("warranty_months") or 0
    if p_date and (w_years > 0 or w_months > 0):
        try:
            status, exp_date, days_left = get_warranty_status(
                purchase_date=p_date,
                warranty_years=w_years,
                warranty_months=w_months,
                as_of=as_of,
            )
            record["warranty_status"] = status
            record["warranty_expiry_date"] = exp_date.isoformat()
            record["days_remaining"] = days_left
        except Exception:
            pass

    return record


def list_user_batteries(
    user_id: int | None = None,
    as_of: date | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return all batteries registered to this user account with fresh warranty status."""
    from utils.warranty import get_warranty_status

    init_db(path)
    records = []
    if _is_mongodb_active(path):
        db = _get_mongodb()
        cfg = get_db_config()
        cursor = db[cfg["batteries_collection"]].find({"user_id": user_id}).sort([("battery_id", 1)])
        records = [_mongo_doc_to_battery(doc) for doc in cursor]
    else:
        with connect(path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM battery_records
                WHERE ((user_id IS NULL AND ? IS NULL) OR user_id = ?)
                ORDER BY battery_identifier ASC
                """,
                (user_id, user_id),
            ).fetchall()
            records = [_row_to_battery(row) for row in rows]

    for r in records:
        p_date = r.get("purchase_date")
        w_years = r.get("warranty_years") or 0
        w_months = r.get("warranty_months") or 0
        if p_date and (w_years > 0 or w_months > 0):
            try:
                status, exp_date, days_left = get_warranty_status(
                    purchase_date=p_date,
                    warranty_years=w_years,
                    warranty_months=w_months,
                    as_of=as_of,
                )
                r["warranty_status"] = status
                r["warranty_expiry_date"] = exp_date.isoformat()
                r["days_remaining"] = days_left
            except Exception:
                pass

    return records


# Legacy / Lifecycle support aliases
get_battery = get_battery_warranty
list_batteries = list_user_batteries


def update_battery_details(
    identifier: str,
    details: dict[str, Any],
    user_id: int | None = None,
    path: Path | None = None,
) -> bool:
    """Update lifecycle and warranty attributes for an existing battery record."""
    register_battery_warranty(
        battery_id=identifier,
        manufacturer=details.get("manufacturer") or "Generic",
        battery_model=details.get("battery_model") or "Generic",
        purchase_date=details.get("purchase_date") or utc_now()[:10],
        warranty_years=details.get("warranty_years") or 1,
        warranty_months=details.get("warranty_months") or 0,
        expected_lifespan_years=details.get("expected_lifespan_years"),
        user_id=user_id,
        path=path,
    )
    return True


# ---------------------------------------------------------------------------
# Assessment Operations
# ---------------------------------------------------------------------------

def add_assessment(payload: dict, path: Path | None = None) -> int:
    """Store a lithium-ion assessment record, returning its numeric assessment ID."""
    init_db(path)
    clean_id = payload["battery_identifier"].strip()
    category = payload.get("category", "lithium_ion")
    battery_id = add_battery(
        identifier=clean_id,
        category=category,
        user_id=payload.get("user_id"),
        purchase_date=payload.get("purchase_date"),
        warranty_months=payload.get("warranty_months"),
        expected_lifespan_years=payload.get("expected_lifespan_years"),
        manufacturer=payload.get("manufacturer"),
        warranty_alert_days=payload.get("warranty_alert_days"),
        path=path,
    )
    limitations = payload.get("limitations") or []
    if isinstance(limitations, str):
        limitations = [limitations]

    if _is_mongodb_active(path):
        db = _get_mongodb()
        cfg = get_db_config()
        assessment_id = _get_next_sequence(db, "assessments")
        doc = {
            "id": assessment_id,
            "battery_record_id": battery_id,
            "user_id": payload.get("user_id"),
            "battery_identifier": clean_id,
            "battery_id": clean_id,
            "assessed_at": payload.get("assessed_at") or utc_now(),
            "category": category,
            "model_version": payload.get("model_version"),
            "predicted_soh": payload.get("predicted_soh"),
            "predicted_condition": payload.get("predicted_condition"),
            "soh_band": payload.get("soh_band"),
            "status": payload.get("status", "predicted"),
            "inputs": payload.get("inputs") or {},
            "warnings": payload.get("warnings") or [],
            "guidance": payload.get("guidance") or [],
            "limitations": limitations,
        }
        db[cfg["assessments_collection"]].insert_one(doc)
        return assessment_id

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
    """Convert an SQLite Row to a standard assessment dictionary with joined battery attributes."""
    keys = row.keys()
    b_id = row["battery_id"] if "battery_id" in keys and row["battery_id"] else row["battery_identifier"]
    return {
        "id": int(row["id"]),
        "battery_record_id": int(row["battery_record_id"]),
        "user_id": row["user_id"],
        "battery_identifier": row["battery_identifier"],
        "battery_id": b_id,
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
        "purchase_date": row["purchase_date"] if "purchase_date" in keys else None,
        "warranty_years": row["warranty_years"] if "warranty_years" in keys else 0,
        "warranty_months": row["warranty_months"] if "warranty_months" in keys else 0,
        "warranty_expiry_date": row["warranty_expiry_date"] if "warranty_expiry_date" in keys else None,
        "warranty_status": row["warranty_status"] if "warranty_status" in keys else None,
        "expected_lifespan_years": row["expected_lifespan_years"] if "expected_lifespan_years" in keys else None,
        "manufacturer": row["manufacturer"] if "manufacturer" in keys else None,
        "battery_model": row["battery_model"] if "battery_model" in keys else None,
    }


def _mongo_doc_to_assessment(doc: dict) -> dict:
    """Normalize a MongoDB document to standard assessment dictionary."""
    b_id = doc.get("battery_id") or doc.get("battery_identifier", "")
    return {
        "id": int(doc["id"]),
        "battery_record_id": int(doc["battery_record_id"]),
        "user_id": doc.get("user_id"),
        "battery_identifier": doc.get("battery_identifier", b_id),
        "battery_id": b_id,
        "assessed_at": doc.get("assessed_at", ""),
        "category": doc.get("category", "lithium_ion"),
        "model_version": doc.get("model_version"),
        "predicted_soh": doc.get("predicted_soh"),
        "predicted_condition": doc.get("predicted_condition"),
        "soh_band": doc.get("soh_band"),
        "status": doc.get("status", "predicted"),
        "inputs": doc.get("inputs") or {},
        "warnings": doc.get("warnings") or [],
        "guidance": doc.get("guidance") or [],
        "limitations": doc.get("limitations") or [],
        "purchase_date": doc.get("purchase_date"),
        "warranty_years": doc.get("warranty_years", 0),
        "warranty_months": doc.get("warranty_months", 0),
        "warranty_expiry_date": doc.get("warranty_expiry_date"),
        "warranty_status": doc.get("warranty_status"),
        "expected_lifespan_years": doc.get("expected_lifespan_years"),
        "manufacturer": doc.get("manufacturer"),
        "battery_model": doc.get("battery_model"),
    }


def list_assessments(user_id: int | None, path: Path | None = None) -> list[dict]:
    """Retrieve all assessments visible to user_id, sorted newest first."""
    init_db(path)

    if _is_mongodb_active(path):
        db = _get_mongodb()
        cfg = get_db_config()
        query: dict[str, Any] = {"user_id": user_id} if user_id is not None else {"user_id": None}
        cursor = (
            db[cfg["assessments_collection"]]
            .find(query)
            .sort([("assessed_at", -1), ("id", -1)])
        )
        return [_mongo_doc_to_assessment(doc) for doc in cursor]

    query = """
        SELECT a.*, b.battery_identifier, b.battery_id, b.purchase_date, b.warranty_years,
               b.warranty_months, b.warranty_expiry_date, b.warranty_status,
               b.expected_lifespan_years, b.manufacturer, b.battery_model
        FROM assessments AS a
        JOIN battery_records AS b ON b.id = a.battery_record_id
        WHERE ((? IS NULL AND a.user_id IS NULL) OR a.user_id = ?)
        ORDER BY a.assessed_at DESC, a.id DESC
    """
    with connect(path) as connection:
        rows = connection.execute(query, (user_id, user_id)).fetchall()
    return [_row_to_assessment(row) for row in rows]


def get_assessment(assessment_id: int, user_id: int | None, path: Path | None = None) -> dict | None:
    """Retrieve a specific assessment by ID, scoped to user_id."""
    init_db(path)

    if _is_mongodb_active(path):
        db = _get_mongodb()
        cfg = get_db_config()
        query: dict[str, Any] = {"id": int(assessment_id)}
        if user_id is not None:
            query["user_id"] = user_id
        else:
            query["user_id"] = None
        doc = db[cfg["assessments_collection"]].find_one(query)
        return _mongo_doc_to_assessment(doc) if doc is not None else None

    query = """
        SELECT a.*, b.battery_identifier, b.battery_id, b.purchase_date, b.warranty_years,
               b.warranty_months, b.warranty_expiry_date, b.warranty_status,
               b.expected_lifespan_years, b.manufacturer, b.battery_model
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
    """Calculate aggregated health metrics, alerts, and warranty status for the dashboard."""
    records = list_assessments(user_id, path)
    batteries = list_user_batteries(user_id, path=path)

    predicted = [row for row in records if row["status"] == "predicted" and row["predicted_soh"] is not None]
    conditions = {"Good": 0, "Fair": 0, "Poor": 0}
    for row in predicted:
        label = row["predicted_condition"]
        if label in conditions:
            conditions[label] += 1
    alerts = [
        row
        for row in predicted
        if row["predicted_condition"] == "Poor" or (row["predicted_soh"] is not None and row["predicted_soh"] <= EOL_THRESHOLD)
    ]

    average = None
    if predicted:
        average = float(sum(row["predicted_soh"] for row in predicted) / len(predicted))

    active_warranties = sum(1 for b in batteries if b.get("warranty_status") == "ACTIVE")
    expired_warranties = sum(1 for b in batteries if b.get("warranty_status") == "EXPIRED")

    return {
        "battery_count": len(batteries),
        "assessment_count": len(records),
        "predicted_count": len(predicted),
        "average_predicted_soh": average,
        "conditions": conditions,
        "recent": records[:8],
        "alerts": alerts[:8],
        "batteries": batteries,
        "active_warranties": active_warranties,
        "expired_warranties": expired_warranties,
    }


# ---------------------------------------------------------------------------
# Battery Lifecycle Alerts Operations (MongoDB & fallback)
# ---------------------------------------------------------------------------

def save_battery_alert(alert_doc: dict[str, Any], path: Path | None = None) -> int | str:
    """Save an alert record into the database."""
    init_db(path)
    clean_doc = {
        "battery_id": alert_doc["battery_id"],
        "user_id": alert_doc.get("user_id"),
        "alert_type": alert_doc["alert_type"],
        "alert_message": alert_doc["alert_message"],
        "category": alert_doc.get("category", "lifecycle"),
        "severity": alert_doc.get("severity", "yellow"),
        "generated_at": alert_doc.get("generated_at") or utc_now(),
        "alert_status": alert_doc.get("alert_status", "Unread"),
        "relevant_expiry_date": alert_doc.get("relevant_expiry_date"),
        "acknowledged_at": alert_doc.get("acknowledged_at"),
    }

    if _is_mongodb_active(path):
        db = _get_mongodb()
        alerts_col = db["battery_alerts"]
        alert_id = _get_next_sequence(db, "battery_alerts")
        clean_doc["id"] = alert_id
        alerts_col.insert_one(clean_doc)
        return alert_id

    with connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO battery_alerts (
                battery_id, user_id, alert_type, alert_message,
                category, severity, generated_at, alert_status,
                relevant_expiry_date, acknowledged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_doc["battery_id"],
                clean_doc["user_id"],
                clean_doc["alert_type"],
                clean_doc["alert_message"],
                clean_doc["category"],
                clean_doc["severity"],
                clean_doc["generated_at"],
                clean_doc["alert_status"],
                clean_doc["relevant_expiry_date"],
                clean_doc["acknowledged_at"],
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def list_battery_alerts(
    user_id: int | None = None,
    battery_id: str | None = None,
    status: str | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Retrieve alerts filtered by user, battery, and/or status, newest first."""
    init_db(path)

    if _is_mongodb_active(path):
        db = _get_mongodb()
        query: dict[str, Any] = {"user_id": user_id} if user_id is not None else {"user_id": None}
        if battery_id:
            query["battery_id"] = battery_id
        if status:
            query["alert_status"] = status

        cursor = db["battery_alerts"].find(query).sort([("generated_at", -1), ("id", -1)])
        results = []
        for doc in cursor:
            results.append({
                "id": doc.get("id") or str(doc.get("_id")),
                "battery_id": doc["battery_id"],
                "user_id": doc.get("user_id"),
                "alert_type": doc["alert_type"],
                "alert_message": doc["alert_message"],
                "category": doc.get("category", "lifecycle"),
                "severity": doc.get("severity", "yellow"),
                "generated_at": doc.get("generated_at", ""),
                "alert_status": doc.get("alert_status", "Unread"),
                "relevant_expiry_date": doc.get("relevant_expiry_date"),
                "acknowledged_at": doc.get("acknowledged_at"),
            })
        return results

    query_parts = ["((? IS NULL AND user_id IS NULL) OR user_id = ?)"]
    params: list[Any] = [user_id, user_id]

    if battery_id:
        query_parts.append("battery_id = ?")
        params.append(battery_id)
    if status:
        query_parts.append("alert_status = ?")
        params.append(status)

    sql = f"""
        SELECT * FROM battery_alerts
        WHERE {' AND '.join(query_parts)}
        ORDER BY generated_at DESC, id DESC
    """
    with connect(path) as connection:
        rows = connection.execute(sql, params).fetchall()

    return [
        {
            "id": int(row["id"]),
            "battery_id": row["battery_id"],
            "user_id": row["user_id"],
            "alert_type": row["alert_type"],
            "alert_message": row["alert_message"],
            "category": row["category"],
            "severity": row["severity"],
            "generated_at": row["generated_at"],
            "alert_status": row["alert_status"],
            "relevant_expiry_date": row["relevant_expiry_date"],
            "acknowledged_at": row["acknowledged_at"],
        }
        for row in rows
    ]


def acknowledge_battery_alert(
    alert_id: int | str,
    user_id: int | None = None,
    path: Path | None = None,
) -> bool:
    """Mark an alert as acknowledged in the database."""
    init_db(path)
    now_ts = utc_now()

    if _is_mongodb_active(path):
        db = _get_mongodb()
        from bson import ObjectId  # type: ignore

        query: dict[str, Any] = {}
        if isinstance(alert_id, int) or (isinstance(alert_id, str) and alert_id.isdigit()):
            query["id"] = int(alert_id)
        else:
            try:
                query["_id"] = ObjectId(alert_id)
            except Exception:
                query["id"] = alert_id

        if user_id is not None:
            query["user_id"] = user_id

        res = db["battery_alerts"].update_one(
            query,
            {"$set": {"alert_status": "Acknowledged", "acknowledged_at": now_ts}},
        )
        return res.modified_count > 0

    with connect(path) as connection:
        cursor = connection.execute(
            """
            UPDATE battery_alerts
            SET alert_status = 'Acknowledged', acknowledged_at = ?
            WHERE id = ? AND ((? IS NULL AND user_id IS NULL) OR user_id = ?)
            """,
            (now_ts, alert_id, user_id, user_id),
        )
        connection.commit()
        return cursor.rowcount > 0


def check_and_generate_alerts(
    battery_records: list[dict[str, Any]],
    user_id: int | None = None,
    as_of: date | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Idempotently evaluate and generate warranty & ageing alerts for all batteries."""
    from services.lifecycle import (
        calculate_dynamic_battery_age,
        evaluate_warranty_milestones,
        get_ageing_alert_message,
    )

    current_date = as_of or date.today()
    existing_alerts = list_battery_alerts(user_id=user_id, path=path)

    # Set of existing alert signatures to guarantee idempotency:
    # (battery_id, alert_type, relevant_expiry_date)
    existing_keys = {
        (a["battery_id"], a["alert_type"], a.get("relevant_expiry_date"))
        for a in existing_alerts
    }

    generated_alerts = []

    for b in battery_records:
        b_id = b.get("battery_id") or b.get("battery_identifier")
        if not b_id:
            continue

        p_date = b.get("purchase_date")
        exp_date_str = b.get("warranty_expiry_date")
        days_remaining = b.get("days_remaining")

        # 1. Warranty Alerts Check (90d, 30d, 7d, expired)
        if days_remaining is not None and exp_date_str:
            applicable_w_alerts = evaluate_warranty_milestones(days_remaining)
            for w_alert in applicable_w_alerts:
                key = (b_id, w_alert["alert_type"], exp_date_str)
                if key not in existing_keys:
                    doc = {
                        "battery_id": b_id,
                        "user_id": user_id,
                        "alert_type": w_alert["alert_type"],
                        "alert_message": w_alert["alert_message"],
                        "category": "warranty",
                        "severity": w_alert["severity"],
                        "generated_at": utc_now(),
                        "alert_status": "Unread",
                        "relevant_expiry_date": exp_date_str,
                    }
                    save_battery_alert(doc, path=path)
                    existing_keys.add(key)
                    generated_alerts.append(doc)

        # 2. Battery Ageing Alerts Check (24-36mo, 36-48mo, >48mo)
        if p_date:
            age_info = calculate_dynamic_battery_age(p_date, as_of=current_date)
            tier = age_info["tier"]
            if tier in ("AGEING_INFO", "AGEING_WARNING", "ADVANCED_AGEING"):
                tier_key = (b_id, tier, None)
                if tier_key not in existing_keys:
                    msg = get_ageing_alert_message(tier)
                    if msg:
                        severity = age_info["badge_color"]
                        doc = {
                            "battery_id": b_id,
                            "user_id": user_id,
                            "alert_type": tier,
                            "alert_message": msg,
                            "category": "ageing",
                            "severity": severity,
                            "generated_at": utc_now(),
                            "alert_status": "Unread",
                            "relevant_expiry_date": None,
                        }
                        save_battery_alert(doc, path=path)
                        existing_keys.add(tier_key)
                        generated_alerts.append(doc)

    return generated_alerts


def get_lifecycle_monitoring_summary(
    user_id: int | None = None,
    as_of: date | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Calculate and return full lifecycle fleet metrics, table rows, and alert lists."""
    from services.lifecycle import calculate_dynamic_battery_age, compute_overall_alert_status

    current_date = as_of or date.today()
    batteries = list_user_batteries(user_id=user_id, as_of=current_date, path=path)

    # Automatically check and idempotently generate any due alerts
    check_and_generate_alerts(batteries, user_id=user_id, as_of=current_date, path=path)

    all_alerts = list_battery_alerts(user_id=user_id, path=path)
    unread_alerts = [a for a in all_alerts if a["alert_status"] == "Unread"]
    acknowledged_alerts = [a for a in all_alerts if a["alert_status"] == "Acknowledged"]

    total_batteries = len(batteries)
    active_warranties = sum(1 for b in batteries if b.get("warranty_status") == "ACTIVE")
    expired_warranties = sum(1 for b in batteries if b.get("warranty_status") == "EXPIRED")
    expiring_90_days = sum(
        1 for b in batteries
        if b.get("warranty_status") == "ACTIVE" and b.get("days_remaining") is not None and b.get("days_remaining", 999) <= 90
    )

    ageing_attention_count = 0
    table_rows = []

    for b in batteries:
        b_id = b["battery_id"]
        mfg = b.get("manufacturer") or "Generic"
        p_date = b.get("purchase_date") or "—"
        exp_date = b.get("warranty_expiry_date") or "—"
        w_status = b.get("warranty_status") or "UNKNOWN"
        days_rem = b.get("days_remaining")

        age_info = calculate_dynamic_battery_age(b.get("purchase_date"), as_of=current_date)
        if age_info.get("requires_attention"):
            ageing_attention_count += 1

        status_info = compute_overall_alert_status(
            warranty_status=w_status,
            days_remaining=days_rem,
            age_months=age_info.get("months", 0),
        )

        table_rows.append({
            "Battery ID": b_id,
            "Manufacturer": mfg,
            "Purchase Date": p_date,
            "Battery Age": age_info.get("formatted", "Unknown"),
            "Age Months": age_info.get("months", 0),
            "Warranty Expiry": exp_date,
            "Warranty Status": w_status,
            "Alert Status": status_info["badge"],
            "Alert Color": status_info["color"],
            "Alert Text": status_info["text"],
            "Severity Rank": status_info["severity_rank"],
        })

    return {
        "total_batteries": total_batteries,
        "active_warranties": active_warranties,
        "expiring_90_days": expiring_90_days,
        "expired_warranties": expired_warranties,
        "ageing_attention_count": ageing_attention_count,
        "table_rows": table_rows,
        "unread_alerts": unread_alerts,
        "acknowledged_alerts": acknowledged_alerts,
    }


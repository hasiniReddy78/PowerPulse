"""Database access layer supporting SQLite and MongoDB for lithium-ion battery records.

Reads configuration from .env:
- DB_TYPE: 'mongodb' or 'sqlite' (defaults to 'mongodb' when configured, falling back to 'sqlite')
- MONGODB_URI: MongoDB connection URI (e.g. MongoDB Atlas or local mongod)
- MONGODB_DB_NAME: Database name (default: 'powerpulse_db')
- MONGODB_USERS_COLLECTION: Users collection name (default: 'users')
- MONGODB_BATTERIES_COLLECTION: Battery records collection name (default: 'battery_records')
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from database.schema import INDEXES, STATEMENTS
from ml.config import DATABASE_PATH

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
    mongo_uri = os.environ.get("MONGODB_URI", "").strip()
    db_name = os.environ.get("MONGODB_DB_NAME", "powerpulse_db").strip()
    users_col = os.environ.get("MONGODB_USERS_COLLECTION", "users").strip()
    batt_col = os.environ.get("MONGODB_BATTERIES_COLLECTION", "battery_records").strip()
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
                ("battery_identifier", pymongo.ASCENDING),
                ("category", pymongo.ASCENDING),
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
        # Fallback counter calculation if find_one_and_update fails
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


def _init_sqlite(path: Path | None = None) -> None:
    with connect(path) as connection:
        for statement in STATEMENTS:
            connection.execute(statement)
        for statement in INDEXES:
            connection.execute(statement)
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
# Battery Record Operations
# ---------------------------------------------------------------------------

def add_battery(
    identifier: str,
    category: str = "lithium_ion",
    user_id: int | None = None,
    path: Path | None = None,
) -> int:
    """Register or look up a battery record, returning its numeric record ID."""
    identifier = identifier.strip()
    if not identifier or len(identifier) > 64:
        raise ValueError("Battery identifier is required and must be 64 characters or fewer.")
    category = "lithium_ion"
    init_db(path)

    if _is_mongodb_active(path):
        db = _get_mongodb()
        cfg = get_db_config()
        batt_col = db[cfg["batteries_collection"]]
        query: dict[str, Any] = {
            "battery_identifier": identifier,
            "category": category,
            "user_id": user_id,
        }
        existing = batt_col.find_one(query)
        if existing is not None:
            return int(existing["id"])
        battery_id = _get_next_sequence(db, "battery_records")
        batt_col.insert_one(
            {
                "id": battery_id,
                "user_id": user_id,
                "battery_identifier": identifier,
                "category": category,
                "created_at": utc_now(),
            }
        )
        return battery_id

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


# ---------------------------------------------------------------------------
# Assessment Operations
# ---------------------------------------------------------------------------

def add_assessment(payload: dict, path: Path | None = None) -> int:
    """Store a lithium-ion assessment record, returning its numeric assessment ID."""
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

    if _is_mongodb_active(path):
        db = _get_mongodb()
        cfg = get_db_config()
        assessment_id = _get_next_sequence(db, "assessments")
        doc = {
            "id": assessment_id,
            "battery_record_id": battery_id,
            "user_id": payload.get("user_id"),
            "battery_identifier": payload["battery_identifier"],
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
    """Convert an SQLite Row to a standard assessment dictionary."""
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


def _mongo_doc_to_assessment(doc: dict) -> dict:
    """Normalize a MongoDB document to standard assessment dictionary."""
    return {
        "id": int(doc["id"]),
        "battery_record_id": int(doc["battery_record_id"]),
        "user_id": doc.get("user_id"),
        "battery_identifier": doc.get("battery_identifier", ""),
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
    """Calculate aggregated health metrics and recent alert items."""
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

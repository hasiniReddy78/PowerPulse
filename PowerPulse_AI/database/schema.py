"""SQLite schema. Passwords are stored only as PBKDF2 hashes."""

USERS = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

BATTERY_RECORDS = """
CREATE TABLE IF NOT EXISTS battery_records (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    battery_identifier TEXT NOT NULL,
    category TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
"""

ASSESSMENTS = """
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY,
    battery_record_id INTEGER NOT NULL,
    user_id INTEGER,
    assessed_at TEXT NOT NULL,
    category TEXT NOT NULL,
    model_version TEXT,
    predicted_soh REAL,
    predicted_condition TEXT,
    soh_band TEXT,
    status TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    guidance_json TEXT NOT NULL,
    limitations_json TEXT NOT NULL,
    FOREIGN KEY (battery_record_id) REFERENCES battery_records (id),
    FOREIGN KEY (user_id) REFERENCES users (id)
);
"""

INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_batteries_user ON battery_records (user_id);",
    "CREATE INDEX IF NOT EXISTS idx_assessments_user ON assessments (user_id, assessed_at);",
    "CREATE INDEX IF NOT EXISTS idx_assessments_battery ON assessments (battery_record_id);",
)

STATEMENTS = (USERS, BATTERY_RECORDS, ASSESSMENTS)

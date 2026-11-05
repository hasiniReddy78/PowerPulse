"""Paths, column roles, and a priori modelling choices.

Hyperparameters and selection rules are fixed here so the held-out batteries
are not used to tune them.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "powerpulse_clean.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATABASE_PATH = ROOT / "database" / "powerpulse.db"

RANDOM_SEED = 42
TEST_BATTERY_COUNT = 8
REFERENCE_CAPACITY_AH = 2.0
MODEL_VERSION = "1.0.0"

EXPECTED_COLUMNS = [
    "battery_id",
    "discharge_cycle",
    "ambient_temperature",
    "temperature_mean",
    "temperature_max",
    "voltage_mean",
    "voltage_min",
    "voltage_max",
    "voltage_std",
    "current_mean",
    "current_std",
    "duration_seconds",
    "capacity_ah",
    "soh_percent",
    "condition",
]

# battery_id groups the split. It is never a model feature.
GROUP_COLUMN = "battery_id"

# These columns determine the targets or are the targets themselves.
LEAKAGE_COLUMNS = ["capacity_ah", "soh_percent", "condition"]

TARGET_REGRESSION = "soh_percent"
TARGET_CLASSIFICATION = "condition"
CLASS_LABELS = ["Good", "Fair", "Poor"]

BASE_FEATURES = [
    "ambient_temperature",
    "temperature_mean",
    "temperature_max",
    "voltage_mean",
    "voltage_min",
    "voltage_max",
    "voltage_std",
    "current_mean",
    "current_std",
    "duration_seconds",
]

ENGINEERED_FEATURES = [
    "voltage_range",
    "temperature_rise",
    "temperature_peak_rise",
]

CYCLE_FEATURE = "discharge_cycle"

# Accept engineered features when they are not meaningfully worse than the
# base set. Include discharge cycle only when the gain is practically useful.
ENGINEERED_MAE_TOLERANCE = 0.05
CYCLE_MAE_IMPROVEMENT = 0.5
ALGORITHM_TIE_MAE = 0.25
ALGORITHM_TIE_F1 = 0.005

# Fixed before looking at holdout results.
TREE_PARAMS = {
    "n_estimators": 200,
    "min_samples_leaf": 2,
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
    "verbosity": 0,
    "tree_method": "hist",
}

# Lower rank is preferred when cross-validated scores are effectively tied.
SIMPLICITY_RANK = {
    "linear_regression": 0,
    "logistic_regression": 0,
    "extra_trees": 1,
    "random_forest": 2,
    "gradient_boosting": 3,
    "xgboost": 4,
}

CONDITION_RULES = {
    "poor_max_inclusive": 70.0,
    "fair_max_inclusive": 85.0,
    "good_above": 85.0,
}

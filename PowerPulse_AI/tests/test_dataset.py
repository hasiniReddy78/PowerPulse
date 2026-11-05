import hashlib

from ml.config import DATA_CSV, EXPECTED_COLUMNS
from ml.data_analysis import soh_from_capacity
from ml.dataset import load_clean_dataset

KNOWN_SHA256 = "0b12164078eb5e048fc1ffb2b8aef7bc17c75131e7c4b9dbd9235c5d77554b8b"


def test_dataset_shape_and_hash():
    digest = hashlib.sha256(DATA_CSV.read_bytes()).hexdigest()
    assert digest == KNOWN_SHA256
    frame = load_clean_dataset()
    assert frame.shape == (2485, 15)
    assert list(frame.columns) == EXPECTED_COLUMNS
    assert frame["battery_id"].nunique() == 32
    assert int(frame.isna().sum().sum()) == 0
    assert int(frame.duplicated().sum()) == 0
    assert int(frame.duplicated(subset=["battery_id", "discharge_cycle"]).sum()) == 0


def test_soh_matches_capacity_rule_and_condition_bands():
    frame = load_clean_dataset()
    gap = (soh_from_capacity(frame["capacity_ah"]) - frame["soh_percent"]).abs().max()
    assert gap < 1e-9
    poor = frame.loc[frame["condition"] == "Poor", "soh_percent"]
    fair = frame.loc[frame["condition"] == "Fair", "soh_percent"]
    good = frame.loc[frame["condition"] == "Good", "soh_percent"]
    assert poor.max() <= 70
    assert fair.min() > 70 and fair.max() <= 85
    assert good.min() > 85
    assert set(frame["condition"]) == {"Good", "Fair", "Poor"}


def test_loader_does_not_require_target_columns_to_be_features():
    frame = load_clean_dataset()
    assert "B0005" in set(frame["battery_id"])
    assert int((frame["battery_id"] == "B0005").sum()) == 168

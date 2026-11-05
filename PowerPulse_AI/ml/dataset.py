"""Read-only access to the cleaned laboratory dataset."""

from pathlib import Path

import pandas as pd

from ml.config import DATA_CSV, EXPECTED_COLUMNS


def load_clean_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned CSV. This function never writes to the file."""
    csv_path = Path(path) if path is not None else DATA_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    frame = frame.loc[:, EXPECTED_COLUMNS].copy()
    frame["battery_id"] = frame["battery_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str)
    return frame

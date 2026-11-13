"""Read-only access to the cleaned laboratory dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import CONDITION_RULES, DATA_CSV, EXPECTED_COLUMNS


def derive_condition(soh_series: pd.Series) -> pd.Series:
    """Generate battery condition labels dynamically based on continuous SoH:
    - GOOD: SoH > 85%
    - FAIR: 80% < SoH <= 85%
    - POOR: SoH <= 80%
    """
    poor_max = CONDITION_RULES["poor_max_inclusive"]
    fair_max = CONDITION_RULES["fair_max_inclusive"]
    return pd.Series(
        np.where(
            soh_series <= poor_max,
            "Poor",
            np.where(soh_series <= fair_max, "Fair", "Good"),
        ),
        index=soh_series.index,
    )


def load_clean_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned CSV and generate condition labels dynamically. This function never writes to the file."""
    csv_path = Path(path) if path is not None else DATA_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    frame = frame.loc[:, EXPECTED_COLUMNS].copy()
    frame["battery_id"] = frame["battery_id"].astype(str)
    # Derive condition labels using the centralized condition boundaries
    frame["condition"] = derive_condition(frame["soh_percent"])
    return frame

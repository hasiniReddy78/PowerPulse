"""Battery-wise splits.

Every discharge from one battery stays in a single partition. Rows are not
shuffled independently, because cycles from the same cell are repeated measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import GROUP_COLUMN, RANDOM_SEED, TEST_BATTERY_COUNT


# Eight of 32 batteries (25%). Profiles are the set of chamber temperatures
# recorded for that battery. B0042, B0043 and B0044 are "4+22" because each
# cell has a 22 °C block and a later 4 °C block. Holding one of them out is
# what puts 22 °C discharges in the test set.
TEST_COUNTS_BY_PROFILE = {
    "4": 3,
    "4+22": 1,
    "24": 2,
    "43": 1,
    "44": 1,
}


def temperature_profile(values) -> str:
    temperatures = sorted({float(value) for value in values})
    return "+".join(f"{temperature:.0f}" for temperature in temperatures)


def battery_temperature_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for battery_id, subset in frame.groupby(GROUP_COLUMN):
        rows.append(
            {
                GROUP_COLUMN: battery_id,
                "temperature_profile": temperature_profile(subset["ambient_temperature"]),
                "n_records": int(len(subset)),
            }
        )
    return pd.DataFrame(rows).sort_values(GROUP_COLUMN)


def split_batteries(
    frame: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    """Return reproducible train and test battery identifiers."""
    summary = battery_temperature_table(frame)
    known = set(summary["temperature_profile"])
    expected = set(TEST_COUNTS_BY_PROFILE)
    if known != expected:
        raise ValueError(
            "Temperature profiles differ from the split design. "
            f"Found {sorted(known)}. Revise the split map before training."
        )

    rng = np.random.RandomState(seed)
    train_ids: list[str] = []
    test_ids: list[str] = []
    for profile, n_test in TEST_COUNTS_BY_PROFILE.items():
        members = sorted(summary.loc[summary["temperature_profile"] == profile, GROUP_COLUMN].tolist())
        if n_test >= len(members):
            raise ValueError(f"Profile {profile} cannot hold out {n_test} batteries.")
        order = members.copy()
        rng.shuffle(order)
        test_ids.extend(order[:n_test])
        train_ids.extend(order[n_test:])

    if len(test_ids) != TEST_BATTERY_COUNT:
        raise ValueError(f"Expected {TEST_BATTERY_COUNT} test batteries, got {len(test_ids)}.")
    if set(train_ids) & set(test_ids):
        raise ValueError("A battery was assigned to both train and test.")
    if set(train_ids) | set(test_ids) != set(summary[GROUP_COLUMN]):
        raise ValueError("The split does not cover every battery.")
    return sorted(train_ids), sorted(test_ids)


def partition_frame(
    frame: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame[GROUP_COLUMN].isin(train_ids)].copy()
    test = frame.loc[frame[GROUP_COLUMN].isin(test_ids)].copy()
    overlap = set(train[GROUP_COLUMN]).intersection(test[GROUP_COLUMN])
    if overlap:
        raise ValueError(f"Battery overlap after partition: {sorted(overlap)}")
    return train, test


def ambient_split_summary(
    frame: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
) -> list[dict]:
    """Count records at each chamber temperature. A mixed battery can appear twice."""
    rows = []
    temperatures = sorted(float(value) for value in frame["ambient_temperature"].unique())
    for temperature in temperatures:
        subset = frame.loc[np.isclose(frame["ambient_temperature"], temperature)]
        batteries = set(subset[GROUP_COLUMN])
        rows.append(
            {
                "ambient_temperature": temperature,
                "train_batteries": sorted(batteries & set(train_ids)),
                "test_batteries": sorted(batteries & set(test_ids)),
                "train_records": int(subset[GROUP_COLUMN].isin(train_ids).sum()),
                "test_records": int(subset[GROUP_COLUMN].isin(test_ids).sum()),
            }
        )
    return rows

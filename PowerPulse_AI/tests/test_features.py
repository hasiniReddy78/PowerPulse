import pandas as pd

from ml.config import BASE_FEATURES, LEAKAGE_COLUMNS
from ml.feature_engineering import DischargeFeatureBuilder, assert_no_leakage, engineer_discharge_frame


def test_engineered_features_match_definitions_and_exclude_leakage():
    frame = pd.DataFrame(
        [
            {
                "ambient_temperature": 24.0,
                "temperature_mean": 32.0,
                "temperature_max": 39.0,
                "voltage_mean": 3.5,
                "voltage_min": 2.6,
                "voltage_max": 4.2,
                "voltage_std": 0.23,
                "current_mean": -1.8,
                "current_std": 0.5,
                "duration_seconds": 3600.0,
                "capacity_ah": 1.8,
                "soh_percent": 90.0,
                "condition": "Good",
                "battery_id": "B0005",
            }
        ]
    )
    engineered = engineer_discharge_frame(frame)
    assert engineered.loc[0, "voltage_range"] == frame.loc[0, "voltage_max"] - frame.loc[0, "voltage_min"]
    assert engineered.loc[0, "temperature_rise"] == 8.0
    assert engineered.loc[0, "temperature_peak_rise"] == 15.0
    columns = BASE_FEATURES + ["voltage_range", "temperature_rise", "temperature_peak_rise"]
    assert_no_leakage(columns)
    builder = DischargeFeatureBuilder(columns=columns).fit(frame)
    matrix = builder.transform(frame)
    assert matrix.shape == (1, len(columns))
    assert list(builder.get_feature_names_out()) == columns
    for banned in LEAKAGE_COLUMNS + ["battery_id"]:
        assert banned not in builder.feature_columns_

# PowerPulse dataset analysis

Source file: `data/powerpulse_clean.csv`.
This report is generated from that file. No rows were removed.

- SHA-256: `0b12164078eb5e048fc1ffb2b8aef7bc17c75131e7c4b9dbd9235c5d77554b8b`
- Records: 2485
- Columns: 15
- Unique batteries: 32
- Duplicate rows: 0
- Duplicate battery and cycle pairs: 0

## Columns and data types

- `battery_id`: str
- `discharge_cycle`: int64
- `ambient_temperature`: float64
- `temperature_mean`: float64
- `temperature_max`: float64
- `voltage_mean`: float64
- `voltage_min`: float64
- `voltage_max`: float64
- `voltage_std`: float64
- `current_mean`: float64
- `current_std`: float64
- `duration_seconds`: float64
- `capacity_ah`: float64
- `soh_percent`: float64
- `condition`: str

## Missing values

Every column has zero missing values.

## Records per battery

| Battery | Records |
| --- | ---: |
| B0005 | 168 |
| B0006 | 168 |
| B0007 | 168 |
| B0018 | 132 |
| B0025 | 28 |
| B0026 | 28 |
| B0027 | 28 |
| B0028 | 28 |
| B0029 | 40 |
| B0030 | 40 |
| B0031 | 40 |
| B0032 | 40 |
| B0033 | 181 |
| B0034 | 195 |
| B0036 | 195 |
| B0038 | 35 |
| B0039 | 35 |
| B0040 | 33 |
| B0041 | 25 |
| B0042 | 65 |
| B0043 | 65 |
| B0044 | 65 |
| B0045 | 70 |
| B0046 | 69 |
| B0047 | 69 |
| B0048 | 69 |
| B0049 | 23 |
| B0051 | 22 |
| B0053 | 55 |
| B0054 | 102 |
| B0055 | 102 |
| B0056 | 102 |

## Targets

SoH in the file matches `min(100, capacity_ah / 2.0 × 100)`.
Maximum absolute difference: 4.263e-14.
Records with capacity above 2.0 Ah: 7 (maximum capacity 2.035338 Ah). Those seven B0006 rows have SoH stored as 100%.
The 2.0 Ah divisor is the preprocessing assumption already present in the file. It is the usual nominal capacity quoted for the NASA 18650 cells, but capacities above 2.0 Ah show that it should not be treated as a measured nominal capacity for every record.

Condition labels match the stated bands (Poor ≤ 70, Fair ≤ 85, Good > 85). Mismatches: 0.

| Condition | Records | Share |
| --- | ---: | ---: |
| Good | 606 | 24.4% |
| Fair | 858 | 34.5% |
| Poor | 1021 | 41.1% |

The classes are imbalanced. Poor is the largest class. Accuracy alone would overstate a classifier that mostly predicts Poor.

## SoH summary

- Minimum: 30.347%
- 25th percentile: 62.971%
- Median: 73.376%
- Mean: 72.162%
- 75th percentile: 84.734%
- Maximum: 100.000%
- Standard deviation: 14.904 percentage points

## Unusual values kept in the file

These were not deleted.

- 7 capacities exceed the 2.0 Ah reference, and the corresponding SoH values are capped at 100%.
- Capacity is not strictly decreasing. Increases larger than 0.01 Ah versus the previous stored cycle were counted for 31 batteries.
- Some batteries do not start at discharge cycle 1: B0033 (starts at 8), B0034 (starts at 2), B0036 (starts at 2), B0038 (starts at 13), B0039 (starts at 13), B0040 (starts at 13), B0041 (starts at 43), B0051 (starts at 2).
- Current is negative on every row. That is the discharge sign convention in this file.
- Internal checks found no inverted voltage bounds, no mean voltage outside the min/max, no mean temperature below ambient, and no non-positive duration or standard deviation.

## Why rows are not independent

Each battery contributes many discharge cycles. A random row split would put neighbouring cycles from the same cell in both train and test.
Most cells stay at one chamber temperature. B0042, B0043 and B0044 do not: each has an early 22 °C block and a later 4 °C block. The whole battery still stays in one split. Batteries that contribute at least one row at each temperature:

- 4 °C: 14 batteries
- 22 °C: 3 batteries
- 24 °C: 11 batteries
- 43 °C: 4 batteries
- 44 °C: 3 batteries

Correlation of each numeric column with SoH, computed on all rows and therefore inflated by repeated measures:

| Column | Correlation with SoH |
| --- | ---: |
| `voltage_std` | -0.713 |
| `current_mean` | -0.361 |
| `discharge_cycle` | -0.164 |
| `current_std` | -0.013 |
| `duration_seconds` | 0.132 |
| `voltage_min` | 0.142 |
| `voltage_max` | 0.232 |
| `temperature_max` | 0.562 |
| `temperature_mean` | 0.675 |
| `voltage_mean` | 0.676 |
| `ambient_temperature` | 0.740 |
| `capacity_ah` | 1.000 |

`capacity_ah` correlates with SoH at essentially 1 because SoH was calculated from capacity. It is excluded from the model inputs.

Ambient temperature correlates strongly with SoH (about 0.74) because the cold experiments contain most of the Poor labels. That association is experimental structure, not evidence that a warmer room repairs a battery.
Current is also tied to the experiment: the 43 °C group is discharged near 4 A, while several 4 °C cells are discharged below 1 A. A correlation between current and SoH is therefore confounded.

Median within-battery correlation between discharge duration and capacity: 0.917. Duration tracks capacity inside a cell when the current profile is stable, but it is a weak SoH proxy across the whole file because the discharge rate changes between experiments. Duration is still a legitimate measurement. Capacity is not used as a feature.

## Figures

- `reports/figures/soh_distribution.html`
- `reports/figures/capacity_distribution.html`
- `reports/figures/condition_counts.html`
- `reports/figures/soh_degradation_by_battery.html`
- `reports/figures/capacity_degradation_by_battery.html`

Laboratory discharge curves are not a validation of real-world electric-vehicle packs.

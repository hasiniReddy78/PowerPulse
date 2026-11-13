# Lithium-ion model evaluation

Model version: 1.0.0
Trained at (UTC): 2026-11-13T03:32:43+00:00

Selection used GroupKFold on the training batteries only. The test batteries were scored after the algorithms were fixed.

## Split

- Training batteries (24): B0006, B0007, B0018, B0025, B0026, B0027, B0028, B0029, B0031, B0032, B0034, B0036, B0039, B0040, B0042, B0043, B0045, B0046, B0047, B0048, B0051, B0053, B0054, B0056
- Test batteries (8): B0005, B0030, B0033, B0038, B0041, B0044, B0049, B0055

| Ambient °C | Train batteries | Test batteries | Train rows | Test rows |
| ---: | --- | --- | ---: | ---: |
| 4 | B0042, B0043, B0045, B0046, B0047, B0048, B0051, B0053, B0054, B0056 | B0041, B0044, B0049, B0055 | 608 | 175 |
| 22 | B0042, B0043 | B0044 | 80 | 40 |
| 24 | B0006, B0007, B0018, B0025, B0026, B0027, B0028, B0034, B0036 | B0005, B0033 | 970 | 349 |
| 43 | B0029, B0031, B0032 | B0030 | 120 | 40 |
| 44 | B0039, B0040 | B0038 | 68 | 35 |

## Features

Discharge cycle was not included. Grouped cross-validation did not show an MAE improvement of at least 0.5 SoH points, and cycle numbers are not aligned across the NASA experiments (several batteries do not start at cycle 1). Engineered features were kept because they did not make grouped MAE meaningfully worse.

Model features: `ambient_temperature`, `temperature_mean`, `temperature_max`, `voltage_mean`, `voltage_min`, `voltage_max`, `voltage_std`, `current_mean`, `current_std`, `duration_seconds`, `voltage_range`, `temperature_rise`, `temperature_peak_rise`

Excluded because they leak the target or identify the cell: `capacity_ah`, `soh_percent`, `condition`, `battery_id`

## Regression cross-validation (training batteries)

| Algorithm | MAE mean | MAE std | RMSE mean | R² mean |
| --- | ---: | ---: | ---: | ---: |
| linear_regression | 5.508 | 1.219 | 7.156 | 0.755 |
| random_forest | 4.066 | 0.812 | 6.059 | 0.818 |
| extra_trees | 4.316 | 1.514 | 5.867 | 0.814 |
| gradient_boosting | 4.056 | 2.296 | 5.644 | 0.833 |
| xgboost | 4.121 | 1.444 | 5.956 | 0.816 |

Selected regressor: `random_forest`.

## Regression test batteries (scored once)

- MAE: 2.729 percentage points
- RMSE: 4.884 percentage points
- R²: 0.869
- Test records: 639

Leave-one-battery-out MAE on the training batteries (diagnostic, not used for selection): mean 4.382, std 4.573, min 0.875, max 20.611.

## Classification cross-validation (training batteries)

Poor is the largest class, so macro F1 is the selection metric.

| Algorithm | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| logistic_regression | 0.751 | 0.702 | 0.697 | 0.650 |
| random_forest | 0.841 | 0.782 | 0.716 | 0.701 |
| extra_trees | 0.840 | 0.740 | 0.718 | 0.695 |
| gradient_boosting | 0.818 | 0.716 | 0.723 | 0.705 |
| xgboost | 0.823 | 0.744 | 0.757 | 0.731 |

Selected classifier: `xgboost`.

## Classification test batteries (scored once)

- Accuracy: 0.862
- Macro precision: 0.795
- Macro recall: 0.724
- Macro F1: 0.753
- Weighted F1: 0.854

Confusion matrix rows and columns are Good, Fair, Poor:

```
[[89, 6, 11], [1, 30, 48], [0, 22, 432]]
```

```
{
  "Good": {
    "precision": 0.9888888888888889,
    "recall": 0.839622641509434,
    "f1-score": 0.9081632653061225,
    "support": 106.0
  },
  "Fair": {
    "precision": 0.5172413793103449,
    "recall": 0.379746835443038,
    "f1-score": 0.43795620437956206,
    "support": 79.0
  },
  "Poor": {
    "precision": 0.879837067209776,
    "recall": 0.9515418502202643,
    "f1-score": 0.9142857142857143,
    "support": 454.0
  },
  "accuracy": 0.8622848200312989,
  "macro avg": {
    "precision": 0.7953224451363367,
    "recall": 0.7236371090575787,
    "f1-score": 0.7534683946571329,
    "support": 639.0
  },
  "weighted avg": {
    "precision": 0.8530990918638148,
    "recall": 0.8622848200312989,
    "f1-score": 0.8543811589266802,
    "support": 639.0
  }
}
```

## Limitations

- The model estimates SoH for one lithium-ion cell discharge summarised the way the NASA laboratory file was summarised. It does not assess a full electric-vehicle pack.
- Held-out scores use a small number of batteries. A low or negative R² means the model does not generalise reliably to unseen cells.
- Ambient temperature and discharge current are confounded with the original experiments. They are not causal maintenance advice.
- The 2.0 Ah reference is the assumption already used to label this file. Seven early B0006 capacities exceed 2.0 Ah and their SoH labels are capped at 100%.
- Class probabilities are not calibrated and are not shown as confidence.
- Laboratory NASA results do not validate real-world vehicle performance.
- Assessment is calibrated specifically for lithium-ion battery cells under laboratory discharge test conditions.

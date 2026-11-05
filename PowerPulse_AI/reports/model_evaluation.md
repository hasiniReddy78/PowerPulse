# Lithium-ion model evaluation

Model version: 1.0.0
Trained at (UTC): 2026-11-04T16:49:36+00:00

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
| logistic_regression | 0.771 | 0.784 | 0.738 | 0.721 |
| random_forest | 0.826 | 0.834 | 0.824 | 0.801 |
| extra_trees | 0.835 | 0.835 | 0.827 | 0.813 |
| gradient_boosting | 0.777 | 0.797 | 0.792 | 0.764 |
| xgboost | 0.834 | 0.849 | 0.846 | 0.825 |

Selected classifier: `xgboost`.

## Classification test batteries (scored once)

- Accuracy: 0.917
- Macro precision: 0.906
- Macro recall: 0.928
- Macro F1: 0.916
- Weighted F1: 0.917

Confusion matrix rows and columns are Good, Fair, Poor:

```
[[103, 3, 0], [17, 229, 14], [0, 19, 254]]
```

```
{
  "Good": {
    "precision": 0.8583333333333333,
    "recall": 0.9716981132075472,
    "f1-score": 0.911504424778761,
    "support": 106.0
  },
  "Fair": {
    "precision": 0.9123505976095617,
    "recall": 0.8807692307692307,
    "f1-score": 0.8962818003913894,
    "support": 260.0
  },
  "Poor": {
    "precision": 0.9477611940298507,
    "recall": 0.9304029304029304,
    "f1-score": 0.9390018484288355,
    "support": 273.0
  },
  "accuracy": 0.917057902973396,
  "macro avg": {
    "precision": 0.9061483749909153,
    "recall": 0.9276234247932361,
    "f1-score": 0.9155960245329954,
    "support": 639.0
  },
  "weighted avg": {
    "precision": 0.9185184580312498,
    "recall": 0.917057902973396,
    "f1-score": 0.9170582812979374,
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

# Lithium-Ion Model Retraining & Continuous Improvement Workflow

The Streamlit application loads pre-trained serialized model artifacts from `models/`. It does not train models on startup. Replacing those files with a newly validated artifact version updates health predictions across the application without modifying application UI code.

---

## 1. Core ML Rules & Principles

- **Preserve Source Data**: Never overwrite or delete `data/powerpulse_clean.csv`.
- **Target Leakage Prevention**: Never use `capacity_ah`, `soh_percent`, `condition`, or `battery_id` as model features.
- **Battery-Wise Splitting**: Always partition by whole battery cell (`battery_id`), never with random row-level splitting. Repeated discharge cycles from the same cell must not cross partitions.
- **Model Selection Protocol**: Select algorithms with grouped cross-validation on training batteries only. Score holdout test batteries only once post-selection.
- **Targeted Scope**: PowerPulse focuses exclusively on lithium-ion batteries. Do not introduce unvalidated proxy models.
- **Version Tracking**: Assign a clean `model_version` tag (e.g. `1.1.0`) for every trained candidate.

---

## 2. Train a Candidate Model

When a new lithium-ion dataset is introduced (such as laboratory cycling or EV BMS diagnostic logs, e.g. `data/incoming/ev_field_v2.csv`):

```bash
python -m ml.train_models --data data/incoming/ev_field_v2.csv --version 1.1.0 --output models/candidates/1.1.0
```

1. Inspect `models/candidates/1.1.0/model_evaluation.md` and compare it with the active `reports/model_evaluation.md`.
2. Check macro F1-score across all classes (`Good`, `Fair`, `Poor`), holdout MAE/RMSE, and performance across different operating temperatures.

---

## 3. Review & Promotion

Only after confirming that the new candidate outperforms or expands upon the existing baseline:

```bash
python -m ml.train_models --promote models/candidates/1.1.0
```

This updates the five core artifacts in `models/` (`regression_model.pkl`, `classification_model.pkl`, `preprocessing.pkl`, `model_metadata.json`, and `evaluation_metrics.json`) and refreshes `reports/model_evaluation.md`.

The Streamlit application loader dynamically detects the updated model artifacts and serves the new model on the next inference call.

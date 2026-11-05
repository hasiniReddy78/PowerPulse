# PowerPulse: Lithium-Ion Battery Health Assessment & Predictive Maintenance Platform

**PowerPulse** is a machine-learning-based battery intelligence platform focused exclusively on **Lithium-Ion Battery Health Assessment, Degradation Tracking, and Predictive Maintenance**.

The system analyses measurable battery operating parameters to:
- Estimate State of Health (SoH %).
- Classify battery condition (`Good`, `Fair`, `Poor`).
- Explain predictions with **SHAP (Shapley Additive exPlanations)** local and global feature attribution.
- Detect abnormal battery behaviour and voltage/temperature anomalies.
- Analyse longitudinal capacity fade trajectories.
- Provide early warnings before critical failure.
- Estimate Remaining Useful Life (RUL) when multi-cycle historical data is available.
- Generate downloadable PDF and CSV battery health assessment reports.

> **Dataset & Operating Domain:**
> The initial models are trained and calibrated on the cleaned NASA lithium-ion battery degradation dataset (`data/powerpulse_clean.csv`), consisting of 2.0 Ah 18650 cylindrical cells cycled across multiple ambient temperatures.
> These laboratory discharge experiments provide single-cell baseline characteristics and do not represent a vehicle-fleet trial or on-road EV pack validation. The platform architecture is designed to integrate electric vehicle BMS telemetry and field cycling data as compatible datasets become available.

---

## Technology Stack

- **Application & UI**: Python, Streamlit (modular multi-page structure)
- **Data Processing & ML**: Pandas, NumPy, Scikit-learn, XGBoost, Joblib
- **Model Interpretability**: SHAP (TreeExplainer, Shapley Additive attributions)
- **Visualization**: Plotly Interactive Visualizations, Streamlit Charts
- **Database & Security**: SQLite with PBKDF2-HMAC-SHA256 password hashing
- **Testing**: Pytest automated test suite

---

## Project Structure

```
minorProject/
├── .gitignore                  # Repository-level exclusions
├── requirements.txt            # Project dependencies
├── README.md                   # Repository documentation
├── powerpulse_clean.csv        # Cleaned NASA lithium-ion battery dataset
│
└── PowerPulse_AI/              # Core application package
    ├── app.py                  # Application home & project overview
    ├── pages/
    │   ├── dashboard.py        # Overview statistics, condition distribution, degradation trends, alerts, global SHAP
    │   ├── battery_assessment.py # Lithium-ion health estimation, anomaly detection, RUL & local SHAP waterfall
    │   ├── history.py          # Longitudinal degradation curves & assessment history
    │   └── reports.py          # Downloadable CSV & PDF assessment reports
    ├── ml/
    │   ├── config.py           # Constants, column roles, and ML hyperparameters
    │   ├── dataset.py          # Dataset loading & integrity verification
    │   ├── data_analysis.py    # Phase 1 exploratory data analysis & figures
    │   ├── splitting.py        # Battery-wise stratified splitting (no row-level leakage)
    │   ├── feature_engineering.py # Safe feature transformations without target leakage
    │   ├── preprocessing.py    # Training ranges & column schemas
    │   ├── train_regression.py # SoH regressors (RF, ExtraTrees, GB, XGB, Linear)
    │   ├── train_classification.py # Condition classifiers (RF, ExtraTrees, GB, XGB, LogReg)
    │   ├── evaluate.py         # MAE, RMSE, R2, F1, Confusion Matrix, Classification Report
    │   ├── predict.py          # Validation, inference, and RUL estimation logic
    │   ├── explainability.py   # Local & global SHAP attribution, waterfalls & importance
    │   ├── model_loader.py     # Joblib deserializer and model artifact manager
    │   └── errors.py           # Domain exception classes
    ├── models/
    │   ├── regression_model.pkl    # Serialized Random Forest SoH regressor
    │   ├── classification_model.pkl# Serialized XGBoost Condition classifier
    │   ├── preprocessing.pkl       # Feature transformation pipeline
    │   ├── model_metadata.json     # Model version, feature names, ranges, limitations
    │   └── evaluation_metrics.json # Full CV, test, and LOBO metrics
    ├── database/
    │   ├── schema.py           # SQLite schema definition
    │   └── db.py               # Data access layer for users, batteries, assessments
    ├── reports/
    │   ├── dataset_analysis.md # Phase 1 dataset inspection findings
    │   ├── model_evaluation.md # Phase 4 model evaluation & benchmark results
    │   ├── battery_summary.csv # Per-battery summary statistics
    │   └── figures/            # Plotly HTML figures
    ├── docs/
    │   └── retraining.md       # Model retraining, evaluation, and promotion workflow
    ├── tests/                  # Automated pytest suite
    └── requirements.txt        # Application dependencies
```

---

## Machine Learning & Explainability Pipeline

### 1. Data Leakage Prevention
- Target variables (`soh_percent`, `condition`) and target-defining measurements (`capacity_ah`) are strictly excluded from predictive features.
- Cell identifier (`battery_id`) is strictly used as the split grouping key, never as an input feature.
- Feature ablation demonstrated that `discharge_cycle` degraded grouped MAE from 4.06 to 4.92 pp (due to unaligned cycle numbering across NASA experiment batches) and was excluded from predictive inputs.
- Engineered features (`voltage_range`, `temperature_rise`, `temperature_peak_rise`) were verified to prevent leakage and retained.

### 2. Battery-Wise Train/Test Splitting
- Split: 24 training batteries (1,846 records) and 8 holdout test batteries (639 records).
- All discharge cycles of any given battery remain 100% within one partition.
- Stratified by experimental ambient temperature profile (4 °C, 22 °C, 24 °C, 43 °C, 44 °C).

### 3. Model Benchmark & Selection
Models were selected strictly using 4-split `GroupKFold` across training batteries only:

#### Task A: State of Health (SoH %) Regression
| Algorithm | CV MAE (pp) | CV RMSE | CV R² | Selection |
| :--- | :---: | :---: | :---: | :---: |
| Linear Regression | 5.508 | 7.156 | 0.755 | |
| Extra Trees Regressor | 4.316 | 5.867 | 0.814 | |
| XGBoost Regressor | 4.121 | 5.956 | 0.816 | |
| Gradient Boosting Regressor | 4.056 | 5.644 | 0.833 | |
| **Random Forest Regressor** | **4.066** | **6.059** | **0.818** | **Selected** |

*Holdout Test Performance (8 unseen batteries, 639 records):*
- **MAE**: 2.729 percentage points
- **RMSE**: 4.884 percentage points
- **R²**: 0.869

#### Task B: Condition Classification (Good / Fair / Poor)
| Algorithm | CV Accuracy | CV Macro Precision | CV Macro Recall | CV Macro F1 | Selection |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.771 | 0.784 | 0.738 | 0.721 | |
| Gradient Boosting Classifier | 0.777 | 0.797 | 0.792 | 0.764 | |
| Random Forest Classifier | 0.826 | 0.834 | 0.824 | 0.801 | |
| Extra Trees Classifier | 0.835 | 0.835 | 0.827 | 0.813 | |
| **XGBoost Classifier** | **0.834** | **0.849** | **0.846** | **0.825** | **Selected** |

*Holdout Test Performance (8 unseen batteries, 639 records):*
- **Accuracy**: 91.71%
- **Macro F1-Score**: 0.9156
- **Weighted F1-Score**: 0.9171
- **Good class F1**: 0.9115 (Recall 97.17%)
- **Fair class F1**: 0.8963 (Recall 88.08%)
- **Poor class F1**: 0.9390 (Recall 93.04%)

### 4. SHAP Explainability Features
- **Local Attribution (`explain_prediction`)**: Decomposes every single SoH assessment into exact Shapley contributions:
  $$\text{Predicted SoH} = \text{Base Value} + \sum_{i=1}^M \phi_i$$
  Visualized via interactive Plotly horizontal waterfall attribution charts in the assessment workflow. Identifies specific positive drivers (health boosters) and negative drivers (degradation factors).
- **Global Interpretability (`get_global_feature_importance`)**: Computes mean absolute SHAP values across representative discharge cycles on the Dashboard, revealing macro trends in battery degradation physics.
- **Report Integration**: SHAP positive and negative attribution factors are embedded directly into generated PDF and CSV assessment reports.

---

## Installation & Execution

### 1. Environment Setup
From the repository root (`minorProject`):
```bash
pip install -r requirements.txt
```

### 2. Database Configuration (SQLite or MongoDB)

PowerPulse supports dual database storage engines configured via environment variables in `.env`:
- **SQLite (Default / Fallback)**: Zero-configuration local database (`database/powerpulse.db`).
- **MongoDB**: Cloud (MongoDB Atlas) or local MongoDB instance for scalable multi-user assessment storage.

#### Switching to MongoDB:
1. Copy `.env.example` to `.env` (or edit the existing `.env`):
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and configure your settings:
   ```env
   # Database Engine Selection ('mongodb' or 'sqlite')
   DB_TYPE=mongodb

   # MongoDB Connection URI (paste your credentials / cluster URI)
   # Example: mongodb+srv://<username>:<password>@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority
   MONGODB_URI=your_mongodb_connection_uri_here

   # Database and collection names
   MONGODB_DB_NAME=powerpulse_db
   MONGODB_USERS_COLLECTION=users
   MONGODB_BATTERIES_COLLECTION=battery_records
   MONGODB_ASSESSMENTS_COLLECTION=assessments
   MONGODB_TIMEOUT_MS=5000
   ```
3. **Safety Fallback**: If `MONGODB_URI` is left blank, `pymongo` is missing, or the cluster is temporarily unreachable, PowerPulse automatically falls back to local SQLite without interrupting the application.

### 3. Run the Streamlit Application
```bash
cd PowerPulse_AI
streamlit run app.py
```
Open `http://localhost:8501` in your browser. Pre-trained models and SHAP explainers load instantly without retraining.

### 4. Run the Automated Tests
```bash
cd PowerPulse_AI
pytest
```
All unit, integration, and explainability tests will execute.

### 5. Optional: Retraining
```bash
cd PowerPulse_AI
python -m ml.train_models
```
Performs battery-wise splitting, cross-validation, holdout evaluation, and serializes fresh model artifacts into `models/`.



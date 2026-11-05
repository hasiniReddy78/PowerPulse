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
- **Database & Security**: SQLite and MongoDB (configurable via `.env`) with PBKDF2-HMAC-SHA256 password hashing
- **Testing**: Pytest automated test suite

---

## Project Structure

```
minorProject/
├── .env                        # Local database & environment configuration (ignored by git)
├── .env.example                # Template for environment configuration
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
    │   └── db.py               # Data access layer for users, batteries, assessments (SQLite + MongoDB)
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
1. Open `.env` (or copy `.env.example` to `.env`):
   ```bash
   cp .env.example .env
   ```
2. Configure your MongoDB connection string in `.env`:
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
All 23 unit, integration, explainability, and database tests will execute.

### 5. Optional: Retraining
```bash
cd PowerPulse_AI
python -m ml.train_models
```
Performs battery-wise splitting, cross-validation, holdout evaluation, and serializes fresh model artifacts into `models/`.

# PowerPulse: Lithium-Ion Battery Health Assessment & Predictive Maintenance Platform

**PowerPulse** is an enterprise-grade machine-learning battery intelligence platform built for **Lithium-Ion Battery Health Assessment, Warranty Validation Policy Enforcement, Degradation Tracking, and Predictive Maintenance**.

The system enables service centres, engineers, and fleet operators to:
- Enforce **Battery Warranty Validation Policies** before executing ML assessments.
- Estimate State of Health (SoH %).
- Classify battery condition (`Good`, `Fair`, `Poor`).
- Explain predictions with **SHAP (Shapley Additive exPlanations)** local and global feature attribution.
- Detect abnormal battery behaviour and voltage/temperature anomalies.
- Analyse longitudinal capacity fade trajectories.
- Provide early warnings before critical failure.
- Estimate Remaining Useful Life (RUL) when multi-cycle historical data is available.
- Manage multi-tenant Service Centre accounts with protected access and session isolation.
- Generate downloadable PDF and CSV battery health assessment reports.

> **Dataset & Operating Domain:**
> The initial models are trained and calibrated on the cleaned NASA lithium-ion battery degradation dataset (`data/powerpulse_clean.csv`), consisting of 2.0 Ah 18650 cylindrical cells cycled across multiple ambient temperatures.
> These laboratory discharge experiments provide single-cell baseline characteristics and do not represent a vehicle-fleet trial or on-road EV pack validation. The platform architecture is designed to integrate electric vehicle BMS telemetry and field cycling data as compatible datasets become available.

---

## Technology Stack

- **Application & UI**: Python, Streamlit (modular multi-page structure with maroon, pink, and white theme)
- **Data Processing & ML**: Pandas, NumPy, Scikit-learn, XGBoost, Joblib
- **Model Interpretability**: SHAP (TreeExplainer, Shapley Additive attributions)
- **Visualization**: Plotly Interactive Visualizations, Streamlit Charts
- **Database & Security**: SQLite and MongoDB (configurable via `.env`) with PBKDF2-HMAC-SHA256 password hashing
- **Testing**: Pytest automated test suite (61 unit, integration, lifecycle, warranty, EOL, and UI tests)

---

## Battery Warranty Management & Policy Enforcement

### 1. Warranty Policy Gate
Before performing any machine learning-based health assessment, the system validates the battery's warranty status:
- **`ACTIVE` Warranty**:
  - Current date is on or before calculated expiry date ($\text{Expiry Date} = \text{Purchase Date} + \text{Warranty Duration}$).
  - Displays green verification badge and allows user to proceed to ML health assessment.
- **`EXPIRED` Warranty**:
  - Current date is after the calculated expiry date.
  - Displays a clear policy block message:
    `"WARRANTY EXPIRED – Battery health assessment is unavailable under the current warranty policy."`
  - Blocks Random Forest prediction, XGBoost classification, SHAP attribution, and RUL prognostics.
  - Prevents creating or saving unauthenticated assessment records.
  - Backend enforcement: Enforced directly in `assess()` to prevent programmatic bypass.

### 2. Multi-Tenant Service Centre Account Isolation
- Batteries and assessments are bound to the authenticated service centre user (`user_id`).
- Unauthorized accounts cannot access, view, or assess another service centre's battery records.

---

## Battery Lifecycle Monitoring & Smart Alerts

PowerPulse includes a dedicated **Battery Lifecycle Monitoring** engine designed to track battery fleet operational lifecycles, dynamically compute battery ageing, generate proactive warranty/ageing alerts, and maintain full separation from NASA ML degradation models.

### 1. Dynamic Battery Age Calculation
- **Non-Static Storage**: Battery age is never stored as a fixed value in MongoDB. Storing static age causes database staleness as calendar time elapses.
- **Dynamic Wall-Clock Arithmetic**: Calculated on-the-fly comparing the battery `purchase_date` against current system date:
  $$\text{Age (months)} = (\text{current\_year} - \text{purchase\_year}) \times 12 + (\text{current\_month} - \text{purchase\_month})$$
  Adjusted accurately when current day of the month is prior to the purchase day, including full support for leap years and month ends.
- **Human-Readable Formats**: Formatted dynamically as `X years Y months` (e.g. `2 years 3 months`) alongside total months.

### 2. Multi-Tier Warranty Alerts
The system continuously evaluates battery warranty expiry dates against dynamic milestone thresholds:
- **90 Days Before Expiry**: `"Your battery warranty expires in 3 months."` (Status: `Expires in 3 Months`, Yellow badge)
- **30 Days Before Expiry**: `"Your battery warranty expires in 30 days."` (Status: `Expiring Soon`, Orange badge)
- **7 Days Before Expiry**: `"Your battery warranty expires in 7 days."` (Status: `Critical Expiry`, Red badge)
- **Post-Expiry**: `"Your battery warranty has expired."` (Status: `Warranty Expired`, Red badge)

### 3. Configurable Battery Ageing Alerts
Proactive operational alerts based on physical service time (separate from ML health classification):
- **Below 24 Months**: `Normal` (No alert required).
- **24–36 Months**: `"This battery has been in service for 2 years. Consider periodic health monitoring."` (`Ageing Information`)
- **36–48 Months**: `"Your battery is approaching 3 years of usage. A health assessment is recommended."` (`Ageing Warning`)
- **Above 48 Months**: `"This battery has been in service for more than 4 years. Consider a detailed battery inspection."` (`Advanced Ageing Alert`)

> **Note on ML Model Non-Interference:**
> Battery age and purchase dates are operational business metrics. They are **never** injected into the Random Forest SoH regressor or XGBoost condition classifier feature vectors. Ageing alerts serve as maintenance reminders and do not label a battery as degraded without ML sensor verification.

### 4. Alert Tracking & MongoDB Persistence
- Alert records are persisted in MongoDB `battery_alerts` (or SQLite fallback) with fields:
  `battery_id`, `user_id`, `alert_type`, `severity`, `title`, `message`, `alert_status` (`Unread` / `Acknowledged`), and timestamps.
- **Idempotent De-duplication**: Unique compound indexes ensure the dashboard re-evaluation never inserts duplicate alerts on page reloads.
- **Service Centre Acknowledgement**: Service center personnel can acknowledge active alerts directly within the Dashboard UI, immediately updating alert state in MongoDB.

### 5. Streamlit Dashboard Integration
A dedicated **Battery Lifecycle Monitoring** command view featuring:
- **5 KPI Metrics**: Total Registered Batteries, Active Warranties, Expiring Within 90 Days, Expired Warranties, and Ageing Alerts Needed.
- **Fleet Table**: Standardized columns (`Battery ID`, `Manufacturer`, `Purchase Date`, `Battery Age`, `Warranty Expiry`, `Warranty Status`, `Alert Status`) with color badges (Green, Yellow, Orange, Red).
- **Interactive Lifecycle Alerts Table**: Shows active notifications with direct **Acknowledge** button triggers.

---

## Authentication & Navigation UI

1. **Clean Navigation**: Removed persistent Login/Signup radio inputs from the left sidebar.
2. **Dedicated Auth Portal**: Unauthenticated sessions are greeted by a branded Maroon (`#7B112D`), Pink (`#FCE4EC`), and White portal with dedicated Log In and Sign Up tabs.
3. **Session Gating**: Protected pages (`battery_assessment`, `dashboard`, `history`, `reports`) are strictly inaccessible to unauthenticated sessions.
4. **Header Profile & Logout**: Authenticated users see their service centre username in the sidebar with a clean Logout button that immediately terminates the session.

---

## End-of-Life (EOL) 80% Threshold & RUL Prognostics

### 1. Centralized 80% SoH EOL Reference
PowerPulse centralizes the End-of-Life State of Health criterion at:
$$\text{EOL\_THRESHOLD} = 80.0\%$$
- **EV Industry Standard Benchmark**: In automotive and industrial battery engineering, 80% State of Health is the globally accepted standard definition for an electric vehicle (EV) battery's "end of first life". At this threshold, power delivery, internal resistance, and vehicle driving range begin to decline rapidly, prompting cell retirement or second-life stationary energy storage reuse.
- **Modelling Criterion vs Universal Rule**: This threshold is an engineering modelling reference for predictive maintenance and cycle life projections. It is **not** a universal mandatory scrappage rule; actual retirement or decommissioning criteria depend on vehicle OEM specifications, safety tolerances, and application operating envelopes.

### 2. Battery Condition Classification Ranges
Condition boundaries are consistently defined across models, datasets, UI, and reports:
- **`GOOD`**: $\text{SoH} > 85.0\%$ (Normal operational health; continue routine service intervals).
- **`FAIR`**: $80.0\% < \text{SoH} \le 85.0\%$ (Moderate degradation; 85% boundary belongs to Fair; shorten inspection intervals).
- **`POOR`**: $\text{SoH} \le 80.0\%$ (Critical degradation; 80% boundary belongs to Poor; prompt inspection or cell replacement).

### 3. Remaining Useful Life (RUL) Prognostics
When $\ge 2$ distinct discharge cycles are recorded for a battery identifier, the system evaluates the linear degradation trajectory:
$$\text{SoH} = m \cdot \text{cycle} + b$$
Where:
- $m$: degradation rate (loss in SoH percentage points per discharge cycle)
- $b$: initial capacity intercept
- $\text{EOL Cycle} = \frac{80.0 - b}{m}$
- $\text{RUL} = \text{EOL Cycle} - \text{Current Cycle}$

**Operational Edge Case Handling**:
- **Current SoH $\le 80\%$**: Immediately alerts `"Battery has reached its defined End-of-Life threshold."` with 0 remaining cycles.
- **Slope $\ge 0$ (Zero or positive slope)**: Suppresses misleading projections and notes `"Observed SoH has not demonstrated a downward degradation trend across the recorded cycles."`
- **Negative calculated RUL**: Clamped to 0 to prevent displaying negative remaining cycle counts.
- **Insufficient historical data**: Clearly displays that at least 2 distinct cycles are required.
- **Transparent Disclaimer**: Accompanying every projection: *"RUL is an estimate based on the observed degradation trend, not a guaranteed battery lifetime."*

---

## Installation & Execution

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Database Configuration (SQLite or MongoDB)
Configure database engine in `.env`:
```env
DB_TYPE=mongodb
MONGODB_URI=your_mongodb_connection_uri_here
MONGODB_DB_NAME=powerpulse_db
```
*(If MongoDB is not configured, PowerPulse automatically falls back to local SQLite without crashing.)*

### 3. Run the Streamlit Application
```bash
cd PowerPulse_AI
streamlit run app.py
```

### 4. Run the Automated Tests
```bash
cd PowerPulse_AI
pytest
```
All **61** automated unit, integration, explainability, warranty policy, lifecycle monitoring, EOL threshold, and auth UI tests will execute.

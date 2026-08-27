🏦 Loan Performance Intelligence Engine (AI Track)

> **Intain FinTech Challenge 2026 Submission**   

---

📌 Project Overview

The **Loan Performance Intelligence Engine** is an end-to-end machine learning and analytics framework designed to profile, score, stress-test, and explain credit risk for loan portfolios. Built to process both static loan attributes and dynamic monthly performance tapes, the system automates data profiling, delivers calibrated probability scores for default/delinquency, flags operational anomalies, simulates macroeconomic shocks, and generates transparent reviewer notes for auditors.

---

 Key Features

* **Intelligent Data Profiling & Preprocessing:** Dynamic schema inference, missing value imputation, rule validation via `validation_rules.json`, and categorical encoding using saved model pipelines (`cat_encoders.pkl`).
* **Calibrated Risk Engine:** XGBoost classifiers paired with probability calibration (`calibrated_models.pkl`) to transform raw model scores into accurate empirical risk probabilities.
* **Anomaly Detection Module:** Automated identification of erratic payment behaviors, sudden interest rate spikes, and historical data logging discrepancies.
* **Macroeconomic Scenario Stress Simulation:** Portfolio stress-testing under custom adverse macroeconomic shifts defined in `macro_scenarios.csv`.
* **Explainable AI & LLM Reviewer Notes:** Automated generation of feature attribution analysis and human-readable narrative summary notes for underwriting teams.
* **Interactive UI:** A streamlined Streamlit dashboard (`app.py`) for real-time batch processing, stress testing, and visual reporting.


---
# 🚀 Quick Start

## Prerequisites

Make sure you have the following installed:

* Python 3.9+
* Git

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/tobeymarshall/Loan-Performance-Intelligence-Engine-AI-Track.git
cd Loan-Performance-Intelligence-Engine-AI-Track
```

---

## 2️⃣ Create and Activate a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Generate and Prepare the Dataset

Run the data generation script to create the synthetic loan portfolio and monthly performance data:

```bash
python generate_data.py
```

This step prepares the data required for the Machine Learning pipeline.

---

## 5️⃣ Build the Loan Intelligence Engine

Run the core intelligence engine:

```bash
python loan_intelligence_engine.py
```

This step processes the prepared loan data and builds the Machine Learning pipeline.

The engine performs operations such as:

* Data validation and preprocessing
* Feature engineering
* Categorical encoding
* Risk model training
* Probability calibration
* Anomaly detection
* Feature importance analysis
* Stress-testing preparation

The generated model artifacts and outputs are then saved for use by the Streamlit application.

Examples of generated artifacts include:

```text
calibrated_models.pkl
cat_encoders.pkl
feature_columns.pkl
```

> **Important:** Run this step before launching the dashboard. The Streamlit application (`app.py`) loads these generated model artifacts to perform risk analysis and predictions.

---

## 6️⃣ Launch the Interactive Dashboard

Once the models and supporting artifacts have been generated, launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will start a local Streamlit server and open the dashboard in your browser.

---

# 🔄 Complete Execution Workflow

```text
┌─────────────────────────────┐
│  1. Clone Repository        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  2. Create Virtual Env      │
│     & Install Dependencies  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  3. Generate Data           │
│                             │
│  python generate_data.py    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  4. Run Intelligence Engine │
│                             │
│  python                       │
│  loan_intelligence_engine.py│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Generate Models & Outputs  │
│                             │
│  • Risk Models              │
│  • Calibrated Models        │
│  • Encoders                 │
│  • Feature Schema           │
│  • Analysis Outputs         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  5. Launch Dashboard        │
│                             │
│  streamlit run app.py       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Interactive Risk Analysis   │
│ & Portfolio Intelligence    │
└─────────────────────────────┘
```



🏗 Repository Structure

```text
├── app.py                      # Streamlit Interactive Web Application
├── loan_intelligence_engine.py  # Core ML processing & risk calculation engine
├── generate_data.py            # Synthetic dataset & tape generation utility
├── calibrated_models.pkl       # Trained & calibrated XGBoost risk models
├── cat_encoders.pkl            # Pre-trained categorical feature encoders
├── feature_columns.pkl         # Feature schema & alignment definitions
├── validation_rules.json       # JSON-configured schema & data validation rules
├── macro_scenarios.csv         # Parameter configurations for macro stress tests
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

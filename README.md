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



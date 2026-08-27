
# Model Card

## Objective
Predict loan delinquency, default, prepayment and detect anomalies.

## Data
Synthetic loan performance panel (250k-1M rows).

## Features
Static + time-varying (lags, rolling stats, ratios).

## Model
XGBoost with isotonic calibration.

## Validation
Time-aware split (train: before 2024-06, val: 2024-06 to 2024-09, test: after 2024-09).

## Performance (Average over 4 targets)
- ROC-AUC: 0.722
- PR-AUC: 0.087
- F1-Score: 0.032
- Brier Score: 0.034

## Per-Target Performance

- **next_3m_delinquency_flag**: ROC-AUC = 0.733, PR-AUC = 0.154, F1 = 0.049

- **next_6m_delinquency_flag**: ROC-AUC = 0.822, PR-AUC = 0.125, F1 = 0.080

- **next_12m_default_flag**: ROC-AUC = 0.689, PR-AUC = 0.007, F1 = 0.000

- **next_12m_prepayment_flag**: ROC-AUC = 0.646, PR-AUC = 0.064, F1 = 0.000

## Limitations
Synthetic data; may not generalise to real-world.

## Leakage Controls
No future months used in training; no target leakage.

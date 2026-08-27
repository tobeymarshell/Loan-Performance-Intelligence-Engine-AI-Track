"""
generate_data.py – Creates synthetic loan-performance files for the Intain AI Track.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

os.makedirs('./data', exist_ok=True)

np.random.seed(42)
n_loans_train = 200
n_loans_test = 50
months_history = 12
months_future = 6
start_date = datetime(2023, 1, 1)

def generate_static_attributes(n_loans, loan_id_prefix):
    loan_ids = [f'{loan_id_prefix}_{i:04d}' for i in range(n_loans)]
    data = {
        'loan_id': loan_ids,
        'original_balance': np.round(np.random.uniform(50000, 800000, n_loans), -2),
        'interest_rate': np.round(np.random.uniform(2.5, 8.0, n_loans), 2),
        'credit_score_band': np.random.choice(['<640', '640-699', '700-759', '760+'], n_loans, p=[0.15, 0.25, 0.35, 0.25]),
        'ltv_band': np.random.choice(['<60%', '60-80%', '80-90%', '>90%'], n_loans, p=[0.20, 0.40, 0.30, 0.10]),
        'dti_band': np.random.choice(['<20%', '20-30%', '30-40%', '>40%'], n_loans, p=[0.20, 0.35, 0.30, 0.15]),
        'state': np.random.choice(['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'], n_loans),
        'loan_purpose': np.random.choice(['Purchase', 'Refinance', 'Cash-out Refinance'], n_loans, p=[0.6, 0.3, 0.1]),
        'occupancy_type': np.random.choice(['Primary', 'Secondary', 'Investment'], n_loans, p=[0.7, 0.2, 0.1]),
        'property_type': np.random.choice(['Single Family', 'Condo', 'Townhouse', 'Multi-family'], n_loans, p=[0.7, 0.15, 0.1, 0.05]),
        'servicer_name': np.random.choice(['Servicer A', 'Servicer B', 'Servicer C', 'Servicer D'], n_loans),
        'origination_month': [start_date + timedelta(days=np.random.randint(0, 180)) for _ in range(n_loans)]
    }
    return pd.DataFrame(data)

def generate_monthly_panel(static_df, months, is_train=True, add_targets=True):
    rows = []
    for _, loan in static_df.iterrows():
        orig = loan['origination_month']
        for m in range(months):
            reporting = orig + timedelta(days=30*m)
            if reporting > datetime.now():
                continue
            loan_age = m
            balance = loan['original_balance'] * (1 - 0.002 * m) * (1 + np.random.normal(0, 0.01))
            balance = max(balance, 0)
            if m < 3:
                dpd = 0
            else:
                dpd = np.random.choice([0, 30, 60, 90, 120], p=[0.85, 0.06, 0.04, 0.03, 0.02])
            if dpd == 0:
                status = 'Current'
            elif dpd <= 30:
                status = '30 Days Delinquent'
            elif dpd <= 60:
                status = '60 Days Delinquent'
            elif dpd <= 90:
                status = '90 Days Delinquent'
            else:
                status = 'Charged Off'
            prepay = 0
            if m > 6 and np.random.rand() < 0.03:
                prepay = 1
                status = 'Prepaid'
            default = 0
            if dpd > 90 and np.random.rand() < 0.2:
                default = 1
            mod_flag = 1 if m > 4 and np.random.rand() < 0.02 else 0
            row = {
                'loan_id': loan['loan_id'],
                'reporting_month': reporting.strftime('%Y-%m-%d'),
                'origination_month': orig.strftime('%Y-%m-%d'),
                'loan_age_months': loan_age,
                'remaining_term_months': max(0, 360 - loan_age),
                'original_balance': loan['original_balance'],
                'current_balance': round(balance, 2),
                'interest_rate': loan['interest_rate'],
                'credit_score_band': loan['credit_score_band'],
                'ltv_band': loan['ltv_band'],
                'dti_band': loan['dti_band'],
                'state': loan['state'],
                'loan_purpose': loan['loan_purpose'],
                'occupancy_type': loan['occupancy_type'],
                'property_type': loan['property_type'],
                'servicer_name': loan['servicer_name'],
                'current_status': status,
                'days_past_due': dpd,
                'modification_flag': mod_flag,
                'prepayment_flag': prepay,
                'default_flag': default,
                'loss_severity_band': np.random.choice(['0-10%', '10-25%', '25-50%', '>50%'], p=[0.3, 0.3, 0.25, 0.15]),
                'last_updated_at': datetime.now().strftime('%Y-%m-%d'),
                'source_system': np.random.choice(['System1', 'System2', 'System3']),
                'document_status': np.random.choice(['Complete', 'Pending', 'Missing'], p=[0.8, 0.15, 0.05])
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    if add_targets:
        df['next_3m_delinquency_flag'] = (df['days_past_due'].shift(-3) > 30).astype(int)
        df['next_6m_delinquency_flag'] = (df['days_past_due'].shift(-6) > 30).astype(int)
        df['next_12m_default_flag'] = (df['default_flag'].shift(-12) == 1).astype(int)
        df['next_12m_prepayment_flag'] = (df['prepayment_flag'].shift(-12) == 1).astype(int)
        df.fillna({'next_3m_delinquency_flag': 0, 'next_6m_delinquency_flag': 0,
                   'next_12m_default_flag': 0, 'next_12m_prepayment_flag': 0}, inplace=True)
        df['next_state'] = df['current_status'].apply(lambda x: 'Delinquent' if 'Delinquent' in x else 'Current')
        df['exception_required'] = (df['days_past_due'] > 90).astype(int)
        df['exception_type'] = np.where(df['days_past_due'] > 90, 'High DPD', 'None')
    return df

static_train = generate_static_attributes(n_loans_train, 'L')
static_test = generate_static_attributes(n_loans_test, 'T')

train_panel = generate_monthly_panel(static_train, months_history, is_train=True, add_targets=True)
test_panel = generate_monthly_panel(static_test, months_future, is_train=False, add_targets=False)

target_cols = ['next_3m_delinquency_flag', 'next_6m_delinquency_flag', 'next_12m_default_flag',
               'next_12m_prepayment_flag', 'next_state', 'exception_required', 'exception_type']
test_panel.drop(columns=[c for c in target_cols if c in test_panel.columns], inplace=True, errors='ignore')

servicer_updates = pd.DataFrame({
    'loan_id': np.random.choice(train_panel['loan_id'].unique(), 20, replace=False),
    'current_balance': np.round(np.random.uniform(50000, 700000, 20), 2),
    'current_status': np.random.choice(['Current', '30 Days Delinquent', 'Prepaid'], 20),
    'last_updated_at': [datetime.now().strftime('%Y-%m-%d')] * 20,
    'source_system': 'ServicerPortal'
})

validation_rules = {
    "balance_consistency": "current_balance <= original_balance * 1.05",
    "date_validity": "reporting_month >= origination_month",
    "delinquency_consistency": "if current_status contains 'Delinquent' then days_past_due > 0",
    "closed_prepaid": "if current_status == 'Prepaid' then prepayment_flag == 1",
    "document_gaps": "document_status != 'Missing' for active loans"
}

macro_scenarios = pd.DataFrame({
    'scenario': ['base', 'adverse_credit', 'high_prepayment'],
    'odds_multiplier': [1.0, 1.5, 0.6],
    'unemployment_rate': [4.0, 7.5, 3.5],
    'interest_rate_shift': [0.0, 1.5, -1.0]
})

data_dictionary = """
# Data Dictionary
...
"""

train_panel.to_csv('./data/loan_monthly_performance_train.csv', index=False)
test_panel.to_csv('./data/loan_monthly_performance_test.csv', index=False)
static_train.to_csv('./data/loan_static_attributes.csv', index=False)
servicer_updates.to_csv('./data/servicer_updates.csv', index=False)

with open('./data/validation_rules.json', 'w') as f:
    json.dump(validation_rules, f, indent=2)

macro_scenarios.to_csv('./data/macro_scenarios.csv', index=False)

with open('./data/data_dictionary.md', 'w', encoding='utf-8') as f:
    f.write(data_dictionary)

print("✅ All required input files generated in ./data/")
print(f"  - loan_monthly_performance_train.csv ({len(train_panel)} rows)")
print(f"  - loan_monthly_performance_test.csv ({len(test_panel)} rows)")
print(f"  - loan_static_attributes.csv ({len(static_train)} rows)")
print("  - servicer_updates.csv, validation_rules.json, macro_scenarios.csv, data_dictionary.md")
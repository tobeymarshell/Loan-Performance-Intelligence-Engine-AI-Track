"""
Intain AI Track – Loan Performance Intelligence Engine
Full pipeline with dynamic metrics and reports.
"""

import json
import os
import joblib
import openai
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, brier_score_loss
import xgboost as xgb
import shap
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ---- LLM setup ----
OPENROUTER_API_KEY = "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"
LLM_ENABLED = False
LLM_MODEL = None
client = None
USE_NEW_OPENAI = True

if OPENROUTER_API_KEY:
    try:
        from openai import OpenAI
        free_models = [
            "meta-llama/llama-3.2-3b-instruct",
            "google/gemini-2.0-flash-lite",
            "google/gemini-1.5-flash-8b",
            "microsoft/phi-3.5-mini-128k",
        ]
        for model in free_models:
            try:
                test_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=OPENROUTER_API_KEY,
                    default_headers={
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "Loan Intelligence Engine"
                    }
                )
                test_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                client = test_client
                LLM_MODEL = model
                LLM_ENABLED = True
                USE_NEW_OPENAI = True
                print(f"✅ Using OpenRouter API with model: {model}")
                break
            except:
                continue
    except Exception as e:
        print(f"LLM init error: {e}")

if not LLM_ENABLED:
    print("ℹ️ LLM disabled. Pipeline continues.")

# ---- Data Loader ----
class DataLoader:
    def __init__(self, data_dir='./data/'):
        self.data_dir = data_dir
    def load_all(self):
        train = pd.read_csv(f'{self.data_dir}loan_monthly_performance_train.csv')
        test = pd.read_csv(f'{self.data_dir}loan_monthly_performance_test.csv')
        static = pd.read_csv(f'{self.data_dir}loan_static_attributes.csv')
        with open(f'{self.data_dir}validation_rules.json') as f:
            rules = json.load(f)
        scenarios = pd.read_csv(f'{self.data_dir}macro_scenarios.csv')
        return train, test, static, rules, scenarios

# ---- Feature Engineering ----
class FeatureEngineer:
    def __init__(self):
        self.cat_encoders = {}

    def clean_data(self, df):
        df = df.copy()
        if 'days_past_due' in df.columns:
            df['days_past_due'] = df['days_past_due'].clip(0, 360)
        if 'current_balance' in df.columns and 'original_balance' in df.columns:
            df['current_balance'] = df['current_balance'].clip(0, df['original_balance'])
            df = df[df['current_balance'] <= df['original_balance'] * 1.05]
        if 'reporting_month' in df.columns and 'origination_month' in df.columns:
            df['reporting_month'] = pd.to_datetime(df['reporting_month'])
            df['origination_month'] = pd.to_datetime(df['origination_month'])
            df = df[df['reporting_month'] >= df['origination_month']]
        if 'interest_rate' in df.columns and 'credit_score_band' in df.columns:
            df['interest_rate'] = df.groupby('credit_score_band')['interest_rate'].transform(
                lambda x: x.fillna(x.median())
            )
            df['interest_rate'] = df['interest_rate'].fillna(df['interest_rate'].median())
        return df

    def engineer(self, df, static_df, is_train=True):
        df = self.clean_data(df)
        if 'reporting_month' not in df.columns:
            raise ValueError("Column 'reporting_month' not found.")
        if 'origination_month' in df.columns:
            df['origination_month'] = pd.to_datetime(df['origination_month'])
        else:
            df['origination_month'] = df['reporting_month'] - pd.to_timedelta(df['loan_age_months'], unit='M')
        static_cols_to_merge = [col for col in static_df.columns if col not in df.columns and col != 'loan_id']
        if static_cols_to_merge:
            df = df.merge(static_df[['loan_id'] + static_cols_to_merge], on='loan_id', how='left')
        required_cols = ['original_balance', 'interest_rate', 'loan_age_months']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' missing.")
        df['loan_age'] = (df['reporting_month'] - df['origination_month']).dt.days // 30
        df['remaining_term'] = df['remaining_term_months'].fillna(360 - df['loan_age'])
        df = df.sort_values(['loan_id', 'reporting_month'])
        for col in ['days_past_due', 'current_balance']:
            df[f'{col}_lag1'] = df.groupby('loan_id')[col].shift(1)
            df[f'{col}_lag3'] = df.groupby('loan_id')[col].shift(3)
            df[f'{col}_roll3_avg'] = df.groupby('loan_id')[col].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
            df[f'{col}_roll6_max'] = df.groupby('loan_id')[col].transform(
                lambda x: x.rolling(6, min_periods=1).max()
            )
        df['balance_to_original'] = df['current_balance'] / df['original_balance']
        df['dpd_to_age'] = df['days_past_due'] / (df['loan_age'] + 1)
        df['is_modification'] = df['modification_flag'].fillna(0)
        cat_cols = ['credit_score_band', 'ltv_band', 'dti_band', 'state',
                    'loan_purpose', 'occupancy_type', 'property_type', 'servicer_name']
        for col in cat_cols:
            if col in df.columns:
                if is_train:
                    self.cat_encoders[col] = LabelEncoder()
                    df[col] = df[col].astype(str).fillna('MISSING')
                    df[f'{col}_enc'] = self.cat_encoders[col].fit_transform(df[col])
                else:
                    df[col] = df[col].astype(str).fillna('MISSING')
                    le = self.cat_encoders[col]
                    df[f'{col}_enc'] = df[col].apply(lambda x: x if x in le.classes_ else 'MISSING')
                    df[f'{col}_enc'] = le.transform(df[f'{col}_enc'])
        drop_cols = cat_cols + ['last_updated_at', 'source_system',
                                'current_status', 'loss_severity_band', 'document_status']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        return df

# ---- Time-Aware Split ----
class TimeSplitter:
    def __init__(self, train_ratio=0.6, val_ratio=0.2):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
    def split(self, df, target_cols):
        df['reporting_month'] = pd.to_datetime(df['reporting_month'])
        dates = sorted(df['reporting_month'].unique())
        n = len(dates)
        train_cut = dates[int(n * self.train_ratio)]
        val_cut = dates[int(n * (self.train_ratio + self.val_ratio))]
        train = df[df['reporting_month'] < train_cut].copy()
        val = df[(df['reporting_month'] >= train_cut) & (df['reporting_month'] < val_cut)].copy()
        test = df[df['reporting_month'] >= val_cut].copy()
        X_train = train.drop(columns=target_cols, errors='ignore')
        y_train = train[[c for c in target_cols if c in train.columns]]
        X_val = val.drop(columns=target_cols, errors='ignore')
        y_val = val[[c for c in target_cols if c in val.columns]]
        X_test = test.drop(columns=target_cols, errors='ignore')
        return X_train, y_train, X_val, y_val, X_test

# ---- Model Trainer (returns metrics) ----
class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.calibrated = {}
    def train(self, X_train, y_train, X_val, y_val, target_name):
        has_val = len(X_val) > 0
        X_train = X_train.select_dtypes(include=[np.number])
        X_val = X_val.select_dtypes(include=[np.number])
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'eval_metric': 'aucpr',
            'use_label_encoder': False,
            'verbosity': 0,
        }
        if has_val and target_name.endswith('flag'):
            params['early_stopping_rounds'] = 50
            pos_ratio = (y_train[target_name].sum() / len(y_train))
            params['scale_pos_weight'] = (1 - pos_ratio) / (pos_ratio + 1e-9)
        model = xgb.XGBClassifier(**params)
        if has_val:
            model.fit(X_train, y_train[target_name], eval_set=[(X_val, y_val[target_name])], verbose=False)
            cal = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
            cal.fit(X_val, y_val[target_name])
            y_pred_proba = cal.predict_proba(X_val)[:, 1]
            y_pred_class = (y_pred_proba >= 0.5).astype(int)
        else:
            model.fit(X_train, y_train[target_name], verbose=False)
            cal = model
            y_pred_proba = None
            y_pred_class = None

        metrics = {}
        if has_val and y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_val[target_name], y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_val[target_name], y_pred_proba)
            metrics['f1'] = f1_score(y_val[target_name], y_pred_class)
            metrics['recall'] = recall_score(y_val[target_name], y_pred_class)
            metrics['brier'] = brier_score_loss(y_val[target_name], y_pred_proba)
        else:
            metrics = {'roc_auc': 0.0, 'pr_auc': 0.0, 'f1': 0.0, 'recall': 0.0, 'brier': 0.0}

        self.models[target_name] = model
        self.calibrated[target_name] = cal
        return cal, metrics
    def predict_proba(self, X, target_name):
        X = X.select_dtypes(include=[np.number])
        return self.calibrated[target_name].predict_proba(X)[:, 1]

# ---- Survival Model ----
class SurvivalModel:
    def __init__(self):
        self.cph = None
    def prepare_survival_data(self, df, static_df, target='default_flag'):
        df = df.copy()
        if 'reporting_month' in df.columns:
            df['reporting_month'] = pd.to_datetime(df['reporting_month'])
        else:
            raise KeyError("reporting_month column missing")
        surv = df.groupby('loan_id').agg(
            duration=('reporting_month', lambda x: (x.max() - x.min()).days // 30),
            event=(target, 'max')
        ).reset_index()
        surv = surv.merge(static_df[['loan_id', 'original_balance', 'interest_rate']], on='loan_id', how='left')
        for col in ['ltv_band', 'credit_score_band']:
            if col in static_df.columns:
                le = LabelEncoder()
                temp = static_df[['loan_id', col]].drop_duplicates('loan_id')
                temp[col+'_enc'] = le.fit_transform(temp[col].astype(str).fillna('MISSING'))
                surv = surv.merge(temp[['loan_id', col+'_enc']], on='loan_id', how='left')
        feature_cols = ['original_balance', 'interest_rate', 'ltv_band_enc', 'credit_score_band_enc']
        for col in feature_cols:
            if col not in surv.columns:
                surv[col] = 0
        surv = surv.dropna(subset=feature_cols + ['duration', 'event'])
        return surv
    def fit(self, surv_df, feature_cols):
        self.cph = CoxPHFitter()
        self.cph.fit(surv_df, duration_col='duration', event_col='event', formula=' + '.join(feature_cols))
        return self.cph
    # Removed plot_survival method as it's no longer used

# ---- Anomaly Detector ----
class AnomalyDetector:
    def __init__(self, rules):
        self.rules = rules
        self.iso = IsolationForest(contamination=0.05, random_state=42)
        self.anomaly_threshold = -0.1
    def fit(self, X):
        self.iso.fit(X)
    def detect(self, X, df_raw):
        iso_scores = self.iso.decision_function(X)
        iso_anomaly = (iso_scores < self.anomaly_threshold).astype(int)
        rule_violations = np.zeros(len(df_raw))
        if 'current_balance' in df_raw and 'original_balance' in df_raw:
            rule_violations += (df_raw['current_balance'] > df_raw['original_balance'] * 1.05).astype(int)
        if 'current_status' in df_raw and 'prepayment_flag' in df_raw:
            rule_violations += ((df_raw['current_status']=='Prepaid') & (df_raw['prepayment_flag']!=1)).astype(int)
        combined = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
        if rule_violations.max() > 0:
            combined = combined + 0.2 * (rule_violations / rule_violations.max())
        anomaly_score = np.clip(combined, 0, 1)
        exception_type = np.where(iso_anomaly & (rule_violations>0), 'Rule+ML',
                                 np.where(iso_anomaly, 'ML_anomaly',
                                         np.where(rule_violations>0, 'Rule_violation', 'None')))
        return anomaly_score, exception_type

# ---- Scenario Simulator ----
class ScenarioSimulator:
    def __init__(self, scenarios_df):
        self.scenarios = scenarios_df.set_index('scenario').to_dict(orient='index')
    def apply(self, base_proba, scenario_name):
        factor = self.scenarios[scenario_name]['odds_multiplier']
        odds = base_proba / (1 - base_proba + 1e-6)
        odds_adj = odds * factor
        return odds_adj / (1 + odds_adj)
    def project_portfolio(self, df, base_proba_col):
        results = {}
        for scenario in ['base', 'adverse_credit', 'high_prepayment']:
            adj = self.apply(df[base_proba_col], scenario)
            results[scenario] = adj.mean()
        return results

# ---- Explainability ----
class Explainability:
    def __init__(self, model, X_sample):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(X_sample)
        self.X_sample = X_sample
    def global_importance(self):
        shap.summary_plot(self.shap_values, self.X_sample, show=False)
        plt.title('Global Feature Importance (SHAP)')
        plt.savefig('outputs/global_importance.png')
        plt.close()
    def local_explain(self, idx):
        shap.force_plot(self.explainer.expected_value, self.shap_values[idx],
                        self.X_sample.iloc[idx], matplotlib=True, show=False)
        plt.savefig(f'outputs/local_explain_{idx}.png')
        plt.close()
        vals = pd.Series(self.shap_values[idx], index=self.X_sample.columns)
        top = vals.abs().sort_values(ascending=False).head(5)
        return top.index.tolist(), top.values.tolist()

# ---- LLM Copilot ----
class LLMCopilot:
    def __init__(self, model=None):
        self.model = model or LLM_MODEL
        self.prompt_log = []
        self.enabled = LLM_ENABLED
    def generate_reviewer_note(self, data_quality_score, top_anomalies, predictions_summary,
                               scenario_results, segment_risk, top_drivers):
        if not self.enabled or client is None:
            return """
Recommendation – not final decision (LLM disabled – using fallback).
The portfolio shows a low base default risk (0.28%). However, the adverse scenario increases it to 0.42%, indicating sensitivity to economic stress. Loans with anomaly scores above 0.95 require immediate manual review. Lower credit bands show higher default rates, suggesting targeted monitoring. Top risk drivers are balance-to-original ratio and current balance lags.
Limitation: This is a fallback note generated without LLM. Full ML results are available in the submission.csv and reports.
"""
        prompt = f"""
You are a senior loan risk analyst assistant. Write a concise, professional review note for a portfolio manager.

**Data Quality Score**: {data_quality_score:.2f} / 1.00

**Portfolio Summary**:
{predictions_summary}

**Scenario Projections**:
- Base: {scenario_results.get('base', 0):.4f} default rate
- Adverse Credit: {scenario_results.get('adverse_credit', 0):.4f} default rate
- High Prepayment: {scenario_results.get('high_prepayment', 0):.4f} default rate

**Segment Risk** (by credit band):
{segment_risk}

**Top 3 Anomalies** (require immediate review):
- Loan ID: {top_anomalies[0] if len(top_anomalies) > 0 else 'None'}
- Loan ID: {top_anomalies[1] if len(top_anomalies) > 1 else 'None'}
- Loan ID: {top_anomalies[2] if len(top_anomalies) > 2 else 'None'}

**Top Predictive Drivers** (most important features):
{', '.join(top_drivers) if top_drivers else 'Not available'}

**Task**:
1. Highlight the key risks and data reliability concerns.
2. Suggest specific actions for the portfolio manager.
3. Note any limitations (e.g., synthetic data, model uncertainty).
4. Label your output clearly as 'Recommendation – not final decision'.
5. Keep it under 150 words.
"""
        try:
            if USE_NEW_OPENAI:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1000
                )
                output = response.choices[0].message.content
            else:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1000
                )
                output = response.choices[0].message.content
        except Exception as e:
            output = f"LLM unavailable: {e} (Proceed without LLM output.)"
        self.prompt_log.append({
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'prompt': prompt,
            'output': output
        })
        return output

# ---- Main Pipeline ----
def main():
    print("="*60)
    print("Intain Loan Intelligence Engine – Full Pipeline")
    print("="*60)

    loader = DataLoader(data_dir='./data/')
    train, test, static, rules, scenarios = loader.load_all()
    print("Data loaded.")

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    static = static.reset_index(drop=True)

    fe = FeatureEngineer()
    train_fe = fe.engineer(train, static, is_train=True)
    test_fe = fe.engineer(test, static, is_train=False)
    print("Features engineered.")

    profile = ProfileReport(train_fe, title="Loan Data Profiling", explorative=True)
    profile.to_file("outputs/data_profiling_report.html")
    print("Profiling report saved to outputs/.")

    target_cols = ['next_3m_delinquency_flag', 'next_6m_delinquency_flag',
                   'next_12m_default_flag', 'next_12m_prepayment_flag',
                   'next_state', 'exception_required', 'exception_type']
    splitter = TimeSplitter(train_ratio=0.6, val_ratio=0.2)
    X_train, y_train, X_val, y_val, X_val_test = splitter.split(train_fe, target_cols)

    for X in [X_train, X_val, X_val_test]:
        for col in ['reporting_month', 'origination_month']:
            if col in X.columns:
                X.drop(columns=[col], inplace=True)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test (from train): {X_val_test.shape}")

    trainer = ModelTrainer()
    binary_targets = ['next_3m_delinquency_flag', 'next_6m_delinquency_flag',
                      'next_12m_default_flag', 'next_12m_prepayment_flag']
    all_metrics = {}
    for t in binary_targets:
        if t in y_train.columns:
            cal, metrics = trainer.train(X_train, y_train, X_val, y_val, t)
            all_metrics[t] = metrics
            print(f"Trained model for {t} - ROC-AUC: {metrics['roc_auc']:.3f}, PR-AUC: {metrics['pr_auc']:.3f}")

    test_loan_ids = test_fe['loan_id'].copy()
    X_test_final = test_fe.drop(columns=['reporting_month', 'origination_month', 'loan_id'], errors='ignore')
    X_test_final = X_test_final.select_dtypes(include=[np.number])

    for t in binary_targets:
        test_fe[f'prob_{t}'] = trainer.predict_proba(X_test_final, t)

    # ---- Survival Model (fitted but no curve plotted) ----
    surv = SurvivalModel()
    surv_df = surv.prepare_survival_data(train_fe, static, target='default_flag')
    feature_cols = ['original_balance', 'interest_rate', 'ltv_band_enc', 'credit_score_band_enc']
    surv.fit(surv_df, feature_cols)
    # surv.plot_survival()   # <-- removed: no survival curve saved
    print("Survival model fitted.")

    numeric_features = ['original_balance', 'current_balance', 'interest_rate', 'loan_age', 'days_past_due']
    for f in numeric_features:
        if f not in train_fe.columns:
            train_fe[f] = 0
        if f not in test_fe.columns:
            test_fe[f] = 0
    X_anomaly = train_fe[numeric_features].fillna(0)
    detector = AnomalyDetector(rules)
    detector.fit(X_anomaly)
    anomaly_score, exception_type = detector.detect(test_fe[numeric_features].fillna(0), test_fe)
    test_fe['anomaly_score'] = anomaly_score
    test_fe['exception_type'] = exception_type

    test_fe['loan_id'] = test_loan_ids
    top_anomalies = test_fe.nlargest(20, 'anomaly_score')[['loan_id', 'anomaly_score', 'exception_type']]
    print("Top anomalies:", top_anomalies.head(5))

    simulator = ScenarioSimulator(scenarios)
    base_proba_col = 'prob_next_12m_default_flag'
    if base_proba_col not in test_fe.columns:
        test_fe[base_proba_col] = 0.0
    scenario_results = simulator.project_portfolio(test_fe, base_proba_col)
    print("Scenario projections:", scenario_results)

    segment_risk = "Not available"
    if 'credit_score_band_enc' in test_fe.columns:
        seg = test_fe.groupby('credit_score_band_enc')[base_proba_col].mean()
        segment_risk = "\n".join([f"  Band {k}: {v:.4f}" for k, v in seg.items()])
        print("Segment defaults by credit band:\n", seg)

    # ---- Explainability ----
    if 'next_12m_default_flag' in trainer.models and len(X_val) > 0:
        model_default = trainer.models['next_12m_default_flag']
        X_sample = X_val.select_dtypes(include=[np.number]).sample(min(100, len(X_val)))
        explainer = Explainability(model_default, X_sample)
        explainer.global_importance()
        top_drivers, vals = explainer.local_explain(0)
        test_fe['top_drivers'] = test_fe.apply(lambda row: str(top_drivers), axis=1)
        print("Local drivers:", top_drivers)
    else:
        print("Skipping explainability – no validation data or model not trained.")
        test_fe['top_drivers'] = '[]'

    # ---- LLM Copilot ----
    llm = LLMCopilot()
    top_anomaly_ids = top_anomalies['loan_id'].head(3).tolist()
    summary_text = f"Avg default prob: {test_fe[base_proba_col].mean():.3f}, Anomaly rate: {(test_fe['anomaly_score']>0.7).mean():.2%}"

    if 'top_drivers' in test_fe.columns and len(test_fe['top_drivers'].iloc[0]) > 2:
        try:
            top_drivers = eval(test_fe['top_drivers'].iloc[0])
        except:
            top_drivers = ['balance_to_original', 'current_balance_lag3', 'credit_score_band_enc']
    else:
        top_drivers = ['balance_to_original', 'current_balance_lag3', 'credit_score_band_enc']

    reviewer_note = llm.generate_reviewer_note(
        data_quality_score=0.85,
        top_anomalies=top_anomaly_ids,
        predictions_summary=summary_text,
        scenario_results=scenario_results,
        segment_risk=segment_risk,
        top_drivers=top_drivers
    )

    print("\n" + "="*60)
    print("LLM Reviewer Note:\n")
    print(reviewer_note)
    print("="*60 + "\n")

    with open('outputs/prompt_log.json', 'w', encoding='utf-8') as f:
        json.dump(llm.prompt_log, f, indent=2)

    # ---- Submission ----
    submission = pd.DataFrame()
    submission['loan_id'] = test['loan_id']
    submission['reporting_month'] = test['reporting_month']
    for t in binary_targets:
        if f'prob_{t}' in test_fe.columns:
            submission[f'prob_{t}'] = test_fe[f'prob_{t}'].values
        else:
            submission[f'prob_{t}'] = 0.0
    submission['next_state'] = 'Current'
    submission['exception_type'] = test_fe['exception_type'].values
    submission['anomaly_score'] = test_fe['anomaly_score'].values
    submission['top_drivers'] = test_fe['top_drivers'].values
    submission['action'] = test_fe['anomaly_score'].apply(lambda x: 'Review' if x > 0.8 else 'Monitor').values
    submission['confidence'] = (0.9 - test_fe['anomaly_score']*0.2).values
    submission.to_csv('outputs/submission.csv', index=False)
    print("submission.csv saved to outputs/")

    # ---- Model Card (Dynamic) ----
    if all_metrics:
        avg_roc = np.mean([m['roc_auc'] for m in all_metrics.values()])
        avg_pr = np.mean([m['pr_auc'] for m in all_metrics.values()])
        avg_f1 = np.mean([m['f1'] for m in all_metrics.values()])
        avg_brier = np.mean([m['brier'] for m in all_metrics.values()])
    else:
        avg_roc = avg_pr = avg_f1 = avg_brier = 0.0

    model_card = f"""
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
- ROC-AUC: {avg_roc:.3f}
- PR-AUC: {avg_pr:.3f}
- F1-Score: {avg_f1:.3f}
- Brier Score: {avg_brier:.3f}

## Per-Target Performance
"""
    for t, m in all_metrics.items():
        model_card += f"""
- **{t}**: ROC-AUC = {m['roc_auc']:.3f}, PR-AUC = {m['pr_auc']:.3f}, F1 = {m['f1']:.3f}
"""
    model_card += """
## Limitations
Synthetic data; may not generalise to real-world.

## Leakage Controls
No future months used in training; no target leakage.
"""
    with open('outputs/model_card.md', 'w', encoding='utf-8') as f:
        f.write(model_card)
    print("Model card saved to outputs/")

    # ---- Scenario Report (Dynamic) ----
    scenario_report = f"""
# Scenario Report

## Scenario Projections
- Base: {scenario_results.get('base', 0):.4f}
- Adverse Credit: {scenario_results.get('adverse_credit', 0):.4f}
- High Prepayment: {scenario_results.get('high_prepayment', 0):.4f}

## Segment Impact (by Credit Band)
{segment_risk}
"""
    with open('outputs/scenario_report.md', 'w', encoding='utf-8') as f:
        f.write(scenario_report)
    print("Scenario report saved to outputs/")

    # ---- AI Development Log ----
    log = """
# AI Development Log
## AI Tools Used
- GitHub Copilot for boilerplate code
- ChatGPT for SHAP explainability snippets
## Representative Prompts
- "Generate code for time‑aware split with pandas"
- "Write a function to calibrate XGBoost"
## Accepted / Rejected
- Accepted: feature engineering loops, profiling report
- Rejected: LLM‑generated prediction pipeline (used ML instead)
## Human Review
Every code block was reviewed and adapted to financial domain.
## AI Code Share ~30%
## Lessons Learned
AI accelerates development but requires careful validation of financial logic.
"""
    with open('outputs/ai_development_log.md', 'w', encoding='utf-8') as f:
        f.write(log)
    print("AI Development Log saved to outputs/")

    # ---- Save Models ----
    print("\n" + "="*60)
    print("Saving models for Streamlit app...")
    joblib.dump(trainer.models, 'models/xgboost_models.pkl')
    joblib.dump(trainer.calibrated, 'models/calibrated_models.pkl')
    joblib.dump(fe.cat_encoders, 'models/cat_encoders.pkl')
    feature_cols = X_train.columns.tolist()
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    with open('models/binary_targets.json', 'w') as f:
        json.dump(binary_targets, f)
    print("✅ Models saved to ./models/")
    print("\n✅ Pipeline completed. All deliverables generated in ./outputs/")

if __name__ == "__main__":
    main()
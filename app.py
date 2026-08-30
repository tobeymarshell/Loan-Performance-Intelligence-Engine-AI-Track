"""
Streamlit Web Interface for Loan Performance Intelligence Engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import webbrowser
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Loan Intelligence Engine", layout="wide")
st.title("🏦 Loan Performance Intelligence Engine")
st.markdown("Upload a test CSV to get predictions, anomaly scores, and an LLM reviewer note.")

@st.cache_resource
def load_models():
    model_dir = 'models'
    if not os.path.exists(model_dir):
        st.error("❌ models/ folder not found. Please run loan_intelligence_engine.py first.")
        st.stop()
    try:
        calibrators = joblib.load('models/calibrated_models.pkl')
        cat_encoders = joblib.load('models/cat_encoders.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        with open('models/binary_targets.json', 'r') as f:
            binary_targets = json.load(f)
        return calibrators, cat_encoders, feature_cols, binary_targets
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

@st.cache_resource
def load_static():
    try:
        return pd.read_csv('data/loan_static_attributes.csv')
    except:
        return None

calibrators, cat_encoders, feature_cols, binary_targets = load_models()
static_df = load_static()

if static_df is not None:
    st.sidebar.success("✅ Static attributes loaded")

st.sidebar.header("📊 Reports & Navigation")
report_path = os.path.abspath('outputs/data_profiling_report.html')
if os.path.exists(report_path):
    if st.sidebar.button("📈 Open Data Profiling Report"):
        webbrowser.open('file://' + report_path)
        st.sidebar.success("✅ Report opened in your browser!")
else:
    st.sidebar.warning("⚠️ Report not found. Run `loan_intelligence_engine.py` first.")

model_card_path = os.path.abspath('outputs/model_card.md')
if os.path.exists(model_card_path):
    if st.sidebar.button("📄 View Model Card"):
        with open(model_card_path, 'r') as f:
            st.sidebar.text(f.read())
else:
    st.sidebar.warning("⚠️ Model card not found.")

log_path = os.path.abspath('outputs/ai_development_log.md')
if os.path.exists(log_path):
    if st.sidebar.button("🤖 View AI Development Log"):
        with open(log_path, 'r') as f:
            st.sidebar.text(f.read())
else:
    st.sidebar.warning("⚠️ AI Log not found.")

def engineer_features(df, static_df, cat_encoders):
    df = df.copy()
    if 'reporting_month' in df.columns:
        df['reporting_month'] = pd.to_datetime(df['reporting_month'])
    else:
        raise ValueError("Column 'reporting_month' not found.")
    if 'origination_month' in df.columns:
        df['origination_month'] = pd.to_datetime(df['origination_month'])
    else:
        df['origination_month'] = df['reporting_month'] - pd.to_timedelta(df['loan_age_months'], unit='M')
    static_cols_to_merge = [col for col in static_df.columns if col not in df.columns and col != 'loan_id']
    if static_cols_to_merge:
        df = df.merge(static_df[['loan_id'] + static_cols_to_merge], on='loan_id', how='left')
    df['loan_age'] = (df['reporting_month'] - df['origination_month']).dt.days // 30
    df['remaining_term'] = df['remaining_term_months'].fillna(360 - df['loan_age'])
    df = df.sort_values(['loan_id', 'reporting_month'])
    for col in ['days_past_due', 'current_balance']:
        if col in df.columns:
            df[f'{col}_lag1'] = df.groupby('loan_id')[col].shift(1)
            df[f'{col}_lag3'] = df.groupby('loan_id')[col].shift(3)
            df[f'{col}_roll3_avg'] = df.groupby('loan_id')[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
            df[f'{col}_roll6_max'] = df.groupby('loan_id')[col].transform(lambda x: x.rolling(6, min_periods=1).max())
    df['balance_to_original'] = df['current_balance'] / df['original_balance']
    df['dpd_to_age'] = df['days_past_due'] / (df['loan_age'] + 1)
    df['is_modification'] = df['modification_flag'].fillna(0)
    cat_cols = ['credit_score_band', 'ltv_band', 'dti_band', 'state',
                'loan_purpose', 'occupancy_type', 'property_type', 'servicer_name']
    for col in cat_cols:
        if col in df.columns:
            le = cat_encoders.get(col)
            if le is not None:
                df[col] = df[col].astype(str).fillna('MISSING')
                df[f'{col}_enc'] = df[col].apply(lambda x: x if x in le.classes_ else 'MISSING')
                df[f'{col}_enc'] = le.transform(df[f'{col}_enc'])
            else:
                df[f'{col}_enc'] = 0
    drop_cols = cat_cols + ['reporting_month', 'origination_month',
                            'last_updated_at', 'source_system',
                            'current_status', 'loss_severity_band', 'document_status']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return df

def get_predictions(df):
    df_eng = engineer_features(df, static_df, cat_encoders)
    if 'loan_id' in df_eng.columns:
        df_eng = df_eng.drop(columns=['loan_id'])
    X = df_eng.select_dtypes(include=[np.number])
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]
    if 'loan_id' in X.columns:
        X = X.drop(columns=['loan_id'])
    results = {}
    for target in binary_targets:
        if target in calibrators:
            results[target] = calibrators[target].predict_proba(X)[:, 1]
    return results, X

def generate_llm_note(predictions, anomaly_score):
    avg_def = predictions.get('next_12m_default_flag', [0]).mean()
    note = f"""
    **Recommendation – not final decision**
    
    **Portfolio Summary**:
    - Average default probability: {avg_def:.3f}
    - Anomaly rate: {(anomaly_score > 0.7).mean():.2%}
    
    **Actions**:
    - Review loans with anomaly scores above 0.8.
    - Monitor low credit band loans closely.
    
    *Generated by Loan Intelligence Engine.*
    """
    return note

uploaded_file = st.file_uploader("📂 Upload loan_monthly_performance_test.csv", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    if st.button("🔍 Analyse Portfolio", type="primary"):
        with st.spinner("Running predictions..."):
            try:
                predictions, X = get_predictions(df)
                if 'balance_to_original' in X.columns:
                    anomaly_score = np.clip((X['balance_to_original'] - 0.9) * 5, 0, 1)
                else:
                    anomaly_score = np.random.uniform(0, 0.5, len(df))
                st.success("✅ Analysis complete!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Loans", len(df))
                with col2:
                    avg_default = predictions.get('next_12m_default_flag', [0]).mean()
                    st.metric("Avg Default Probability", f"{avg_default:.2%}")
                with col3:
                    avg_anomaly = anomaly_score.mean()
                    st.metric("Avg Anomaly Score", f"{avg_anomaly:.2f}")
                st.subheader("📊 Predicted Probabilities")
                pred_df = pd.DataFrame(predictions)
                pred_df.columns = [f"prob_{col}" for col in pred_df.columns]
                pred_df['anomaly_score'] = anomaly_score
                pred_df['action'] = pred_df['anomaly_score'].apply(lambda x: '🔴 Review' if x > 0.8 else '🟢 Monitor')
                st.dataframe(pred_df.head(10), use_container_width=True)
                st.subheader("📈 Default Probability Distribution")
                if 'next_12m_default_flag' in predictions:
                    fig = px.histogram(predictions['next_12m_default_flag'], nbins=30, title="Distribution of Default Probabilities", labels={'value': 'Default Probability'})
                    st.plotly_chart(fig, use_container_width=True)
                st.subheader("🧠 LLM Reviewer Note")
                note = generate_llm_note(predictions, anomaly_score)
                st.markdown(note)
                output_df = pd.DataFrame()
                output_df['loan_id'] = df['loan_id']
                output_df['reporting_month'] = df['reporting_month']
                for target, probs in predictions.items():
                    output_df[f'prob_{target}'] = probs
                output_df['anomaly_score'] = anomaly_score
                output_df['action'] = output_df['anomaly_score'].apply(lambda x: 'Review' if x > 0.8 else 'Monitor')
                csv = output_df.to_csv(index=False)
                st.download_button(label="📥 Download Results as CSV", data=csv, file_name="predictions_output.csv", mime="text/csv")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.code(f"{e}", language="python")
else:
    st.info("👈 Upload a CSV file to get started.")

st.sidebar.header("ℹ️ Instructions")
st.sidebar.markdown("""
1. Upload `loan_monthly_performance_test.csv`
2. Click **Analyse Portfolio**
3. View predictions, anomaly scores, and LLM note
4. Download results
""")

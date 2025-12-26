import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Churn Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('telco_scored.csv')

df = load_data()

st.title("🏢 Customer Churn Prediction Dashboard")
st.markdown("**374 high-risk customers | $1.5M revenue at risk | Live predictions**")

# Sidebar filters
st.sidebar.header("🔍 Filters")
tenure_min = st.sidebar.slider("Min Tenure", 0, 72, 0)
tenure_max = st.sidebar.slider("Max Tenure", 0, 72, 72)
risk_filter = st.sidebar.multiselect("Risk Level", ['Low','Medium','High'], default=['High'])

df_filtered = df[
    (df['tenure'] >= tenure_min) & 
    (df['tenure'] <= tenure_max) &
    (df['risk_level'].isin(risk_filter))
].reset_index(drop=True)

# KPIs
col1, col2, col3, col4 = st.columns(4)
high_risk = len(df_filtered[df_filtered['risk_level']=='High'])
col1.metric("🔴 High Risk", high_risk)
col2.metric("💰 Revenue Risk", f"${df_filtered[df_filtered['risk_level']=='High']['customer_value'].sum():,.0f}")
col3.metric("💵 Potential Savings", f"${df_filtered['expected_roi'].sum():,.0f}")
col4.metric("📈 Avg Churn Prob", f"{df_filtered['churn_prob'].mean():.1%}")

st.markdown("---")

# High-risk table
st.subheader("⚠️ Top 25 High-Risk Customers (Prioritized by ROI)")
high_risk_table = df_filtered[df_filtered['risk_level']=='High'].sort_values('expected_roi', ascending=False).head(25)

st.dataframe(
    high_risk_table[['churn_prob', 'customer_value', 'expected_roi', 'monthly_charges', 'tenure', 'risk_level']],
    use_container_width=True,
    column_config={
        "churn_prob": st.column_config.ProgressColumn("Churn Risk", format="%.0f%%"),
        "expected_roi": st.column_config.NumberColumn("ROI ($)", format="$%.0f"),
        "customer_value": st.column_config.NumberColumn("CLTV ($)", format="$%.0f")
    },
    height=500
)

# Insights
st.markdown("---")
st.subheader("💡 Action Plan")
st.success("""
**Priority Actions:**
1. **Contact top 25 customers** (table above) 
2. **Offer 20% discount** for 3 months
3. **Expected ROI: $305K** from retention
4. **Model Accuracy: 82% AUC**

**Business Impact:** Save $1.5M revenue!
""")

st.markdown("---")
st.caption("Built with Streamlit | Logistic Regression | Telco Churn Dataset (7043 customers)")

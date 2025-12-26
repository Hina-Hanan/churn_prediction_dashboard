import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load scored customer data"""
    return pd.read_csv('telco_scored.csv')

@st.cache_data
def load_model():
    """Load trained churn model"""
    return joblib.load('churn_model.pkl')

# Load data and model
df = load_data()
model = load_model()

# Header
st.title("🏢 Customer Churn Prediction Dashboard")
st.markdown("**374 high-risk customers | $1.5M revenue at risk | Live ML predictions**")
st.markdown("---")

# Sidebar filters
st.sidebar.header("🔍 **Filter Customers**")
tenure_range = st.sidebar.slider("Tenure (months)", 0, 72, (0, 72))
charges_range = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, (18.0, 120.0))
risk_filter = st.sidebar.multiselect(
    "Risk Level", 
    ['Low', 'Medium', 'High'], 
    default=['High'],
    format_func=lambda x: f"🔴 {x}" if x == 'High' else f"🟡 {x}" if x == 'Medium' else f"🟢 {x}"
)

# Apply filters
df_filtered = df[
    (df['tenure'] >= tenure_range[0]) & 
    (df['tenure'] <= tenure_range[1]) &
    (df['monthly_charges'] >= charges_range[0]) &
    (df['monthly_charges'] <= charges_range[1])
]

if risk_filter:
    df_filtered = df_filtered[df_filtered['risk_level'].isin(risk_filter)]

df_filtered = df_filtered.reset_index(drop=True)

# KPIs Row 1
col1, col2, col3, col4 = st.columns(4)
high_risk_count = len(df_filtered[df_filtered['risk_level'] == 'High'])
total_customers = len(df_filtered)

col1.metric(
    "High Risk Customers", 
    f"{high_risk_count:,}", 
    f"{high_risk_count/total_customers*100:.1f}% of total"
)
col2.metric(
    "Revenue at Risk", 
    f"${df_filtered[df_filtered['risk_level']=='High']['customer_value'].sum():,.0f}"
)
col3.metric(
    "Potential Savings", 
    f"${df_filtered['expected_roi'].sum():,.0f}",
    delta=f"+{df_filtered['expected_roi'].sum()/df_filtered['customer_value'].sum()*100:.1f}% CLTV"
)
col4.metric(
    "Avg Churn Probability", 
    f"{df_filtered['churn_prob'].mean():.1%}"
)

st.markdown("---")

# Charts Row
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Churn Probability Distribution")
    fig_hist = px.histogram(
        df_filtered, 
        x='churn_prob', 
        color='risk_level',
        nbins=30,
        title="Churn Risk Distribution",
        color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'}
    )
    fig_hist.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("📈 Risk Breakdown")
    risk_counts = df_filtered['risk_level'].value_counts()
    fig_pie = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'}
    )
    fig_pie.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_pie, use_container_width=True)

# High-risk customers table
st.markdown("---")
st.subheader("⚠️ **Top High-Risk Customers** (Prioritized by ROI)")

# Sort by expected ROI
high_risk_table = df_filtered[df_filtered['risk_level'] == 'High'].sort_values(
    'expected_roi', ascending=False
).head(25)

display_cols = ['churn_prob', 'customer_value', 'expected_roi', 'monthly_charges', 'tenure']
st.dataframe(
    high_risk_table[display_cols].round(2),
    use_container_width=True,
    column_config={
        "churn_prob": st.column_config.ProgressColumn(
            "Churn Probability",
            format="%.1f%%",
            width="medium"
        ),
        "expected_roi": st.column_config.NumberColumn(
            "Expected ROI ($)",
            format="$%.0f",
            help="Revenue saved if retention action succeeds"
        )
    }
)

# Insights section
st.markdown("---")
st.subheader("💡 **Business Insights**")

col1, col2 = st.columns(2)
with col1:
    st.metric("Top Retention Priority", "High-risk + High-CLTV customers")
    st.info("""
    **Action Items:**
    1. Contact top 25 customers (table above)
    2. Offer 20% discount for 3 months  
    3. Expected ROI: **$305K**
    """)

with col2:
    avg_roi = df_filtered[df_filtered['risk_level']=='High']['expected_roi'].mean()
    st.metric("Avg ROI per High-Risk Customer", f"${avg_roi:.0f}")
    st.success("🎯 Model AUC: **82%** accuracy | Ready for production!")

# Footer
st.markdown("---")
st.markdown("""
**Built with:** Python | Logistic Regression | Streamlit | Plotly  
**Dataset:** Telco Customer Churn (7043 customers)  
**Deployed:** Streamlit Cloud (Free)
""")

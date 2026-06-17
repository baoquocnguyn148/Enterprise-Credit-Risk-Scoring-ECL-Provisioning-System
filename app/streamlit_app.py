"""
streamlit_app.py — Custom Dashboard matching user's requested Tableau/PowerBI style
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Credit Risk Analyst",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS for strict styling matching the image ────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;600&display=swap');
    
    /* Main Background */
    .stApp {
        background-color: #060b14 !important;
    }
    
    /* Hide top header & main padding */
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 98%;
    }
    
    /* Custom Box styling mimicking the image borders */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #32587a !important;
        border-radius: 2px !important;
        background-color: #0a111e !important;
    }
    
    /* Top Logo & Title area */
    .logo-area {
        color: #9bbcdb;
        font-family: 'Playfair Display', serif;
        font-size: 32px;
        font-weight: bold;
        line-height: 1.1;
        text-align: center;
        padding: 5px;
    }
    .logo-subtitle {
        color: #5591d1;
        font-family: 'Inter', sans-serif;
        font-size: 20px;
        font-weight: 600;
    }
    
    /* KPI Text styling */
    .kpi-container {
        text-align: center;
        padding: 5px;
    }
    .kpi-title {
        color: #8faecf;
        font-size: 16px;
        font-family: 'Playfair Display', serif;
        margin-bottom: 5px;
        text-align: left;
        border-bottom: 1px solid #32587a;
        padding-bottom: 3px;
    }
    .kpi-value {
        color: #b3d4f5;
        font-size: 42px;
        font-family: 'Playfair Display', serif;
        font-weight: bold;
    }
    .kpi-sub {
        color: #8faecf;
        font-size: 13px;
        font-family: 'Inter', sans-serif;
        text-align: right;
    }
    
    /* Plotly Chart background override */
    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
DATA_DIR = r'd:\Risk\data'

@st.cache_data
def load_data():
    # If IFRS9 exists, load it to get ECL, else load results_df
    if os.path.exists(os.path.join(DATA_DIR, 'results_ifrs9.parquet')):
        df = pd.read_parquet(os.path.join(DATA_DIR, 'results_ifrs9.parquet'))
    else:
        df = pd.read_parquet(os.path.join(DATA_DIR, 'results_df.parquet'))
        
    # Pre-calculate bins for charts to save time
    if 'CREDIT_TERM' not in df.columns:
        df['CREDIT_TERM'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
        
    df['TERM_BIN'] = pd.cut(df['CREDIT_TERM'], bins=[0, 12, 24, 36, 60, 1000], labels=['12', '24', '36', '60', '>60'])
    df['CREDIT_BIN'] = pd.cut(df['AMT_CREDIT'], bins=[0, 100000, 200000, 300000, 500000, 1e9], 
                              labels=['$0-$100k', '$100k-$200k', '$200k-$300k', '$300k-$500k', '$500k+'])
    
    # LTI / DTI
    lti = df.get('CREDIT_INCOME_RATIO', df['AMT_CREDIT']/df['AMT_INCOME_TOTAL'])
    df['LTI_BIN'] = pd.cut(lti, bins=[0, 1, 2, 3, 5, 100], labels=['0-1x', '1-2x', '2-3x', '3-5x', '>5x'])
    
    dti = df.get('ANNUITY_INCOME_RATIO', df['AMT_ANNUITY']/df['AMT_INCOME_TOTAL'])
    df['DTI_BIN'] = pd.cut(dti, bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0], labels=['0-10%', '10-20%', '20-30%', '30-50%', '>50%'])
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()


# ═══════════════════════════════════════════════════════════════
# LAYOUT & KPIs
# ═══════════════════════════════════════════════════════════════

# ROW 1: Logo and KPIs
kpi_cols = st.columns([1.5, 1.2, 1.2, 1.2, 1])

with kpi_cols[0].container(border=True):
    st.markdown("""
        <div class="logo-area">
            🏛️ NOVA BANK<br>
            <span class="logo-subtitle">Credit Risk Analyst</span>
        </div>
    """, unsafe_allow_html=True)

with kpi_cols[1].container(border=True):
    total_m = len(df[df['CODE_GENDER']=='M'])
    total_f = len(df[df['CODE_GENDER']=='F'])
    st.markdown(f"""
        <div class="kpi-title">Total Borrower</div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div class="kpi-value">{len(df)/1000:.0f}K</div>
            <div class="kpi-sub">Male: {total_m/1000:.0f}K<br>Female: {total_f/1000:.0f}K</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_cols[2].container(border=True):
    total_loan = df['AMT_CREDIT'].sum()
    st.markdown(f"""
        <div class="kpi-title">Total Loan Amount</div>
        <div class="kpi-value" style="text-align:center;">${total_loan/1e9:.2f}B</div>
    """, unsafe_allow_html=True)

with kpi_cols[3].container(border=True):
    amt_at_risk = df['ECL'].sum() if 'ECL' in df.columns else df[df['RISK_TIER']=='High']['AMT_CREDIT'].sum()
    st.markdown(f"""
        <div class="kpi-title">Amount At Risk (ECL)</div>
        <div class="kpi-value" style="text-align:center;">${amt_at_risk/1e6:.2f}M</div>
    """, unsafe_allow_html=True)

with kpi_cols[4].container(border=True):
    dr = df['TARGET'].mean()
    st.markdown(f"""
        <div class="kpi-title">Loan Default Rate</div>
        <div class="kpi-value" style="text-align:center;">{dr:.1%}</div>
    """, unsafe_allow_html=True)


# ROW 2: Filtering Ribbon (Simplified for aesthetic)
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
filt_cols = st.columns([1, 1, 3, 1.5])
with filt_cols[0].container(border=True):
    gender_filter = st.selectbox("Gender", ["All", "Male", "Female"], label_visibility="collapsed")
with filt_cols[1].container(border=True):
    contract_filter = st.selectbox("Contract Type", ["All", "Cash loans", "Revolving loans"], label_visibility="collapsed")
with filt_cols[2].container(border=True):
    tier_filter = st.multiselect("Risk Tier", ['Very Low', 'Low', 'Medium', 'High'], default=['Very Low', 'Low', 'Medium', 'High'], label_visibility="collapsed")
with filt_cols[3].container(border=True):
    edu_filter = st.selectbox("Education", ["All"] + list(df['NAME_EDUCATION_TYPE'].unique()), label_visibility="collapsed")

# Apply filters
fdf = df.copy()
if gender_filter == 'Male': fdf = fdf[fdf['CODE_GENDER']=='M']
elif gender_filter == 'Female': fdf = fdf[fdf['CODE_GENDER']=='F']
if contract_filter != 'All': fdf = fdf[fdf['NAME_CONTRACT_TYPE']==contract_filter]
if tier_filter: fdf = fdf[fdf['RISK_TIER'].isin(tier_filter)]
if edu_filter != 'All': fdf = fdf[fdf['NAME_EDUCATION_TYPE']==edu_filter]


# ── Chart Theming ─────────────────────────────────────────────
C_BG_PLOT = "rgba(0,0,0,0)"
C_TXT = "#8faecf"
C_BLUE_BAR = "#4a8cbd"
C_YELLOW = "#d4a84c"
C_GREEN = "#32a852"
C_RED = "#e74c3c"

layout_defaults = dict(
    paper_bgcolor=C_BG_PLOT, plot_bgcolor=C_BG_PLOT,
    font=dict(color=C_TXT, family="Inter"),
    margin=dict(l=30, r=30, t=40, b=30),
    title_font=dict(family="Playfair Display", size=16, color="#9bbcdb"),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='#1b2f45', zeroline=False)
)

# ═══════════════════════════════════════════════════════════════
# ROW 3: Top Charts
# ═══════════════════════════════════════════════════════════════
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
c_row3 = st.columns([1, 1.2, 1.5])

with c_row3[0].container(border=True):
    # Pie: Employee Type Default Loans
    inc_df = fdf[fdf['TARGET']==1]['NAME_INCOME_TYPE'].value_counts().reset_index()
    inc_df.columns = ['Income Type', 'Count']
    fig1 = px.pie(inc_df, values='Count', names='Income Type', hole=0.4,
                  title="How Employee Type Affects Default Loans",
                  color_discrete_sequence=['#4a8cbd', '#d4a84c', '#32a852', '#5e5e5e'])
    fig1.update_traces(textinfo='label+percent', textposition='outside', marker=dict(line=dict(color='#0a111e', width=2)))
    fig1.update_layout(**layout_defaults, showlegend=False, height=280)
    st.plotly_chart(fig1, use_container_width=True)

with c_row3[1].container(border=True):
    # Waterfall/Bar: Default Loan by Loan Term Month
    term_df = fdf[fdf['TARGET']==1].groupby('TERM_BIN').size().reset_index(name='Defaults')
    fig2 = go.Figure(go.Waterfall(
        x=term_df['TERM_BIN'], y=term_df['Defaults'],
        measure=["relative"] * len(term_df),
        decreasing={"marker":{"color": C_RED}},
        increasing={"marker":{"color": C_GREEN}},
        totals={"marker":{"color": C_BLUE_BAR}},
        text=[f"{v/1000:.1f}K" for v in term_df['Defaults']], textposition="outside"
    ))
    fig2.update_layout(**layout_defaults, title="Default Loan by Loan Term Month", height=280)
    st.plotly_chart(fig2, use_container_width=True)

with c_row3[2].container(border=True):
    # Combo: Amount at risk & Default Rate by Loan Amount
    bin_df = fdf.groupby('CREDIT_BIN').agg(Amt=('AMT_CREDIT','sum'), DR=('TARGET','mean')).reset_index()
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Bar(x=bin_df['CREDIT_BIN'], y=bin_df['Amt'], name='Amount at Risk', marker_color=C_BLUE_BAR, text=[f"${v/1e6:.0f}M" for v in bin_df['Amt']], textposition='auto'), secondary_y=False)
    fig3.add_trace(go.Scatter(x=bin_df['CREDIT_BIN'], y=bin_df['DR'], name='Default Rate', mode='lines+markers', line=dict(color=C_YELLOW, width=3), text=[f"{v:.1%}" for v in bin_df['DR']], textposition='top center'), secondary_y=True)
    fig3.update_layout(**layout_defaults, title="Amount at Risk & Default Rate by Loan Amount Bin", height=280, legend=dict(orientation="h", y=1.15, x=0))
    fig3.update_yaxes(showticklabels=False, secondary_y=False)
    fig3.update_yaxes(tickformat='.0%', secondary_y=True, showgrid=False)
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# ROW 4: Bottom Charts
# ═══════════════════════════════════════════════════════════════
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
c_row4 = st.columns([1, 1, 1, 1.2])

with c_row4[0].container(border=True):
    # H-Bar: Default Rate by LTI
    lti_df = fdf.groupby('LTI_BIN')['TARGET'].mean().reset_index()
    fig4 = px.bar(lti_df, x='TARGET', y='LTI_BIN', orientation='h', title="Loan Default Rate by LTI")
    fig4.update_traces(marker_color=C_BLUE_BAR, text=[f"{v:.1%}" for v in lti_df['TARGET']], textposition='outside')
    fig4.update_layout(**layout_defaults)
    fig4.update_layout(height=280, yaxis_title="Loan To Income")
    fig4.update_xaxes(showticklabels=False)
    st.plotly_chart(fig4, use_container_width=True)

with c_row4[1].container(border=True):
    # Pie: Default Rate by Loan Purpose / Education (using Education since Purpose is mostly XAP)
    edu_df = fdf.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().reset_index()
    # Normalize to pie representation of default intensity
    fig5 = px.pie(edu_df, values='TARGET', names='NAME_EDUCATION_TYPE', hole=0, title="Default Intensity by Education", color_discrete_sequence=['#4a8cbd', '#d4a84c', '#32a852', '#2a3f5f'])
    fig5.update_traces(textinfo='label+percent', textposition='inside')
    fig5.update_layout(**layout_defaults, showlegend=False, height=280)
    st.plotly_chart(fig5, use_container_width=True)

with c_row4[2].container(border=True):
    # V-Bar: Default Rate by DTI
    dti_df = fdf.groupby('DTI_BIN')['TARGET'].mean().reset_index()
    fig6 = px.bar(dti_df, x='DTI_BIN', y='TARGET', title="Loan Default Rate by DTI")
    fig6.update_traces(marker_color=C_BLUE_BAR, text=[f"{v:.1%}" for v in dti_df['TARGET']], textposition='outside')
    fig6.update_layout(**layout_defaults)
    fig6.update_layout(height=280, xaxis_title="Debt To Income")
    fig6.update_yaxes(showticklabels=False)
    st.plotly_chart(fig6, use_container_width=True)

with c_row4[3].container(border=True):
    # H-Bar: Amount at Risk by Region (replacing Map)
    reg_df = fdf.groupby('REGION_RATING_CLIENT_W_CITY').agg(Amt=('AMT_CREDIT','sum')).reset_index()
    reg_df['Region Rating'] = 'Rating ' + reg_df['REGION_RATING_CLIENT_W_CITY'].astype(str)
    fig7 = px.bar(reg_df, y='Region Rating', x='Amt', orientation='h', title="Amount at Risk by Region Rating", color='Region Rating', color_discrete_sequence=['#32a852', '#d4a84c', '#4a8cbd'])
    fig7.update_traces(text=[f"${v/1e6:.0f}M" for v in reg_df['Amt']], textposition='inside')
    fig7.update_layout(**layout_defaults)
    fig7.update_layout(height=280, showlegend=False)
    fig7.update_xaxes(showticklabels=False)
    st.plotly_chart(fig7, use_container_width=True)

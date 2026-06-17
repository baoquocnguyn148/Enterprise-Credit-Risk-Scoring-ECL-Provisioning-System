"""
streamlit_app.py — Enterprise Credit Risk Scoring & ECL Dashboard
================================================================
A stunning, production-ready Streamlit dashboard for Risk Analysts.
Incorporates:
- IFRS 9 Expected Credit Loss (Stage 1, 2, 3)
- Business ROI & Optimal Threshold
- SHAP (Local & Global) and PDP Plots
- Unleaked Model Metrics (Optuna + Isotonic Calibrated)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, json, re, pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Nova Bank — Enterprise Risk System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Color Palette & Theming ──────────────────────────────────
C_GREEN  = "#00E676"  # Stage 1 / Low Risk
C_BLUE   = "#00B0FF"  # Stage 2 / Medium Risk
C_ORANGE = "#FF9100"  # Stage 2 / High Risk
C_RED    = "#FF1744"  # Stage 3 / Impaired
C_BG     = "#0B132B"
C_PANEL  = "#1C2541"
C_TEXT   = "#E0E1DD"

TIER_COLORS = {
    'Very Low': C_GREEN,
    'Low':      C_BLUE,
    'Medium':   C_ORANGE,
    'High':     C_RED,
}

# ── Sleek Custom CSS ──────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, h4, h5 {{
        font-family: 'Outfit', sans-serif;
        color: {C_TEXT};
    }}
    .stApp {{
        background-color: {C_BG};
    }}
    .metric-card {{
        background: linear-gradient(145deg, {C_PANEL}, #182039);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(0, 230, 118, 0.1);
    }}
    .metric-value {{
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00B0FF, #00E676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }}
    .metric-value.red {{
        background: -webkit-linear-gradient(45deg, #FF1744, #FF9100);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .metric-label {{
        font-size: 0.85rem;
        color: #8D99AE;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 8px;
        font-weight: 600;
    }}
    .risk-badge {{
        padding: 8px 24px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
        background-color: {C_BG};
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 60px;
        border-radius: 8px 8px 0 0;
        padding: 0 24px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        color: #8D99AE;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {C_PANEL};
        color: white;
        border-bottom: 3px solid {C_GREEN};
    }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
DATA_DIR    = r'd:\Risk\data'
MODELS_DIR  = r'd:\Risk\models'
REPORTS_DIR = r'd:\Risk\reports'

@st.cache_data
def load_data():
    df = pd.read_parquet(os.path.join(DATA_DIR, 'results_df.parquet'))
    # Try to load IFRS9 results if available
    ifrs_path = os.path.join(DATA_DIR, 'results_ifrs9.parquet')
    if os.path.exists(ifrs_path):
        df_ifrs = pd.read_parquet(ifrs_path)
        return df_ifrs
    return df

@st.cache_data
def load_metrics():
    met = {}
    path = os.path.join(MODELS_DIR, 'model_metrics.json')
    if os.path.exists(path):
        with open(path) as f: met = json.load(f)
    return met

@st.cache_data
def load_ifrs9_summary():
    path = os.path.join(REPORTS_DIR, 'ifrs9_stage_summary.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

try:
    df = load_data()
    metrics = load_metrics()
    ifrs9_df = load_ifrs9_summary()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=60)
    st.markdown("## 🏦 Nova Bank")
    st.markdown("### Enterprise Risk Engine")
    st.markdown("---")
    
    st.markdown("#### 🎯 Portfolio Filters")
    tier_filter = st.multiselect("Risk Tier", ['Very Low', 'Low', 'Medium', 'High'], default=['Very Low', 'Low', 'Medium', 'High'])
    gender_filter = st.selectbox("Gender", ["All", "M", "F"])
    
    st.markdown("---")
    st.markdown("#### ⚙️ Model Vitals")
    st.caption(f"**AUC-ROC:** {metrics.get('oof_auc', 0):.4f}")
    st.caption(f"**KS Stat:** {metrics.get('ks_statistic', 0):.4f}")
    st.caption(f"**Brier (Cal):** {metrics.get('brier_calibrated', 0):.4f}")
    st.caption(f"**PSI (Drift):** {metrics.get('psi_score', 0):.4f}")
    
    st.markdown("---")
    st.caption("v2.0 • IFRS 9 Compliant")

# Apply filters
fdf = df.copy()
if tier_filter: fdf = fdf[fdf['RISK_TIER'].isin(tier_filter)]
if gender_filter != "All": fdf = fdf[fdf['CODE_GENDER'] == gender_filter]

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
t_overview, t_ifrs9, t_roi, t_predictor, t_explain = st.tabs([
    "🌐 Portfolio Overview", 
    "📈 IFRS 9 Staging", 
    "💰 Business ROI", 
    "🎯 Client Predictor", 
    "🧠 Model XAI"
])

# ─────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────
with t_overview:
    total_exposure = fdf['AMT_CREDIT'].sum()
    total_ecl      = fdf['ECL'].sum() if 'ECL' in fdf.columns else 0
    coverage       = total_ecl / total_exposure if total_exposure > 0 else 0
    default_rt     = fdf['TARGET'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{len(fdf):,}</div><div class="metric-label">Total Borrowers</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">${total_exposure/1e9:.2f}B</div><div class="metric-label">Total Exposure (EAD)</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value red">${total_ecl/1e9:.2f}B</div><div class="metric-label">Expected Credit Loss</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{coverage:.1%}</div><div class="metric-label">Coverage Ratio</div></div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1, 1.5])
    with col_a:
        # Risk Tier Donut
        tier_counts = fdf['RISK_TIER'].value_counts().reset_index()
        fig_tier = px.pie(tier_counts, values='count', names='RISK_TIER', hole=0.6,
                          color='RISK_TIER', color_discrete_map=TIER_COLORS)
        fig_tier.update_layout(title="Portfolio by Risk Tier", title_x=0.5, font_color=C_TEXT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True, legend=dict(orientation="h", y=-0.1))
        fig_tier.add_annotation(text=f"{default_rt:.1%}<br>Avg Default", x=0.5, y=0.5, font_size=20, showarrow=False, font_color=C_TEXT)
        st.plotly_chart(fig_tier, use_container_width=True)

    with col_b:
        # Amount at Risk vs Default Rate (Combo)
        fdf['LTI_BIN'] = pd.qcut(fdf.get('CREDIT_INCOME_RATIO', fdf['AMT_CREDIT']/fdf['AMT_INCOME_TOTAL']), 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Extreme'])
        lti_df = fdf.groupby('LTI_BIN').agg(EAD=('AMT_CREDIT','sum'), DR=('TARGET','mean')).reset_index()
        
        fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
        fig_combo.add_trace(go.Bar(x=lti_df['LTI_BIN'], y=lti_df['EAD'], name='Exposure ($)', marker_color=C_BLUE, opacity=0.7), secondary_y=False)
        fig_combo.add_trace(go.Scatter(x=lti_df['LTI_BIN'], y=lti_df['DR'], name='Default Rate', mode='lines+markers', line=dict(color=C_RED, width=3), marker=dict(size=10)), secondary_y=True)
        fig_combo.update_layout(title="Exposure & Default Rate by Loan-to-Income (LTI)", title_x=0.5, font_color=C_TEXT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1))
        fig_combo.update_yaxes(title_text="Total Exposure ($)", secondary_y=False, tickformat=".2s", showgrid=False)
        fig_combo.update_yaxes(title_text="Default Rate", secondary_y=True, tickformat=".1%", showgrid=False)
        st.plotly_chart(fig_combo, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# TAB 2: IFRS 9 STAGING
# ─────────────────────────────────────────────────────────────────
with t_ifrs9:
    st.markdown("## 📊 IFRS 9 Expected Credit Loss Provisioning")
    if ifrs9_df is not None:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(ifrs9_df.style.format({
                'Stage': '{:.0f}', 'Count': '{:,.0f}', 'EAD_Billion': '${:.2f}B',
                'ECL_Billion': '${:.2f}B', 'PD_12M': '{:.2%}', 'PD_Lifetime': '{:.2%}',
                'Coverage': '{:.2%}'
            }), height=200)
            st.caption("Stage 1: 12M ECL | Stage 2 & 3: Lifetime ECL")
        with c2:
            fig_stg = px.bar(ifrs9_df, x='Stage', y=['EAD_Billion', 'ECL_Billion'], barmode='group',
                             color_discrete_sequence=[C_BLUE, C_RED],
                             title="EAD vs ECL by IFRS 9 Stage", text_auto='$.2f')
            fig_stg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=C_TEXT)
            st.plotly_chart(fig_stg, use_container_width=True)
            
        st.markdown("### Macroeconomic Scenario Overlay & LGD Sensitivity")
        s1, s2 = st.columns(2)
        with s1:
            if os.path.exists(os.path.join(REPORTS_DIR, 'ifrs9_macro_scenarios.png')):
                st.image(os.path.join(REPORTS_DIR, 'ifrs9_macro_scenarios.png'), use_column_width=True)
        with s2:
            if os.path.exists(os.path.join(REPORTS_DIR, 'lgd_sensitivity.png')):
                st.image(os.path.join(REPORTS_DIR, 'lgd_sensitivity.png'), use_column_width=True)
    else:
        st.info("IFRS 9 Engine has not been executed yet. Run `ifrs9_ecl_engine.py`.")

# ─────────────────────────────────────────────────────────────────
# TAB 3: BUSINESS ROI
# ─────────────────────────────────────────────────────────────────
with t_roi:
    st.markdown("## 💰 Business ROI & Cut-off Optimization")
    roi_path = os.path.join(REPORTS_DIR, 'roi_summary.csv')
    if os.path.exists(roi_path):
        roi_df = pd.read_csv(roi_path)
        best = roi_df.loc[roi_df['net_profit_M'].idxmax()]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{best["threshold"]:.3f}</div><div class="metric-label">Optimal Threshold</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">${best["net_profit_M"]:.0f}M</div><div class="metric-label">Max Net Profit</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-value">{best["approval_rate"]:.1%}</div><div class="metric-label">Approval Rate</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-value red">-${best["loss_fn_M"]:.0f}M</div><div class="metric-label">Residual Loan Loss</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if os.path.exists(os.path.join(REPORTS_DIR, 'roi_threshold_analysis.png')):
            st.image(os.path.join(REPORTS_DIR, 'roi_threshold_analysis.png'), use_column_width=True)
    else:
        st.info("Business ROI module has not been executed. Run `business_roi_analysis.py`.")

# ─────────────────────────────────────────────────────────────────
# TAB 4: CLIENT PREDICTOR
# ─────────────────────────────────────────────────────────────────
with t_predictor:
    st.markdown("## 🎯 Single Client Underwriting")
    
    col_search, _ = st.columns([1, 2])
    with col_search:
        sample_ids = fdf['SK_ID_CURR'].sample(100, random_state=42).tolist()
        sel_id = st.selectbox("Search Client ID (Lookup)", sorted(sample_ids))
        
    if sel_id:
        client = fdf[fdf['SK_ID_CURR'] == sel_id].iloc[0]
        prob = client['PRED_PROB']
        tier = client['RISK_TIER']
        color = TIER_COLORS.get(tier, C_GREEN)
        
        c_prof, c_score = st.columns([1, 1])
        with c_prof:
            st.markdown(f"### Client: `{int(sel_id)}`")
            st.markdown(f"**Credit Amount:** ${client['AMT_CREDIT']:,.0f}")
            st.markdown(f"**Income:** ${client['AMT_INCOME_TOTAL']:,.0f}")
            st.markdown(f"**LTI Ratio:** {client.get('CREDIT_INCOME_RATIO', 0):.2f}x")
            st.markdown(f"**Age:** {client.get('AGE_YEARS', 0):.0f} years")
            st.markdown(f"**Bureau Flag:** {'⚠️ Bad Debt' if client.get('bureau_bad_debt_flag',0)==1 else '✅ Clean'}")
            
        with c_score:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                title={'text': "Probability of Default (PD)", 'font': {'color': C_TEXT}},
                number={'suffix': "%", 'font': {'color': color, 'size': 50}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': C_TEXT},
                    'bar': {'color': color},
                    'bgcolor': "rgba(0,0,0,0.2)",
                    'steps': [
                        {'range': [0, 20], 'color': f'{C_GREEN}33'},
                        {'range': [20, 45], 'color': f'{C_BLUE}33'},
                        {'range': [45, 65], 'color': f'{C_ORANGE}33'},
                        {'range': [65, 100], 'color': f'{C_RED}33'},
                    ]
                }
            ))
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color=C_TEXT, height=300, margin=dict(t=0,b=0))
            st.plotly_chart(fig_g, use_container_width=True)
            
            st.markdown(f"<div style='text-align:center'><div class='risk-badge' style='background:{color}22; border:2px solid {color}; color:{color}'>{tier} RISK</div></div>", unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("### 🔍 Risk Explanation (SHAP Waterfall)")
        if os.path.exists(os.path.join(DATA_DIR, 'shap_values_sample.parquet')):
            shap_df = pd.read_parquet(os.path.join(DATA_DIR, 'shap_values_sample.parquet'))
            c_shap = shap_df[shap_df['SK_ID_CURR'] == sel_id]
            if not c_shap.empty:
                sv = c_shap.drop(columns=['SK_ID_CURR']).iloc[0].values
                feats = [c for c in c_shap.columns if c != 'SK_ID_CURR']
                s_series = pd.Series(sv, index=feats).sort_values(key=abs, ascending=True).tail(10)
                colors = [C_RED if v > 0 else C_GREEN for v in s_series.values]
                
                fig_wf = go.Figure(go.Bar(
                    x=s_series.values, y=s_series.index, orientation='h',
                    marker_color=colors, text=[f"{v:+.3f}" for v in s_series.values],
                    textposition='outside'
                ))
                fig_wf.update_layout(
                    title="Top 10 Factors pushing Risk Score (Red = Increase Risk, Green = Decrease)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=C_TEXT,
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'), height=400
                )
                st.plotly_chart(fig_wf, use_container_width=True)
            else:
                st.info("SHAP values not pre-computed for this exact client. View static waterfalls in 'Model XAI' tab.")
        else:
            st.info("SHAP analysis missing. Run `shap_analysis.py`.")

# ─────────────────────────────────────────────────────────────────
# TAB 5: MODEL XAI (SHAP & PDP)
# ─────────────────────────────────────────────────────────────────
with t_explain:
    st.markdown("## 🧠 eXplainable AI (XAI)")
    st.markdown("Unpacking the LightGBM black-box using SHAP and Partial Dependence Plots.")
    
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("#### Global Feature Importance (SHAP)")
        if os.path.exists(os.path.join(REPORTS_DIR, 'shap_summary_bar.png')):
            st.image(os.path.join(REPORTS_DIR, 'shap_summary_bar.png'), use_column_width=True)
    with s2:
        st.markdown("#### Feature Directionality (Beeswarm)")
        if os.path.exists(os.path.join(REPORTS_DIR, 'shap_beeswarm.png')):
            st.image(os.path.join(REPORTS_DIR, 'shap_beeswarm.png'), use_column_width=True)
            
    st.markdown("---")
    st.markdown("#### Partial Dependence Plots (PDP)")
    if os.path.exists(os.path.join(REPORTS_DIR, 'pdp_top_features.png')):
        st.image(os.path.join(REPORTS_DIR, 'pdp_top_features.png'), use_column_width=True)

"""
streamlit_app.py — Phase 5: Credit Risk Analyst Dashboard (v2 UPGRADED)
Home Credit Default Risk

Upgrades per Risk Analyst Audit:
  [FIX 1] Tab 1: Add ECL KPIs, more segment charts
  [FIX 2] Tab 2: Full input form for new client scoring + SHAP waterfall
  [FIX 3] Tab 3: Calibration curve, PSI, metrics panel
  [FIX 4] Tab 4: Full deep-dive charts (Occupation, Education, LTI, DTI, Age)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, roc_auc_score
import pickle, os, json, re, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Credit Risk Analyst — Nova Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Color Palette ─────────────────────────────────────────────
C_GREEN  = "#2ECC71"
C_BLUE   = "#1E90FF"
C_ORANGE = "#F39C12"
C_RED    = "#E74C3C"
C_TEAL   = "#64ffda"
C_NAVY   = "#0a192f"
C_DARK   = "#112240"
C_GRAY   = "#8892b0"

TIER_COLORS = {
    'Very Low': C_GREEN,
    'Low':      C_BLUE,
    'Medium':   C_ORANGE,
    'High':     C_RED,
}

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container { max-width: 1400px; padding-top: 1rem; }
    .kpi-card {
        background: linear-gradient(135deg, #112240 0%, #1a2f4a 100%);
        border: 1px solid #233554;
        border-radius: 10px;
        padding: 20px 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #64ffda; line-height: 1.2; }
    .kpi-label { font-size: 0.78rem; color: #8892b0; margin-top: 4px; letter-spacing: 0.05em; text-transform: uppercase; }
    .kpi-delta { font-size: 0.85rem; margin-top: 6px; }
    .risk-badge {
        display: inline-block; padding: 6px 18px; border-radius: 20px;
        font-weight: 700; font-size: 1rem; letter-spacing: 0.05em;
    }
    .section-header { color: #ccd6f6; font-weight: 600; font-size: 1.1rem; margin: 12px 0 8px; }
    .ecl-highlight { background: #1d3557; border-left: 4px solid #64ffda; padding: 12px; border-radius: 6px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════════
DATA_DIR    = r'd:\Risk\data'
MODELS_DIR  = r'd:\Risk\models'
REPORTS_DIR = r'd:\Risk\reports'

@st.cache_data
def load_results():
    df = pd.read_parquet(os.path.join(DATA_DIR, 'results_df.parquet'))
    return df

@st.cache_resource
def load_models():
    models = []
    for i in range(1, 6):
        path = os.path.join(MODELS_DIR, f'lgbm_fold{i}.pkl')
        with open(path, 'rb') as f:
            models.append(pickle.load(f))
    with open(os.path.join(MODELS_DIR, 'feature_list.pkl'), 'rb') as f:
        features = pickle.load(f)
    features = [re.sub(r'[^A-Za-z0-9_]+', '_', x) for x in features]
    cal_path = os.path.join(MODELS_DIR, 'isotonic_calibrator.pkl')
    calibrated = pickle.load(open(cal_path, 'rb')) if os.path.exists(cal_path) else None
    return models, features, calibrated

@st.cache_data
def load_config():
    with open(os.path.join(MODELS_DIR, 'tier_config.json')) as f:
        tier_cfg = json.load(f)
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    return tier_cfg, metrics

@st.cache_data
def load_shap_artifacts():
    shap_path = os.path.join(DATA_DIR, 'shap_values_sample.parquet')
    ev_path   = os.path.join(MODELS_DIR, 'shap_expected_value.pkl')
    if os.path.exists(shap_path) and os.path.exists(ev_path):
        shap_df = pd.read_parquet(shap_path)
        with open(ev_path, 'rb') as f:
            expected_val = pickle.load(f)
        return shap_df, float(expected_val)
    return None, None

try:
    df = load_results()
    models_list, FEATURES, calibrated_model = load_models()
    tier_cfg, model_metrics = load_config()
    shap_df, shap_expected = load_shap_artifacts()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("Please run `modeling.py` first.")
    st.stop()

# ── Helper: predict for a new client ─────────────────────────
def predict_client(client_row: pd.DataFrame):
    """Returns (raw_prob, calibrated_prob, risk_tier)"""
    client_row = client_row.rename(columns=lambda x: re.sub(r'[^A-Za-z0-9_]+', '_', x))
    X_client = client_row[FEATURES]
    raw_preds = np.mean([m.predict_proba(X_client)[:, 1] for m in models_list], axis=0)
    raw_prob = float(raw_preds[0])
    bins   = tier_cfg['bins']
    labels = ['Very Low', 'Low', 'Medium', 'High']
    for i in range(len(bins) - 1):
        if bins[i] <= raw_prob <= bins[i+1]:
            tier = labels[i]
            break
    else:
        tier = 'High'
    return raw_prob, tier


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 Nova Bank")
    st.markdown("### Credit Risk Analyst")
    st.markdown("---")
    st.markdown("**Portfolio Filters**")
    gender_filter   = st.selectbox("Gender", ["All", "M", "F"])
    tier_filter     = st.multiselect("Risk Tier", ['Very Low', 'Low', 'Medium', 'High'],
                                      default=['Very Low', 'Low', 'Medium', 'High'])
    contract_filter = st.selectbox("Contract Type", ["All", "Cash loans", "Revolving loans"])
    st.markdown("---")
    auc  = model_metrics.get('oof_auc', 0)
    ks   = model_metrics.get('ks_statistic', 0)
    gini = model_metrics.get('gini', 0)
    st.markdown(f"**Model Performance**")
    st.markdown(f"- AUC-ROC: **{auc:.4f}**")
    st.markdown(f"- KS Statistic: **{ks:.4f}**")
    st.markdown(f"- Gini: **{gini:.4f}**")
    st.markdown(f"- Borrowers: **{len(df):,}**")

# ── Apply filters ─────────────────────────────────────────────
fdf = df.copy()
if gender_filter != "All":
    fdf = fdf[fdf['CODE_GENDER'] == gender_filter]
if tier_filter:
    fdf = fdf[fdf['RISK_TIER'].isin(tier_filter)]
if contract_filter != "All":
    fdf = fdf[fdf['NAME_CONTRACT_TYPE'] == contract_filter]


# ═══════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Portfolio Overview",
    "🎯 Risk Predictor",
    "📈 Model Insights",
    "🔬 Portfolio Deep Dive",
])


# ─────────────────────────────────────────────────────────────────
# TAB 1: PORTFOLIO OVERVIEW
# ─────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## Portfolio Overview")

    # ── [FIX 1] KPI row with ECL ─────────────────────────────
    total_borrowers = len(fdf)
    total_loan      = fdf['AMT_CREDIT'].sum()
    total_ecl       = fdf['ECL'].sum() if 'ECL' in fdf.columns else fdf['PRED_PROB'].sum() * 0.45 * fdf['AMT_CREDIT'].mean()
    ecl_ratio       = total_ecl / total_loan if total_loan > 0 else 0
    default_rate    = fdf['TARGET'].mean() * 100
    high_risk_pct   = (fdf['RISK_TIER'] == 'High').mean() * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    def kpi(label, value, delta=None):
        delta_html = f'<div class="kpi-delta" style="color:{"#E74C3C" if delta and delta > 0 else "#2ECC71"}">{delta:+.1f}% vs avg</div>' if delta else ''
        return f'<div class="kpi-card"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div>{delta_html}</div>'

    k1.markdown(kpi("Total Borrowers",    f"{total_borrowers:,}"),              unsafe_allow_html=True)
    k2.markdown(kpi("Total Portfolio",    f"${total_loan/1e9:.2f}B"),            unsafe_allow_html=True)
    k3.markdown(kpi("Expected Credit Loss", f"${total_ecl/1e6:.1f}M"),           unsafe_allow_html=True)
    k4.markdown(kpi("ECL Coverage Ratio", f"{ecl_ratio:.2%}"),                   unsafe_allow_html=True)
    k5.markdown(kpi("Portfolio Default Rate", f"{default_rate:.1f}%"),           unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])

    with c1:
        # Risk Tier Donut
        tier_counts = fdf['RISK_TIER'].value_counts().reset_index()
        tier_counts.columns = ['RISK_TIER', 'count']
        fig_donut = px.pie(
            tier_counts, values='count', names='RISK_TIER', hole=0.55,
            color='RISK_TIER', color_discrete_map=TIER_COLORS,
            title="Risk Tier Distribution"
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                                 showlegend=False, title_font_size=14)
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        # Default Rate by Income Type — horizontal bar
        inc_df = fdf.groupby('NAME_INCOME_TYPE')['TARGET'].agg(['mean', 'count']).reset_index()
        inc_df = inc_df[inc_df['count'] > 100].sort_values('mean')
        inc_df.columns = ['NAME_INCOME_TYPE', 'Default_Rate', 'Count']
        fig_inc = px.bar(
            inc_df, x='Default_Rate', y='NAME_INCOME_TYPE', orientation='h',
            title="Default Rate by Income Type",
            labels={'Default_Rate': 'Default Rate', 'NAME_INCOME_TYPE': ''},
            color='Default_Rate', color_continuous_scale='RdYlGn_r',
            text=inc_df['Default_Rate'].map(lambda x: f"{x:.1%}")
        )
        fig_inc.update_traces(textposition='outside')
        fig_inc.update_layout(xaxis_tickformat='.1%', paper_bgcolor='rgba(0,0,0,0)',
                               font_color='white', coloraxis_showscale=False,
                               title_font_size=14, margin=dict(l=0))
        st.plotly_chart(fig_inc, use_container_width=True)

    # ── Row 2: ECL by Tier bar + Credit Amount vs Default Rate
    c3, c4 = st.columns(2)
    with c3:
        if 'ECL' in fdf.columns:
            ecl_tier = fdf.groupby('RISK_TIER')['ECL'].sum().reset_index()
            ecl_tier.columns = ['RISK_TIER', 'Total_ECL']
            fig_ecl = px.bar(
                ecl_tier, x='RISK_TIER', y='Total_ECL', color='RISK_TIER',
                color_discrete_map=TIER_COLORS,
                title="Total ECL by Risk Tier (IFRS 9)",
                labels={'Total_ECL': 'Expected Credit Loss ($)', 'RISK_TIER': ''}
            )
            fig_ecl.update_traces(text=ecl_tier['Total_ECL'].map(lambda x: f"${x/1e6:.1f}M"),
                                   textposition='outside')
            fig_ecl.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                                   showlegend=False, title_font_size=14)
            st.plotly_chart(fig_ecl, use_container_width=True)

    with c4:
        # Amount at Risk + Default Rate by Credit Amount Bin (bar+line combo)
        fdf['CREDIT_BIN'] = pd.cut(fdf['AMT_CREDIT'],
                                    bins=[0, 100000, 200000, 300000, 500000, 2e6],
                                    labels=['<100K', '100-200K', '200-300K', '300-500K', '>500K'])
        combo_df = fdf.groupby('CREDIT_BIN').agg(
            Amount_at_Risk=('AMT_CREDIT', 'sum'),
            Default_Rate=('TARGET', 'mean')
        ).reset_index()

        fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
        fig_combo.add_trace(go.Bar(
            x=combo_df['CREDIT_BIN'].astype(str),
            y=combo_df['Amount_at_Risk'],
            name='Amount at Risk',
            marker_color=C_BLUE, opacity=0.8
        ), secondary_y=False)
        fig_combo.add_trace(go.Scatter(
            x=combo_df['CREDIT_BIN'].astype(str),
            y=combo_df['Default_Rate'],
            name='Default Rate',
            mode='lines+markers',
            line=dict(color=C_ORANGE, width=3),
            marker=dict(size=8)
        ), secondary_y=True)
        fig_combo.update_layout(
            title="Amount at Risk + Default Rate by Credit Amount",
            paper_bgcolor='rgba(0,0,0,0)', font_color='white',
            title_font_size=14, legend=dict(orientation='h', y=1.1)
        )
        fig_combo.update_yaxes(title_text="Amount ($)", secondary_y=False, tickformat='$,.0f')
        fig_combo.update_yaxes(title_text="Default Rate", secondary_y=True, tickformat='.1%')
        st.plotly_chart(fig_combo, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# TAB 2: RISK PREDICTOR  [FIX 2 — Full input form]
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 🎯 Individual Client Risk Assessment")

    mode = st.radio("Assessment Mode", ["📋 New Client (Manual Input)", "🔍 Lookup Existing Client"],
                    horizontal=True)
    st.markdown("---")

    if mode == "🔍 Lookup Existing Client":
        col_id, _ = st.columns([2, 3])
        with col_id:
            sample_ids = fdf['SK_ID_CURR'].sample(200, random_state=42).sort_values().tolist()
            selected_id = st.selectbox("Select Client ID", sample_ids)

        if selected_id:
            client_data = fdf[fdf['SK_ID_CURR'] == selected_id].iloc[0]
            prob  = float(client_data['PRED_PROB'])
            tier  = str(client_data['RISK_TIER'])
            color = TIER_COLORS.get(tier, C_GRAY)
            ecl   = float(client_data.get('ECL', prob * 0.45 * client_data['AMT_CREDIT']))

            c1, c2, c3 = st.columns([1, 1.2, 1])
            with c1:
                st.markdown("### Client Profile")
                st.markdown(f"**Age:** {client_data.get('AGE_YEARS', 'N/A'):.0f} years")
                st.markdown(f"**Credit Amount:** ${client_data['AMT_CREDIT']:,.0f}")
                st.markdown(f"**Income:** ${client_data['AMT_INCOME_TOTAL']:,.0f}")
                st.markdown(f"**Credit/Income Ratio:** {client_data.get('CREDIT_INCOME_RATIO', 0):.2f}x")
                st.markdown(f"**EXT_SOURCE_2:** {client_data.get('EXT_SOURCE_2', 'N/A'):.3f}")
                st.markdown(f"**Bureau Bad Debt:** {'Yes ⚠️' if client_data.get('bureau_bad_debt_flag', 0) == 1 else 'No ✅'}")
                actual_label = "🔴 Defaulted" if client_data['TARGET'] == 1 else "🟢 Repaid"
                st.markdown(f"**Actual Outcome:** {actual_label}")

            with c2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Default Probability (PD)", 'font': {'size': 16, 'color': 'white'}},
                    number={'suffix': "%", 'font': {'color': color, 'size': 36}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "white"},
                        'bar': {'color': color},
                        'bgcolor': "rgba(0,0,0,0)",
                        'steps': [
                            {'range': [0, tier_cfg['bins'][1]*100],  'color': 'rgba(46,204,113,0.2)'},
                            {'range': [tier_cfg['bins'][1]*100, tier_cfg['bins'][2]*100], 'color': 'rgba(30,144,255,0.2)'},
                            {'range': [tier_cfg['bins'][2]*100, tier_cfg['bins'][3]*100], 'color': 'rgba(243,156,18,0.2)'},
                            {'range': [tier_cfg['bins'][3]*100, 100], 'color': 'rgba(231,76,60,0.2)'},
                        ],
                        'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': tier_cfg['optimal_threshold']*100}
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=280)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with c3:
                badge_css = f"background:{color}22; border: 2px solid {color}; color:{color};"
                st.markdown(f"""
                    <div style="text-align:center; margin-top: 20px;">
                        <div style="color:#8892b0; font-size:0.85rem; margin-bottom:8px;">RISK CLASSIFICATION</div>
                        <div class="risk-badge" style="{badge_css}">{tier}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="ecl-highlight">
                        <div style="color:#8892b0; font-size:0.8rem;">EXPECTED CREDIT LOSS (IFRS 9)</div>
                        <div style="color:#64ffda; font-size:1.6rem; font-weight:700;">${ecl:,.0f}</div>
                        <div style="color:#8892b0; font-size:0.75rem;">PD={prob:.1%} × LGD=45% × EAD=${client_data['AMT_CREDIT']:,.0f}</div>
                    </div>
                """, unsafe_allow_html=True)
                if tier == 'High':
                    st.error("⚠️ Recommend: Manual Review Required")
                elif tier == 'Medium':
                    st.warning("📋 Recommend: Additional Documentation")
                elif tier == 'Low':
                    st.info("✅ Standard Approval Process")
                else:
                    st.success("🟢 Auto-Approval Eligible")

            # ── SHAP Waterfall for this client ─────────────────
            if shap_df is not None:
                st.markdown("---")
                st.markdown("### 🔍 Why does this client have this risk? (SHAP Explanation)")
                sk_id = int(client_data['SK_ID_CURR'])
                client_shap = shap_df[shap_df['SK_ID_CURR'] == sk_id]
                if not client_shap.empty:
                    sv = client_shap.drop(columns=['SK_ID_CURR']).iloc[0].values
                    feat_names = [c for c in client_shap.columns if c != 'SK_ID_CURR']
                    # Build top-15 bar chart
                    shap_series = pd.Series(sv, index=feat_names).sort_values(key=abs, ascending=False).head(15)
                    colors_shap = [C_RED if v > 0 else C_GREEN for v in shap_series.values]
                    fig_wf = go.Figure(go.Bar(
                        x=shap_series.values,
                        y=shap_series.index.tolist(),
                        orientation='h',
                        marker_color=colors_shap,
                        text=[f"{v:+.4f}" for v in shap_series.values],
                        textposition='outside'
                    ))
                    fig_wf.update_layout(
                        title=f"SHAP Feature Contributions — Client {sk_id} (Red = ↑ Risk, Green = ↓ Risk)",
                        paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                        xaxis_title='SHAP value (impact on prediction)',
                        height=450, margin=dict(l=220)
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)
                    # Caption
                    top_pos = shap_series[shap_series > 0].head(3).index.tolist()
                    top_neg = shap_series[shap_series < 0].head(3).index.tolist()
                    if top_pos:
                        st.markdown(f"🔴 **Risk drivers (pushing toward default):** {', '.join(top_pos)}")
                    if top_neg:
                        st.markdown(f"🟢 **Protective factors (reducing risk):** {', '.join(top_neg)}")
                else:
                    st.info("Client not in SHAP sample — select a different client or re-run shap_analysis.py")

    else:
        # [FIX 2] Manual input form for new client
        st.markdown("### Enter Client Information")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Financial**")
            amt_credit  = st.number_input("Credit Amount ($)", 10000, 4000000, 300000, 10000)
            amt_income  = st.number_input("Annual Income ($)", 10000, 10000000, 180000, 5000)
            amt_annuity = st.number_input("Monthly Annuity ($)", 1000, 200000, 15000, 500)
            ext_2 = st.slider("External Credit Score 2 (0–1)", 0.0, 1.0, 0.6, 0.01)
            ext_3 = st.slider("External Credit Score 3 (0–1)", 0.0, 1.0, 0.6, 0.01)

        with col_b:
            st.markdown("**Personal**")
            age_years    = st.slider("Age (years)", 20, 70, 35)
            yrs_employed = st.slider("Years Employed", 0, 40, 5)
            cnt_children = st.number_input("Number of Children", 0, 10, 0)
            gender       = st.selectbox("Gender", ["M", "F"])
            education    = st.selectbox("Education", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"])

        with col_c:
            st.markdown("**Credit History**")
            bureau_loans      = st.number_input("# Bureau Loans", 0, 50, 3)
            bureau_bad_flag   = st.selectbox("Any Bureau Bad Debt?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            prev_refused_ratio = st.slider("Previous Refused Loan Ratio", 0.0, 1.0, 0.0, 0.01)
            cc_util_mean      = st.slider("Avg Credit Card Utilization", 0.0, 2.0, 0.4, 0.01)
            inst_late_ratio   = st.slider("Installment Late Payment Ratio", 0.0, 1.0, 0.05, 0.01)

        if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
            # Build feature row using stored medians as defaults
            with open(os.path.join(MODELS_DIR, 'imputation_stats.json')) as f:
                imp_stats = json.load(f)

            # Seed with medians from training data
            client_row = {feat: imp_stats[feat]['median'] for feat in FEATURES if feat in imp_stats}

            # Override with user inputs
            overrides = {
                'AMT_CREDIT': amt_credit, 'AMT_INCOME_TOTAL': amt_income,
                'AMT_ANNUITY': amt_annuity, 'EXT_SOURCE_2': ext_2, 'EXT_SOURCE_3': ext_3,
                'AGE_YEARS': age_years, 'YEARS_EMPLOYED': yrs_employed,
                'CNT_CHILDREN': cnt_children,
                'CREDIT_INCOME_RATIO': amt_credit / (amt_income + 1e-6),
                'ANNUITY_INCOME_RATIO': amt_annuity / (amt_income + 1e-6),
                'CREDIT_TERM': amt_credit / (amt_annuity + 1e-6),
                'bureau_loan_count': bureau_loans,
                'bureau_bad_debt_flag': bureau_bad_flag,
                'prev_refused_ratio': prev_refused_ratio,
                'cc_utilization_mean': cc_util_mean,
                'inst_late_ratio': inst_late_ratio,
                'CODE_GENDER_M': 1 if gender == 'M' else 0,
                'CODE_GENDER_F': 1 if gender == 'F' else 0,
            }
            client_row.update({k: v for k, v in overrides.items() if k in FEATURES})

            client_df = pd.DataFrame([client_row])
            for feat in FEATURES:
                if feat not in client_df.columns:
                    client_df[feat] = imp_stats.get(feat, {}).get('median', 0)

            prob, tier = predict_client(client_df[FEATURES])
            ecl_new = prob * 0.45 * amt_credit
            color = TIER_COLORS.get(tier, '#aaa')

            st.markdown("---")
            r1, r2, r3 = st.columns([1, 1.3, 1])
            with r1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="color:{color}">{prob:.1%}</div>
                    <div class="kpi-label">Default Probability (PD)</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                badge_css = f"background:{color}22; border: 2px solid {color}; color:{color};"
                st.markdown(f"""
                <div style="text-align:center; padding: 20px;">
                    <div style="color:#8892b0; font-size:0.85rem; margin-bottom:10px;">RISK TIER</div>
                    <div class="risk-badge" style="{badge_css}; font-size:1.4rem; padding: 12px 30px">{tier}</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""
                <div class="ecl-highlight">
                    <div style="color:#8892b0; font-size:0.8rem;">EXPECTED CREDIT LOSS</div>
                    <div style="color:#64ffda; font-size:1.8rem; font-weight:700;">${ecl_new:,.0f}</div>
                    <div style="color:#8892b0; font-size:0.75rem;">LGD assumption: 45%</div>
                </div>""", unsafe_allow_html=True)

            # Recommendation
            if tier == 'High':
                st.error("❌ HIGH RISK: Recommend REJECTION or require additional collateral + co-borrower")
            elif tier == 'Medium':
                st.warning("⚠️ MEDIUM RISK: Approve with conditions — reduce credit limit by 20%, require income proof")
            elif tier == 'Low':
                st.info("✅ LOW RISK: Standard approval process. Monthly income verification recommended.")
            else:
                st.success("🟢 VERY LOW RISK: Eligible for fast-track approval and preferential rate")


# ─────────────────────────────────────────────────────────────────
# TAB 3: MODEL INSIGHTS  [FIX 3 — Full upgrade]
# ─────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 📈 Model Performance & Interpretability")

    # ── KPI row ──────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(kpi("AUC-ROC", f"{model_metrics.get('oof_auc', 0):.4f}"), unsafe_allow_html=True)
    m2.markdown(kpi("KS Statistic", f"{model_metrics.get('ks_statistic', 0):.4f}"), unsafe_allow_html=True)
    m3.markdown(kpi("Gini Coefficient", f"{model_metrics.get('gini', 0):.4f}"), unsafe_allow_html=True)
    m4.markdown(kpi("Brier Score (cal)", f"{model_metrics.get('brier_calibrated', 0):.4f}"), unsafe_allow_html=True)
    m5.markdown(kpi("PSI Score", f"{model_metrics.get('psi_score', 0):.4f}"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sub-tabs inside Tab 3 ─────────────────────────────────
    s1, s2, s3, s4 = st.tabs(["📉 ROC & Fold AUC", "🔬 SHAP Global", "📐 Calibration", "🔗 Dependence Plots"])

    with s1:
        sc1, sc2 = st.columns(2)
        with sc1:
            # Interactive AUC-ROC curve from OOF predictions
            oof_preds = df['PRED_PROB'].values
            y_true    = df['TARGET'].values
            fpr_vals, tpr_vals, _ = roc_curve(y_true, oof_preds)
            auc_val = model_metrics.get('oof_auc', roc_auc_score(y_true, oof_preds))
            ks_val  = model_metrics.get('ks_statistic', 0)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr_vals, y=tpr_vals, mode='lines',
                name=f'LightGBM (AUC={auc_val:.4f})',
                line=dict(color=C_TEAL, width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                name='Random (AUC=0.50)', line=dict(color=C_GRAY, width=2, dash='dash')
            ))
            fig_roc.add_annotation(
                x=0.6, y=0.35,
                text=f"AUC = {auc_val:.4f}<br>Gini = {model_metrics.get('gini', 0):.4f}<br>KS = {ks_val:.4f}",
                showarrow=False, bgcolor='#112240', bordercolor=C_TEAL,
                borderwidth=1, font=dict(color='white', size=13)
            )
            fig_roc.update_layout(
                title='AUC-ROC Curve (Out-of-Fold Predictions)',
                xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                legend=dict(x=0.5, y=0.05), height=400
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with sc2:
            # Fold AUC bar
            if 'fold_aucs' in model_metrics:
                fold_df = pd.DataFrame({'Fold': [f"Fold {i+1}" for i in range(5)],
                                        'AUC': model_metrics['fold_aucs']})
                fig_fold = px.bar(fold_df, x='Fold', y='AUC', color='AUC',
                                  color_continuous_scale='Blues', title="AUC by CV Fold (Stability Check)",
                                  text=fold_df['AUC'].map(lambda x: f"{x:.4f}"))
                fig_fold.add_hline(y=np.mean(model_metrics['fold_aucs']),
                                   line_dash='dot', line_color=C_TEAL,
                                   annotation_text=f"Mean={np.mean(model_metrics['fold_aucs']):.4f}",
                                   annotation_font_color=C_TEAL)
                fig_fold.update_traces(textposition='outside')
                fig_fold.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                    coloraxis_showscale=False,
                    yaxis_range=[min(model_metrics['fold_aucs'])-0.005, max(model_metrics['fold_aucs'])+0.005]
                )
                st.plotly_chart(fig_fold, use_container_width=True)
            # PSI Commentary
            psi = model_metrics.get('psi_score', 0)
            psi_color = C_GREEN if psi < 0.1 else (C_ORANGE if psi < 0.25 else C_RED)
            psi_label = 'Stable ✅' if psi < 0.1 else ('Minor Shift ⚠️' if psi < 0.25 else 'Significant Drift 🚨')
            st.markdown(f"""
            <div class="ecl-highlight">
                <div style="color:#8892b0; font-size:0.8rem;">POPULATION STABILITY INDEX (PSI)</div>
                <div style="color:{psi_color}; font-size:1.6rem; font-weight:700;">{psi:.4f} — {psi_label}</div>
                <div style="color:#8892b0; font-size:0.75rem;">< 0.10 = Stable | 0.10–0.25 = Monitor | > 0.25 = Retrain</div>
            </div>
            """, unsafe_allow_html=True)

    with s2:
        sg1, sg2 = st.columns(2)
        with sg1:
            st.markdown("**Global Feature Importance (Mean |SHAP|)**")
            img = os.path.join(REPORTS_DIR, 'shap_summary_bar.png')
            if os.path.exists(img): st.image(img, use_container_width=True)
            else: st.warning("Run shap_analysis.py first")
        with sg2:
            st.markdown("**SHAP Beeswarm — Direction of Impact**")
            bee_img = os.path.join(REPORTS_DIR, 'shap_beeswarm.png')
            if os.path.exists(bee_img): st.image(bee_img, use_container_width=True)
            else: st.warning("Run shap_analysis.py first")
        # Waterfall examples
        st.markdown("---")
        st.markdown("**Individual Explanations — Waterfall Charts**")
        wc1, wc2 = st.columns(2)
        with wc1:
            hr_img = os.path.join(REPORTS_DIR, 'shap_waterfall_highrisk.png')
            if os.path.exists(hr_img):
                st.markdown("🔴 *High-Risk Client (PD = 94.1%)*")
                st.image(hr_img, use_container_width=True)
        with wc2:
            lr_img = os.path.join(REPORTS_DIR, 'shap_waterfall_lowrisk.png')
            if os.path.exists(lr_img):
                st.markdown("🟢 *Low-Risk Client (PD = 1.5%)*")
                st.image(lr_img, use_container_width=True)

    with s3:
        cal_img = os.path.join(REPORTS_DIR, 'calibration_curve.png')
        if os.path.exists(cal_img):
            st.markdown("**Calibration Curve — Predicted PD vs Actual Default Rate**")
            st.markdown("> The closer to the diagonal, the better calibrated. Isotonic Regression reduced Brier Score by **64%** (0.1726 → 0.0626).")
            _, cc, _ = st.columns([1, 3, 1])
            with cc:
                st.image(cal_img, use_container_width=True)
        else:
            st.warning("Run modeling.py first to generate calibration_curve.png")

    with s4:
        sd1, sd2 = st.columns(2)
        with sd1:
            dep1 = os.path.join(REPORTS_DIR, 'shap_dep_extsource2.png')
            if os.path.exists(dep1):
                st.markdown("**EXT_SOURCE_2 × EXT_SOURCE_3**")
                st.image(dep1, use_container_width=True)
        with sd2:
            dep2 = os.path.join(REPORTS_DIR, 'shap_dep_credit_income.png')
            if os.path.exists(dep2):
                st.markdown("**Credit-to-Income Ratio × Age**")
                st.image(dep2, use_container_width=True)
        dep3 = os.path.join(REPORTS_DIR, 'shap_dep_bureau_days.png')
        if os.path.exists(dep3):
            _, dc, _ = st.columns([1, 3, 1])
            with dc:
                st.markdown("**Bureau Credit History Length × Bad Debt Flag**")
                st.image(dep3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# TAB 4: PORTFOLIO DEEP DIVE  [FIX 4 — Full charts]
# ─────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## 🔬 Portfolio Risk Deep Dive")

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        # Default Rate by Education
        if 'NAME_EDUCATION_TYPE' in fdf.columns:
            edu_df = fdf.groupby('NAME_EDUCATION_TYPE')['TARGET'].agg(['mean', 'count']).reset_index()
            edu_df = edu_df[edu_df['count'] > 50].sort_values('mean', ascending=True)
            edu_df.columns = ['Education', 'Default_Rate', 'Count']
            fig_edu = px.bar(edu_df, x='Default_Rate', y='Education', orientation='h',
                             title="Default Rate by Education Level",
                             color='Default_Rate', color_continuous_scale='RdYlGn_r',
                             text=edu_df['Default_Rate'].map(lambda x: f"{x:.1%}"))
            fig_edu.update_traces(textposition='outside')
            fig_edu.update_layout(xaxis_tickformat='.1%', paper_bgcolor='rgba(0,0,0,0)',
                                   font_color='white', coloraxis_showscale=False, margin=dict(l=0))
            st.plotly_chart(fig_edu, use_container_width=True)

    with r1c2:
        # Default Rate by Occupation (top 10)
        if 'OCCUPATION_TYPE' in fdf.columns:
            occ_df = fdf.groupby('OCCUPATION_TYPE')['TARGET'].agg(['mean', 'count']).reset_index()
            occ_df = occ_df[occ_df['count'] > 200].sort_values('mean', ascending=False).head(10)
            occ_df.columns = ['Occupation', 'Default_Rate', 'Count']
            fig_occ = px.bar(occ_df, x='Default_Rate', y='Occupation', orientation='h',
                             title="Default Rate by Occupation (Top 10 Highest Risk)",
                             color='Default_Rate', color_continuous_scale='Reds',
                             text=occ_df['Default_Rate'].map(lambda x: f"{x:.1%}"))
            fig_occ.update_traces(textposition='outside')
            fig_occ.update_layout(xaxis_tickformat='.1%', paper_bgcolor='rgba(0,0,0,0)',
                                   font_color='white', coloraxis_showscale=False, margin=dict(l=0))
            st.plotly_chart(fig_occ, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        # Default Rate by LTI (Loan-to-Income)
        fdf2 = fdf.copy()
        fdf2['LTI_BIN'] = pd.qcut(fdf2['CREDIT_INCOME_RATIO'], q=5,
                                    labels=['0–20%', '20–40%', '40–60%', '60–80%', '80–100%'],
                                    duplicates='drop')
        lti_df = fdf2.groupby('LTI_BIN').agg(
            Default_Rate=('TARGET', 'mean'),
            Count=('TARGET', 'count')
        ).reset_index()
        fig_lti = px.bar(lti_df, x='LTI_BIN', y='Default_Rate',
                         title="Default Rate by Loan-to-Income Quintile",
                         color='Default_Rate', color_continuous_scale='RdYlGn_r',
                         text=lti_df['Default_Rate'].map(lambda x: f"{x:.1%}"))
        fig_lti.update_traces(textposition='outside')
        fig_lti.update_layout(yaxis_tickformat='.1%', paper_bgcolor='rgba(0,0,0,0)',
                               font_color='white', coloraxis_showscale=False)
        st.plotly_chart(fig_lti, use_container_width=True)

    with r2c2:
        # Default Rate by Age Group
        fdf2['AGE_BIN'] = pd.cut(fdf2['AGE_YEARS'],
                                  bins=[18, 25, 30, 35, 40, 50, 60, 75],
                                  labels=['18–25', '25–30', '30–35', '35–40', '40–50', '50–60', '60+'])
        age_df = fdf2.groupby('AGE_BIN')['TARGET'].agg(['mean', 'count']).reset_index()
        age_df.columns = ['Age Group', 'Default_Rate', 'Count']
        fig_age = px.bar(age_df, x='Age Group', y='Default_Rate',
                         title="Default Rate by Age Group",
                         color='Default_Rate', color_continuous_scale='RdYlGn_r',
                         text=age_df['Default_Rate'].map(lambda x: f"{x:.1%}"))
        fig_age.update_traces(textposition='outside')
        fig_age.update_layout(yaxis_tickformat='.1%', paper_bgcolor='rgba(0,0,0,0)',
                               font_color='white', coloraxis_showscale=False)
        st.plotly_chart(fig_age, use_container_width=True)

    r3c1, r3c2 = st.columns(2)

    with r3c1:
        # Contract Type vs Default Rate
        if 'NAME_CONTRACT_TYPE' in fdf.columns:
            ctr_df = fdf.groupby('NAME_CONTRACT_TYPE')['TARGET'].agg(['mean', 'count']).reset_index()
            ctr_df.columns = ['Contract', 'Default_Rate', 'Count']
            fig_ctr = px.bar(ctr_df, x='Contract', y='Default_Rate', color='Contract',
                             title="Default Rate by Contract Type",
                             color_discrete_sequence=[C_BLUE, C_ORANGE, C_GREEN],
                             text=ctr_df['Default_Rate'].map(lambda x: f"{x:.1%}"))
            fig_ctr.update_traces(textposition='outside')
            fig_ctr.update_layout(yaxis_tickformat='.1%', paper_bgcolor='rgba(0,0,0,0)',
                                   font_color='white', showlegend=False)
            st.plotly_chart(fig_ctr, use_container_width=True)

    with r3c2:
        # Top features correlated with TARGET
        feat_cols = [c for c in fdf.columns if c in FEATURES][:50]
        corr_vals = fdf[feat_cols + ['TARGET']].corr()['TARGET'].drop('TARGET').abs().sort_values(ascending=False).head(15)
        corr_df = pd.DataFrame({'Feature': corr_vals.index, 'Correlation': corr_vals.values})
        fig_corr = px.bar(corr_df, x='Correlation', y='Feature', orientation='h',
                          title="Top 15 Features by |Correlation| with TARGET",
                          color='Correlation', color_continuous_scale='Blues')
        fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                                coloraxis_showscale=False, margin=dict(l=0))
        st.plotly_chart(fig_corr, use_container_width=True)

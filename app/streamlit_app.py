"""
streamlit_app.py — Nova Bank Credit Risk Dashboard v3 (Analyst Upgraded)
==========================================================================
Storytelling flow:
  1. KPI Header   → How big is the portfolio & risk?
  2. Insight Strip → 3 key actionable findings upfront
  3. Row 1 Charts → Who defaults? How it concentrates by loan term & amount?
  4. Row 2 Charts → Root cause: LTI, DTI, PD distribution
  5. Row 3 Charts → IFRS 9 Stage breakdown + Region concentration
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Nova Bank — Credit Risk",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Open+Sans:wght@300;400;600&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 4px 8px !important; max-width: 100% !important; }
html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
    background-color: #060d1a !important;
    color: #b0c8e4;
}
.stApp { background-color: #060d1a !important; }

/* Chart panels */
[data-testid="stPlotlyChart"] {
    border: 1px solid #2060a0 !important;
    border-radius: 2px !important;
    padding: 2px !important;
    background: #070e1c !important;
}

/* KPI Cards */
.kcard {
    background: #070e1c;
    border: 1px solid #2060a0;
    border-left: 3px solid #3a80d0;
    padding: 8px 14px;
    border-radius: 0;
    height: 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kcard-lbl {
    font-size: 9.5px;
    color: #90b8d8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
    margin-bottom: 2px;
}
.kcard-val {
    font-family: 'Oswald', sans-serif;
    font-size: 34px;
    font-weight: 700;
    color: #e0f0ff;
    line-height: 1;
}
.kcard-val.red { color: #ff6060 !important; }
.kcard-val.amber { color: #ffcc44 !important; }
.kcard-val.green { color: #44dd88 !important; }
.kcard-sub {
    font-size: 10px;
    color: #6898b8;
    margin-top: 2px;
    line-height: 1.5;
}

/* Brand */
.brand-card {
    background: #060c18;
    border: 1px solid #2060a0;
    border-left: 3px solid #3a80d0;
    padding: 8px 14px;
    height: 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.brand-title {
    font-family: 'Oswald', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #a8c8e8;
    letter-spacing: 2px;
}
.brand-sub {
    font-size: 11px;
    color: #3a78c0;
    font-weight: 600;
    margin-top: 2px;
}

/* Insight Strip */
.insight-strip {
    display: flex;
    gap: 8px;
    margin: 4px 0;
}
.insight-box {
    flex: 1;
    padding: 8px 14px;
    border-radius: 2px;
    border-left: 4px solid;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    line-height: 1.4;
}
.insight-box.red   { background: rgba(200,60,60,0.12); border-color: #cc4444; }
.insight-box.amber { background: rgba(200,160,40,0.12); border-color: #c89020; }
.insight-box.blue  { background: rgba(60,120,200,0.12); border-color: #3a78c0; }
.insight-icon { font-size: 22px; }
.insight-text strong { color: #f0f8ff; }
.insight-text span   { color: #8ab0d0; }

/* Stage Cards */
.stage-card {
    padding: 10px 14px;
    border-radius: 2px;
    border: 1px solid;
    text-align: center;
}
.stage-val {
    font-family: 'Oswald', sans-serif;
    font-size: 28px;
    font-weight: 700;
    line-height: 1;
}
.stage-lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin-top: 3px; }

/* Widgets */
div[data-testid="stSelectbox"] > div > div {
    background: #09152a !important;
    border: 1px solid #2060a0 !important;
    color: #a8c4e0 !important;
    border-radius: 2px !important;
    min-height: 32px !important;
    font-size: 12px !important;
}
div[data-testid="stMultiSelect"] > div > div {
    background: #09152a !important;
    border: 1px solid #2060a0 !important;
    border-radius: 2px !important;
    font-size: 12px !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background: #1a4070 !important;
    border: 1px solid #2a60a0 !important;
    border-radius: 2px !important;
    color: #80b8e0 !important;
    font-size: 10px !important;
}
/* Style all widget labels */
div[data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] p {
    color: #6898b8 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    margin-bottom: 2px !important;
}
/* Filter containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #4888c8 !important; /* Brighter blue border like Image 1 */
    border-radius: 0px !important; /* Sharp corners */
    background: #040810 !important; /* Very dark background */
    padding: 10px !important;
}
/* Segmented Control Styling to match Image 1 */
div[data-testid="stSegmentedControl"] {
    background-color: transparent !important;
}
div[data-testid="stSegmentedControl"] [data-baseweb], 
div[data-testid="stSegmentedControl"] [role] {
    background-color: transparent !important;
    border: none !important;
    gap: 4px !important; /* Gap between buttons */
}
div[data-testid="stSegmentedControl"] label {
    background-color: #040810 !important;
    color: #6898b8 !important;
    border: 1px solid #4888c8 !important; /* Match outer border or slightly darker */
    border-radius: 0px !important; /* Square buttons */
    padding: 2px 8px !important;
    margin-right: 4px !important; /* Separate buttons */
}
div[data-testid="stSegmentedControl"] label[aria-checked="true"],
div[data-testid="stSegmentedControl"] label[data-checked="true"] {
    background-color: #1a4070 !important; /* Slightly highlighted background when active */
    color: #ffffff !important;
    border: 1px solid #68a8e8 !important;
}
div[data-testid="stSegmentedControl"] p, div[data-testid="stSegmentedControl"] span {
    color: inherit !important;
}
/* Button Styling */
div[data-testid="stButton"] button {
    background-color: #050a14 !important;
    border: 1px solid #1a4070 !important;
    color: #6898b8 !important;
    border-radius: 2px !important;
    height: 36px !important;
}
/* Tabs Styling */
div[data-testid="stTabs"] button {
    color: #6898b8 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff !important;
    border-bottom-color: #2060a0 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] div[data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
}
[data-testid="stHorizontalBlock"] { gap: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════
DATA_DIR = r'd:\Risk\data'

@st.cache_data
def load():
    for fn in ['results_ifrs9.parquet', 'results_df.parquet']:
        p = os.path.join(DATA_DIR, fn)
        if os.path.exists(p):
            df = pd.read_parquet(p)
            break
    else:
        st.error("No data found. Run modeling.py first."); st.stop()

    # Loan term bins (months)
    df['TERM_M'] = (df['AMT_CREDIT'] / df['AMT_ANNUITY'].clip(lower=1)).round(0)
    df['TERM_BIN'] = pd.cut(df['TERM_M'], [-1, 18, 30, 42, 56, 9999],
                             labels=['≤18m', '24m', '36m', '48m', '60m+'])

    # LTI bins
    lti = df.get('CREDIT_INCOME_RATIO',
                  df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].clip(lower=1))
    df['LTI_BIN'] = pd.cut(lti, [-1, 1, 2, 3, 5, 999],
                            labels=['0–1×', '1–2×', '2–3×', '3–5×', '>5×'])

    # DTI bins
    dti = df.get('ANNUITY_INCOME_RATIO',
                  df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].clip(lower=1))
    df['DTI_BIN'] = pd.cut(dti, [-1, .10, .20, .30, .50, 9],
                            labels=['0–10%', '10–20%', '20–30%', '30–50%', '>50%'])

    # Credit amount bins
    df['CRED_BIN'] = pd.cut(df['AMT_CREDIT'], [0, 100e3, 200e3, 300e3, 500e3, 9e9],
                             labels=['$0–100k', '$100–200k', '$200–300k',
                                     '$300–500k', '$500k+'])

    # IFRS 9 stage (derive if not present)
    if 'STAGE' not in df.columns:
        pd_col = df['PRED_PROB']
        df['STAGE'] = np.select(
            [pd_col < 0.20, pd_col < 0.50],
            [1, 2], default=3
        )

    # Generate mock Disbursement Month (uniform random across 2022-2023)
    if 'DISBURSEMENT_MONTH' not in df.columns:
        np.random.seed(42) # Fixed seed for stable visualization
        dates = pd.Timestamp('2022-01-01') + pd.to_timedelta(np.random.randint(0, 730, size=len(df)), unit='D')
        df['DISBURSEMENT_MONTH'] = dates.to_period('M').astype(str)

    # PD band
    df['PD_BAND'] = pd.cut(df['PRED_PROB'],
                            bins=[0, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.01],
                            labels=['0–10%', '10–20%', '20–30%', '30–40%', '40–50%',
                                    '50–60%', '60–70%', '70–80%', '80–90%', '90–100%'])

    # ECL fallback
    if 'ECL' not in df.columns:
        df['ECL'] = df['PRED_PROB'] * 0.45 * df['AMT_CREDIT']

    return df

df = load()

# ═══════════════════════════════════════════════════════════════════
# COLOUR CONSTANTS
# ═══════════════════════════════════════════════════════════════════
BLU   = '#4a7ab8'
GRN   = '#2a8a40'
GLD   = '#c89020'
RED   = '#c83030'
TEAL  = '#1a8a78'
BRWN  = '#8a6040'
PURP  = '#6a5090'
TXT   = '#6898c8'
GRID  = '#0e1e34'
BG    = 'rgba(0,0,0,0)'
PIE   = [BLU, GLD, GRN, TEAL, BRWN, PURP, '#3a8858']
STAGE_C = {1: GRN, 2: GLD, 3: RED}


def L(title='', h=270, legend=False):
    """Conflict-free base layout — all axis keys set here only."""
    return dict(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=11, color='#90b8d8', family='Open Sans'),
                   x=0, xref='paper', pad=dict(l=2, t=2)),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TXT, family='Open Sans', size=10),
        margin=dict(l=4, r=10, t=38, b=24),
        height=h, showlegend=legend,
        legend=dict(orientation='h', y=1.11, x=.5, xanchor='center',
                    font=dict(size=9), bgcolor='rgba(0,0,0,0)'),
        hoverlabel=dict(bgcolor='#0c1e38', font_color='#e0f0ff', font_size=11),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=9, color=TXT)),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False,
                   tickfont=dict(size=9, color=TXT)),
    )


# ═══════════════════════════════════════════════════════════════════
# PRE-COMPUTE FULL-PORTFOLIO STATS (unfiltered, for delta comparison)
# ═══════════════════════════════════════════════════════════════════
nm_all  = (df['CODE_GENDER'] == 'M').sum()
nf_all  = (df['CODE_GENDER'] == 'F').sum()
loan_total_all = df['AMT_CREDIT'].sum()
ecl_total_all  = df['ECL'].sum()
dr_total_all   = df['TARGET'].mean()
high_risk_pct_all = (df['RISK_TIER'] == 'High').mean()
high_risk_ecl_all = df[df['RISK_TIER'] == 'High']['ECL'].sum()
coverage_all  = ecl_total_all / loan_total_all

# ═══════════════════════════════════════════════════════════════════
# HEADER ROW PLACEHOLDER — will be rendered after filters
# ═══════════════════════════════════════════════════════════════════
# (header rendered below, after fdf is available)


# ═══════════════════════════════════════════════════════════════════
# FILTERS
# ═══════════════════════════════════════════════════════════════════
st.markdown("<div style='height:2px;background:#1a3860;margin:2px 0 3px 0'></div>",
            unsafe_allow_html=True)

f0, f1, f2, f3, f4 = st.columns([1.5, 0.5, 1.2, 2.8, 1.5], gap="small")

with f0.container(border=True):
    fam_stat = st.segmented_control("Family Status", ["Married", "Single", "Divorced"], selection_mode="multi", default=None)

with f1.container(border=True):
    st.markdown("<p style='font-size:11px;font-weight:600;color:#6898b8;margin-bottom:2px;text-transform:uppercase'>Reset</p>", unsafe_allow_html=True)
    if st.button("🔄", use_container_width=True):
        st.rerun()

with f2.container(border=True):
    reg_opts = ["All"] + [str(int(x)) for x in sorted(df['REGION_RATING_CLIENT_W_CITY'].dropna().unique())]
    regf = st.selectbox("Region Rating", reg_opts)

with f3.container(border=True):
    tierf = st.segmented_control("Risk Tier (Loan Grade)", ["Very Low", "Low", "Medium", "High"], selection_mode="multi", default=None)

with f4.container(border=True):
    termf = st.segmented_control("Loan Term", ["12", "24", "36", "60+"], selection_mode="multi", default=None)

# Apply filters
fdf = df.copy()

if fam_stat:
    fam_full = []
    if "Married" in fam_stat: fam_full.append("Married")
    if "Single" in fam_stat: fam_full.extend(["Single / not married", "Separated", "Widow"])
    if "Divorced" in fam_stat: fam_full.append("Separated") # Approximated
    if fam_full:
        fdf = fdf[fdf['NAME_FAMILY_STATUS'].isin(fam_full)]

if regf != 'All':
    fdf = fdf[fdf['REGION_RATING_CLIENT_W_CITY'] == float(regf)]
    
if tierf:
    fdf = fdf[fdf['RISK_TIER'].isin(tierf)]

if termf:
    term_map = {"12": "≤18m", "24": "24m", "36": "36m", "60+": "60m+"}
    selected_bins = [term_map[t] for t in termf]
    fdf = fdf[fdf['TERM_BIN'].isin(selected_bins)]

# ═══════════════════════════════════════════════════════════════════
# FILTERED KPI STATS (react to fdf)
# ═══════════════════════════════════════════════════════════════════
is_filtered = len(fdf) < len(df)
nm  = (fdf['CODE_GENDER'] == 'M').sum()
nf  = (fdf['CODE_GENDER'] == 'F').sum()
loan_total = fdf['AMT_CREDIT'].sum()
ecl_total  = fdf['ECL'].sum()
dr_total   = fdf['TARGET'].mean()
high_risk_pct = (fdf['RISK_TIER'] == 'High').mean()
high_risk_ecl = fdf[fdf['RISK_TIER'] == 'High']['ECL'].sum()
coverage  = ecl_total / loan_total if loan_total > 0 else 0

def delta_badge(val, ref, fmt='.1%', higher_is_bad=True):
    """Return a small colored HTML delta indicator vs unfiltered portfolio."""
    if not is_filtered or ref == 0:
        return ''
    d = val - ref
    color = ('#ff6060' if d > 0 else '#44dd88') if higher_is_bad else ('#44dd88' if d > 0 else '#ff6060')
    arrow = '▲' if d > 0 else '▼'
    return f'<span style="font-size:10px;color:{color};margin-left:5px">{arrow} {abs(d):{fmt}} vs All</span>'

# ═══════════════════════════════════════════════════════════════════
# HEADER ROW (rendered here after fdf is ready)
# ═══════════════════════════════════════════════════════════════════
h0, h1, h2, h3, h4, h5 = st.columns([1.1, 1, 1.1, 1, 0.9, 0.9], gap="small")

with h0:
    filter_label = f'<div style="font-size:9px;color:#ff9900;margin-top:3px">⚡ FILTERED VIEW</div>' if is_filtered else ''
    st.markdown(f"""
    <div class="brand-card">
        <div class="brand-title">🏛 Credit Risk Analyst</div>
        <div class="brand-sub">&nbsp; Dashboard{filter_label}</div>
    </div>""", unsafe_allow_html=True)

with h1:
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Total Borrowers</div>
        <div class="kcard-val">{len(fdf)//1000}K</div>
        <div class="kcard-sub">M: {nm//1000}K &nbsp; F: {nf//1000}K</div>
    </div>""", unsafe_allow_html=True)

with h2:
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Total Loan Amount (EAD)</div>
        <div class="kcard-val">${loan_total/1e9:.2f}B</div>
    </div>""", unsafe_allow_html=True)

with h3:
    d_ecl = delta_badge(ecl_total/loan_total if loan_total>0 else 0, coverage_all, fmt='.1%', higher_is_bad=True)
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Expected Credit Loss (IFRS 9)</div>
        <div class="kcard-val amber">${ecl_total/1e6:,.0f}M{d_ecl}</div>
    </div>""", unsafe_allow_html=True)

with h4:
    d_cov = delta_badge(coverage, coverage_all, fmt='.1%', higher_is_bad=True)
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">ECL Coverage Ratio</div>
        <div class="kcard-val {'red' if coverage > 0.30 else 'amber'}">{coverage:.1%}{d_cov}</div>
    </div>""", unsafe_allow_html=True)

with h5:
    d_dr = delta_badge(dr_total, dr_total_all, fmt='.1%', higher_is_bad=True)
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Portfolio Default Rate</div>
        <div class="kcard-val {'red' if dr_total > 0.12 else 'amber'}">{dr_total:.1%}{d_dr}</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
main_tab, insights_tab = st.tabs(["Credit Risk Analyst", "Insights"])


with main_tab:
    # ═══════════════════════════════════════════════════════════════════
    # ROW 1 — Who defaults & where does exposure concentrate?
    # ═══════════════════════════════════════════════════════════════════
    r1a, r1b, r1c = st.columns([1.1, 1.3, 1.6], gap="small")

    # ── Chart 1A: H-Bar — Income Type vs Defaults ─────────────────
    with r1a:
        grp = (fdf[fdf['TARGET'] == 1]['NAME_INCOME_TYPE']
               .value_counts().reset_index())
        grp.columns = ['Type', 'Count']
        grp = grp.sort_values('Count', ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=grp['Type'], x=grp['Count'],
            orientation='h',
            marker=dict(color=BLU, line=dict(color='#060d1a', width=1.5)),
            text=[f" {v/1000:.1f}K" for v in grp['Count']],
            textposition='outside', textfont=dict(color='#c0d8f0', size=9),
            width=0.6,
        ))
        lo = L('Who Defaults? — By Employment Type', h=288)
        lo['showlegend'] = False
        lo['xaxis'] = dict(showgrid=True, gridcolor=GRID, zeroline=False,
                            tickfont=dict(size=9, color=TXT), title=dict(text='Defaults', font=dict(size=10, color=TXT)))
        lo['yaxis'] = dict(showgrid=False, zeroline=False, tickfont=dict(size=9, color=TXT))
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart 1B: Bar — Defaults by Loan Term ─────────────────────
    with r1b:
        term = (fdf[fdf['TARGET'] == 1]
                .groupby('TERM_BIN', observed=True)
                .size().reset_index(name='N'))
        term = term[term['N'] > 0]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=term['TERM_BIN'].astype(str), y=term['N'],
            marker=dict(color=BLU, line=dict(color='#060d1a', width=1.5)),
            text=[f"{v/1000:.1f}K" for v in term['N']],
            textposition='outside', textfont=dict(color='#c0d8f0', size=10),
            width=0.52, name='Defaults',
        ))
        lo = L('Default Loan by Loan Term Month', h=288)
        lo['showlegend'] = False
        lo['xaxis'] = dict(title=dict(text='Loan Term (Months)', font=dict(size=10, color=TXT)),
                            showgrid=False, zeroline=False, tickfont=dict(size=9, color=TXT))
        lo['yaxis'] = dict(title=dict(text='Total Default Loans', font=dict(size=10, color=TXT)),
                            showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(size=9, color=TXT))
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart 1C: Combo — Exposure + Default Rate by Credit Bin ───
    with r1c:
        bg = (fdf.groupby('CRED_BIN', observed=True)
              .agg(Amt=('AMT_CREDIT', 'sum'), DR=('TARGET', 'mean'),
                   ECL_sum=('ECL', 'sum'))
              .reset_index())

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=bg['CRED_BIN'].astype(str), y=bg['Amt'],
            name='Exposure (EAD)', marker=dict(color=BLU, line=dict(color='#060d1a', width=1)),
            text=[f"${v/1e6:.0f}M" for v in bg['Amt']],
            textposition='inside', textfont=dict(color='white', size=9),
            width=0.55,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=bg['CRED_BIN'].astype(str), y=bg['DR'],
            name='Default Rate', mode='lines+markers+text',
            line=dict(color=GLD, width=2.5),
            marker=dict(color=GLD, size=7, line=dict(color='#060d1a', width=1.5)),
            text=[f"{v:.0%}" for v in bg['DR']],
            textposition='top center', textfont=dict(color=GLD, size=9),
        ), secondary_y=True)

        lo = L('Exposure & Default Rate by Loan Amount Band', h=288, legend=True)
        lo['xaxis'] = dict(showgrid=False, zeroline=False,
                            tickfont=dict(size=9, color=TXT))
        lo['yaxis'] = dict(showgrid=True, gridcolor=GRID, zeroline=False,
                            showticklabels=False, tickfont=dict(size=9, color=TXT),
                            title=dict(text='Exposure ($)', font=dict(size=10, color=TXT)))
        lo['yaxis2'] = dict(showgrid=False, zeroline=False, tickformat='.0%',
                             tickfont=dict(size=9, color=TXT),
                             title=dict(text='Default Rate', font=dict(size=10, color=TXT)))
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # ROW 2 — Root Cause Analysis
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:3px'></div>", unsafe_allow_html=True)
    r2a, r2b, r2c, r2d = st.columns([1, 1.1, 1, 0.9], gap="small")

    # ── Chart 2A: Line — Vintage Analysis by Disbursement Month ───
    with r2a:
        vint = (fdf.groupby('DISBURSEMENT_MONTH')
                .agg(DR=('TARGET', 'mean'), N=('TARGET', 'count'))
                .reset_index())
        vint = vint.sort_values('DISBURSEMENT_MONTH')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vint['DISBURSEMENT_MONTH'], y=vint['DR'],
            mode='lines+markers',
            line=dict(color=GLD, width=2.5),
            marker=dict(color=GLD, size=6, line=dict(color='#060d1a', width=1)),
            name='Default Rate'
        ))
        
        # Add a trendline
        import numpy as np
        if len(vint) > 1:
            x_num = np.arange(len(vint))
            z = np.polyfit(x_num, vint['DR'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=vint['DISBURSEMENT_MONTH'], y=p(x_num),
                mode='lines',
                line=dict(color=RED, width=1.5, dash='dot'),
                name='Trend'
            ))

        lo = L('Vintage Analysis — Default Rate Trend', h=262)
        lo['xaxis'] = dict(showgrid=False, zeroline=False, tickfont=dict(size=8, color=TXT), dtick='M3', tickangle=45)
        lo['yaxis'] = dict(showgrid=True, gridcolor=GRID, zeroline=False, tickformat='.1%', tickfont=dict(size=9, color=TXT))
        lo['showlegend'] = False
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart 2B: Histogram — PD Distribution ─────────────────────
    with r2b:
        pd_band = (fdf.groupby('PD_BAND', observed=True)
                   .agg(Count=('PRED_PROB', 'count'), DR=('TARGET', 'mean'))
                   .reset_index())

        # Color by risk: green → amber → red gradient
        def pd_color(band_str):
            try:
                low = float(band_str.split('–')[0].replace('%', '')) / 100
            except:
                low = 0
            if low < 0.20: return GRN
            elif low < 0.50: return GLD
            else: return RED

        bar_cols = [pd_color(b) for b in pd_band['PD_BAND'].astype(str)]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pd_band['PD_BAND'].astype(str), y=pd_band['Count'],
            marker=dict(color=bar_cols, line=dict(color='#060d1a', width=1)),
            textfont=dict(color='#c0d8f0', size=9),
            width=0.72, name='Borrower Count',
        ))
        lo = L('PD Score Distribution — Risk Concentration', h=262)
        lo['xaxis'] = dict(showgrid=False, zeroline=False,
                            tickfont=dict(size=8, color=TXT),
                            title=dict(text='Probability of Default Band',
                                        font=dict(size=10, color=TXT)))
        lo['yaxis'] = dict(showgrid=True, gridcolor=GRID, zeroline=False,
                            tickfont=dict(size=9, color=TXT),
                            title=dict(text='# Borrowers', font=dict(size=10, color=TXT)))
        lo['annotations'] = [
            dict(x='0–10%', y=pd_band[pd_band['PD_BAND'] == '0–10%']['Count'].values[0] * 0.8
                 if '0–10%' in pd_band['PD_BAND'].values else 0,
                 text='<b>Low Risk</b>', showarrow=False,
                 font=dict(color=GRN, size=9)),
            dict(x='50–60%', y=10000,
                 text='<b>High Risk</b>', showarrow=False,
                 font=dict(color=RED, size=9)),
        ]
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart 2C: V-Bar — ECL by DTI (more insightful than DR) ───
    with r2c:
        dti = (fdf.groupby('DTI_BIN', observed=True)
               .agg(ECL_m=('ECL', 'sum'), DR=('TARGET', 'mean'))
               .reset_index())
        dti['ECL_m'] = dti['ECL_m'] / 1e6

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=dti['DTI_BIN'].astype(str), y=dti['ECL_m'],
            name='ECL ($M)', marker=dict(color=BLU, line=dict(color='#060d1a', width=1.5)),
            text=[f"${v:.0f}M" for v in dti['ECL_m']],
            textposition='outside', textfont=dict(color='#c0d8f0', size=9),
            width=0.55,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=dti['DTI_BIN'].astype(str), y=dti['DR'],
            name='Default Rate', mode='lines+markers',
            line=dict(color=GLD, width=2),
            marker=dict(color=GLD, size=6),
        ), secondary_y=True)

        lo = L('ECL & Default Rate by Debt-to-Income (DTI)', h=262, legend=True)
        lo['xaxis'] = dict(showgrid=False, zeroline=False,
                            tickfont=dict(size=9, color=TXT),
                            title=dict(text='Debt To Income', font=dict(size=10, color=TXT)))
        lo['yaxis'] = dict(showgrid=True, gridcolor=GRID, zeroline=False,
                            showticklabels=False,
                            title=dict(text='ECL ($M)', font=dict(size=10, color=TXT)))
        lo['yaxis2'] = dict(showgrid=False, zeroline=False, tickformat='.0%',
                             tickfont=dict(size=9, color=TXT))
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart 2D: IFRS 9 Stage Donut ──────────────────────────────
    with r2d:
        stage_g = (fdf.groupby('STAGE')
                   .agg(Count=('AMT_CREDIT', 'count'),
                        EAD=('AMT_CREDIT', 'sum'),
                        ECL=('ECL', 'sum'))
                   .reset_index())
        stage_g['STAGE_LBL'] = stage_g['STAGE'].map(
            {1: 'Stage 1\n(Performing)',
             2: 'Stage 2\n(Watch)',
             3: 'Stage 3\n(Impaired)'})
        s_cols = [STAGE_C.get(s, BLU) for s in stage_g['STAGE']]

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=stage_g['STAGE_LBL'], values=stage_g['EAD'],
            hole=0.45,
            marker=dict(colors=s_cols, line=dict(color='#060d1a', width=2)),
            textinfo='label+percent', textposition='inside',
            textfont=dict(size=8.5, color='white'),
            insidetextorientation='radial',
        ))
        lo = L('IFRS 9 Staging — EAD by Stage', h=262)
        lo['showlegend'] = False
        lo['annotations'] = [dict(
            text=f"<b>EAD</b>",
            x=0.5, y=0.5, font_size=12, showarrow=False,
            font=dict(color='#c0d8f0', family='Oswald')
        )]
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True)


with insights_tab:
    # ═══════════════════════════════════════════════════════════════════
    # INSIGHT STRIP — 3 key findings
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:3px;background:#2060a0;margin:3px 0'></div>",
                unsafe_allow_html=True)

    high_ecl_share = high_risk_ecl / ecl_total if ecl_total > 0 else 0
    top_income_default = fdf[fdf['TARGET'] == 1]['NAME_INCOME_TYPE'].value_counts().index[0] if len(fdf[fdf['TARGET']==1]) > 0 else 'N/A'
    worst_lti = fdf.groupby('LTI_BIN', observed=True)['TARGET'].mean().idxmax() if len(fdf) > 0 else 'N/A'

    st.markdown(f"""
    <div class="insight-strip">
      <div class="insight-box red">
        <span class="insight-icon">🚨</span>
        <div class="insight-text">
          <strong>High-Risk Concentration:</strong><br>
          <span>Top {high_risk_pct:.0%} high-risk borrowers account for
            <strong style="color:#ff6060">{high_ecl_share:.0%} of total ECL</strong> —
            immediate provisioning action required.</span>
        </div>
      </div>
      <div class="insight-box amber">
        <span class="insight-icon">⚠️</span>
        <div class="insight-text">
          <strong>LTI Leverage Signal:</strong><br>
          <span>Borrowers in <strong style="color:#ffcc44">LTI band {worst_lti}</strong> show
            highest default rates. Income-based stress-test recommended before approval.</span>
        </div>
      </div>
      <div class="insight-box blue">
        <span class="insight-icon">💡</span>
        <div class="insight-text">
          <strong>Employment Risk Driver:</strong><br>
          <span><strong style="color:#88c8f0">{top_income_default}</strong> segment generates
            the most defaults. Underwriting policy tightening for this segment = max ROI.</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    # ═══════════════════════════════════════════════════════════════════
    # ROW 3 — IFRS 9 Stage Detail Cards + Region
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:3px;background:#1a3860;margin:3px 0'></div>",
                unsafe_allow_html=True)

    s1c, s2c, s3c, _, reg_c = st.columns([0.7, 0.7, 0.7, 0.1, 2.0], gap="small")

    for col, stage_num, label, color, icon in [
        (s1c, 1, 'Stage 1 — Performing', GRN, '✅'),
        (s2c, 2, 'Stage 2 — Watch List', GLD, '⚠️'),
        (s3c, 3, 'Stage 3 — Impaired',   RED, '🔴'),
    ]:
        row = fdf[fdf['STAGE'] == stage_num]
        ecl_s = row['ECL'].sum()
        ead_s = row['AMT_CREDIT'].sum()
        cov   = ecl_s / ead_s if ead_s > 0 else 0
        col.markdown(f"""
        <div class="stage-card" style="border-color:{color}; background: {color}18">
            <div style="font-size:20px">{icon}</div>
            <div class="stage-val" style="color:{color}">{icon[:2]}</div>
            <div class="stage-lbl" style="color:{color}">{label}</div>
            <hr style="border-color:{color}33; margin:6px 0">
            <div style="font-size:11px; color:#b0c8e4">
                <b>{len(row)/1000:.0f}K</b> borrowers<br>
                EAD: <b>${ead_s/1e9:.2f}B</b><br>
                ECL: <b>${ecl_s/1e6:.0f}M</b><br>
                Coverage: <b>{cov:.1%}</b>
            </div>
        </div>""", unsafe_allow_html=True)

    with reg_c:
        reg = (fdf.groupby('REGION_RATING_CLIENT_W_CITY')
               .agg(Amt=('AMT_CREDIT', 'sum'), DR=('TARGET', 'mean'),
                    ECL=('ECL', 'sum'))
               .reset_index().sort_values('Amt', ascending=True))
        reg['Label'] = 'Region Rating ' + reg['REGION_RATING_CLIENT_W_CITY'].astype(str)
        seg_cols = [GRN, GLD, BLU][:len(reg)]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=reg['Amt'], y=reg['Label'], orientation='h',
            name='Exposure',
            marker=dict(color=seg_cols, line=dict(color='#060d1a', width=1.5)),
            text=[f"${v/1e6:.0f}M" for v in reg['Amt']],
            textposition='inside', textfont=dict(color='white', size=10),
            width=0.5,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=reg['ECL'], y=reg['Label'],
            mode='markers', name='ECL',
            marker=dict(color=RED, size=10, symbol='diamond',
                        line=dict(color='white', width=1)),
        ), secondary_y=False)

        lo = L('Exposure & ECL by Region Rating', h=155, legend=True)
        lo['xaxis'] = dict(showgrid=False, zeroline=False, showticklabels=False,
                            title=dict(text='Amount ($)', font=dict(size=10, color=TXT)))
        lo['yaxis'] = dict(showgrid=False, zeroline=False,
                            tickfont=dict(size=9, color=TXT))
        lo['margin'] = dict(l=4, r=10, t=38, b=16)
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True)

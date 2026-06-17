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
.stSelectbox label, .stMultiSelect label { display: none !important; }
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
# PRE-COMPUTE PORTFOLIO STATS
# ═══════════════════════════════════════════════════════════════════
nm  = (df['CODE_GENDER'] == 'M').sum()
nf  = (df['CODE_GENDER'] == 'F').sum()
loan_total = df['AMT_CREDIT'].sum()
ecl_total  = df['ECL'].sum()
dr_total   = df['TARGET'].mean()
high_risk_pct = (df['RISK_TIER'] == 'High').mean()
high_risk_ecl = df[df['RISK_TIER'] == 'High']['ECL'].sum()
coverage  = ecl_total / loan_total

# ═══════════════════════════════════════════════════════════════════
# HEADER ROW
# ═══════════════════════════════════════════════════════════════════
h0, h1, h2, h3, h4, h5 = st.columns([1.1, 1, 1.1, 1, 0.9, 0.9], gap="small")

with h0:
    st.markdown("""
    <div class="brand-card">
        <div class="brand-title">🏛 NOVA BANK</div>
        <div class="brand-sub">&nbsp; Credit Risk Analyst</div>
    </div>""", unsafe_allow_html=True)

with h1:
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Total Borrowers</div>
        <div class="kcard-val">{len(df)//1000}K</div>
        <div class="kcard-sub">M: {nm//1000}K &nbsp; F: {nf//1000}K</div>
    </div>""", unsafe_allow_html=True)

with h2:
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Total Loan Amount (EAD)</div>
        <div class="kcard-val">${loan_total/1e9:.2f}B</div>
    </div>""", unsafe_allow_html=True)

with h3:
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Expected Credit Loss (IFRS 9)</div>
        <div class="kcard-val amber">${ecl_total/1e6:,.0f}M</div>
    </div>""", unsafe_allow_html=True)

with h4:
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">ECL Coverage Ratio</div>
        <div class="kcard-val {'red' if coverage > 0.30 else 'amber'}">{coverage:.1%}</div>
    </div>""", unsafe_allow_html=True)

with h5:
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Portfolio Default Rate</div>
        <div class="kcard-val {'red' if dr_total > 0.12 else 'amber'}">{dr_total:.1%}</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# INSIGHT STRIP — 3 key findings
# ═══════════════════════════════════════════════════════════════════
st.markdown("<div style='height:3px;background:#2060a0;margin:3px 0'></div>",
            unsafe_allow_html=True)

high_ecl_share = high_risk_ecl / ecl_total
top_income_default = df[df['TARGET'] == 1]['NAME_INCOME_TYPE'].value_counts().index[0]
worst_lti = df.groupby('LTI_BIN', observed=True)['TARGET'].mean().idxmax()

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
# FILTERS
# ═══════════════════════════════════════════════════════════════════
st.markdown("<div style='height:2px;background:#1a3860;margin:2px 0 3px 0'></div>",
            unsafe_allow_html=True)

f0, f1, f2, f3 = st.columns([1, 1, 2, 1], gap="small")
with f0:
    gf = st.selectbox("_g", ["All", "Male", "Female"], label_visibility="collapsed")
with f1:
    cf = st.selectbox("_c", ["All", "Cash loans", "Revolving loans"],
                      label_visibility="collapsed")
with f2:
    tf = st.multiselect("_t", ['Very Low', 'Low', 'Medium', 'High'],
                         default=['Very Low', 'Low', 'Medium', 'High'],
                         label_visibility="collapsed")
with f3:
    edu_opts = ["All"] + sorted(df['NAME_EDUCATION_TYPE'].dropna().unique())
    ef = st.selectbox("_e", edu_opts, label_visibility="collapsed")

st.markdown("<div style='height:2px;background:#2060a0;margin:2px 0 4px 0'></div>",
            unsafe_allow_html=True)

# Apply filters
fdf = df.copy()
if gf == 'Male':    fdf = fdf[fdf['CODE_GENDER'] == 'M']
elif gf == 'Female': fdf = fdf[fdf['CODE_GENDER'] == 'F']
if cf != 'All':     fdf = fdf[fdf['NAME_CONTRACT_TYPE'] == cf]
if tf:              fdf = fdf[fdf['RISK_TIER'].isin(tf)]
if ef != 'All':     fdf = fdf[fdf['NAME_EDUCATION_TYPE'] == ef]

# ═══════════════════════════════════════════════════════════════════
# ROW 1 — Who defaults & where does exposure concentrate?
# ═══════════════════════════════════════════════════════════════════
r1a, r1b, r1c = st.columns([1.1, 1.3, 1.6], gap="small")

# ── Chart 1A: Donut — Income Type vs Defaults ─────────────────
with r1a:
    grp = (fdf[fdf['TARGET'] == 1]['NAME_INCOME_TYPE']
           .value_counts().reset_index())
    grp.columns = ['Type', 'Count']

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=grp['Type'], values=grp['Count'], hole=0.42,
        marker=dict(colors=PIE, line=dict(color='#060d1a', width=2)),
        textinfo='label+percent', textposition='outside',
        textfont=dict(size=9, color='#a8c8e8'),
        rotation=50, pull=[0.04] + [0] * (len(grp) - 1),
    ))
    lo = L('Who Defaults? — By Employment Type', h=288)
    lo['showlegend'] = False
    # Annotation: dominant group
    dominant = grp.iloc[0]
    lo['annotations'] = [dict(
        text=f"<b>{dominant['Count']//1000}K</b><br>defaults",
        x=0.5, y=0.5, font_size=14, showarrow=False,
        font=dict(color='#c0d8f0', family='Oswald')
    )]
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True)

# ── Chart 1B: Bar — Defaults by Loan Term ─────────────────────
with r1b:
    term = (fdf[fdf['TARGET'] == 1]
            .groupby('TERM_BIN', observed=True)
            .size().reset_index(name='N'))
    term = term[term['N'] > 0]
    n = len(term)

    fig = go.Figure()
    bar_cols = [GRN if i < n - 1 else BLU for i in range(n)]
    fig.add_trace(go.Bar(
        x=term['TERM_BIN'].astype(str), y=term['N'],
        marker=dict(color=bar_cols, line=dict(color='#060d1a', width=1.5)),
        text=[f"{v/1000:.1f}K" for v in term['N']],
        textposition='outside', textfont=dict(color='#c0d8f0', size=10),
        width=0.52, name='',
    ))
    for name, col in [('Increase', GRN), ('Decrease', RED), ('Total', BLU)]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                  marker=dict(color=col, size=8, symbol='square'),
                                  name=name))
    lo = L('Default Loan by Loan Term Month', h=288, legend=True)
    lo['xaxis'] = dict(title=dict(text='loan_term_months',
                                    font=dict(size=10, color=TXT)),
                        showgrid=False, zeroline=False,
                        tickfont=dict(size=9, color=TXT))
    lo['yaxis'] = dict(title=dict(text='Total Default Loans',
                                    font=dict(size=10, color=TXT)),
                        showgrid=True, gridcolor=GRID,
                        zeroline=False, tickfont=dict(size=9, color=TXT))
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

# ── Chart 2A: H-Bar — Default Rate by LTI (Loan-to-Income) ───
with r2a:
    lti = (fdf.groupby('LTI_BIN', observed=True)['TARGET']
           .mean().reset_index())
    lti.columns = ['Bin', 'DR']

    # Highlight worst band in red
    bar_cols = [RED if row['DR'] == lti['DR'].max() else BLU
                for _, row in lti.iterrows()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lti['DR'], y=lti['Bin'].astype(str),
        orientation='h',
        marker=dict(color=bar_cols, line=dict(color='#060d1a', width=1.5)),
        text=[f" {v:.0%}" for v in lti['DR']],
        textposition='outside', textfont=dict(color='#c0d8f0', size=10),
        width=0.58,
    ))
    # Add vertical line at portfolio average
    avg_dr = fdf['TARGET'].mean()
    lo = L('Default Rate by Loan-to-Income (LTI)', h=262)
    lo['xaxis'] = dict(showgrid=False, zeroline=False, showticklabels=False,
                        title=dict(text='Loan Default Rate', font=dict(size=10, color=TXT)))
    lo['yaxis'] = dict(showgrid=True, gridcolor=GRID, zeroline=False,
                        tickfont=dict(size=9, color=TXT),
                        title=dict(text='Loan To Income', font=dict(size=10, color=TXT)))
    lo['shapes'] = [dict(type='line', x0=avg_dr, x1=avg_dr, y0=-0.5, y1=len(lti) - 0.5,
                          line=dict(color='#ff9900', dash='dash', width=1.5))]
    lo['annotations'] = [dict(x=avg_dr, y=len(lti) - 0.5, text='Avg', showarrow=False,
                               font=dict(color='#ff9900', size=9), xanchor='left')]
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

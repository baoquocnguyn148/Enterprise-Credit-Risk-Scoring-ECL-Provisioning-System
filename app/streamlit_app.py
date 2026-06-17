"""
streamlit_app.py — Nova Bank Credit Risk Dashboard
Pixel-perfect clone of the reference PowerBI/Tableau style.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings('ignore')

# ── MUST BE FIRST ────────────────────────────────────────────────
st.set_page_config(
    page_title="Nova Bank – Credit Risk Analyst",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════
# GLOBAL CSS  — Dark Navy matching the reference image
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & Root ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #070c18 !important;
    color: #c0d0e8;
}

/* ── Header Band ── */
.hdr {
    display: flex;
    align-items: center;
    background: #070c18;
    border-bottom: 1px solid #1a2e46;
    padding: 12px 28px;
    gap: 0;
    height: 90px;
}
.brand {
    min-width: 220px;
    border-right: 1px solid #1a2e46;
    padding-right: 28px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.brand-icon { font-size: 28px; margin-right: 8px; }
.brand-name {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    font-weight: 700;
    color: #a8c4e0;
    letter-spacing: 1px;
}
.brand-sub {
    font-size: 13px;
    color: #4a82c0;
    font-weight: 600;
    margin-top: 1px;
    letter-spacing: 0.5px;
}
.kpis {
    display: flex;
    flex: 1;
    justify-content: space-around;
    align-items: center;
}
.kpi {
    text-align: left;
    padding: 0 24px;
    border-right: 1px solid #1a2e46;
    flex: 1;
}
.kpi:last-child { border-right: none; }
.kpi-lbl {
    font-size: 11px;
    color: #5a7ea8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 2px;
    font-weight: 600;
}
.kpi-num {
    font-family: 'Playfair Display', serif;
    font-size: 38px;
    font-weight: 700;
    color: #c8daf0;
    line-height: 1;
}
.kpi-detail {
    font-size: 11px;
    color: #6a90b8;
    margin-top: 3px;
    line-height: 1.5;
}

/* ── Filter Band ── */
.filt-band {
    background: #070c18;
    padding: 10px 28px;
    border-bottom: 1px solid #1a2e46;
    display: flex;
    gap: 12px;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div > div {
    background: #0c1628 !important;
    border: 1px solid #1a2e46 !important;
    border-radius: 3px !important;
    color: #a8c0dc !important;
    font-size: 13px !important;
}
.stMultiSelect span[data-baseweb="tag"] {
    background: #1a3a5c !important;
    border: 1px solid #2a5080 !important;
    border-radius: 20px !important;
    color: #80b4d8 !important;
    font-size: 11px !important;
}

/* ── Chart rows padding ── */
.chart-section {
    padding: 6px 16px;
}

/* ── Plotly hover label ── */
.hoverlayer text { font-family: 'Inter' !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
DATA_DIR = r'd:\Risk\data'

@st.cache_data
def load_data():
    for f in ['results_ifrs9.parquet', 'results_df.parquet']:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            df = pd.read_parquet(p)
            break
    else:
        st.error("No data file found. Run modeling.py first.")
        st.stop()

    # Credit term months
    df['CREDIT_TERM_M'] = (df['AMT_CREDIT'] / (df['AMT_ANNUITY'].clip(lower=1)))
    df['TERM_BIN'] = pd.cut(df['CREDIT_TERM_M'],
                             bins=[0, 18, 30, 42, 58, 9999],
                             labels=['12', '24', '36', '48', '60+'])

    # LTI
    lti = df['CREDIT_INCOME_RATIO'] if 'CREDIT_INCOME_RATIO' in df.columns \
          else df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'].clip(lower=1))
    df['LTI_BIN'] = pd.cut(lti, bins=[-1, 1, 2, 3, 5, 999],
                            labels=['0-1x', '1-2x', '2-3x', '3-5x', '>5x'])

    # DTI
    dti = df['ANNUITY_INCOME_RATIO'] if 'ANNUITY_INCOME_RATIO' in df.columns \
          else df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'].clip(lower=1))
    df['DTI_BIN'] = pd.cut(dti, bins=[-1, .10, .20, .30, .50, 9],
                            labels=['0-10%', '10-20%', '20-30%', '30-50%', '>50%'])

    # Credit amount bins
    df['CREDIT_BIN'] = pd.cut(df['AMT_CREDIT'],
                               bins=[0, 100_000, 200_000, 300_000, 500_000, 9e9],
                               labels=['$0-$100k', '$100k-$200k', '$200k-$300k',
                                       '$300k-$500k', '$500k+'])
    return df

df = load_data()

# ═══════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════
BG      = '#070c18'
PANEL   = '#0a1220'
TXT     = '#7a9ec0'
BLU     = '#4a8cbd'   # bars
GLD     = '#c79a3c'   # line overlay
GRN     = '#2e8a58'   # waterfall increase
TEAL    = '#1e8a8a'
RED     = '#a83a2e'
PALETTE = [BLU, GLD, GRN, TEAL, '#6a5090', '#8a6050']


def chart_layout(title='', h=280, show_legend=False):
    """Consistent dark layout for every Plotly figure."""
    return dict(
        title=dict(
            text=title,
            font=dict(size=13, color='#7a9ec0', family='Inter'),
            x=0.0, pad=dict(l=4, t=4)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TXT, family='Inter', size=11),
        margin=dict(l=8, r=12, t=42, b=28),
        height=h,
        showlegend=show_legend,
        legend=dict(orientation='h', y=1.14, x=.5, xanchor='center',
                    font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(size=10, color=TXT), linecolor='#1a2e46'),
        yaxis=dict(showgrid=True, gridcolor='#111c2c', zeroline=False,
                   tickfont=dict(size=10, color=TXT)),
        hoverlabel=dict(bgcolor='#0f1e30', font_color='white', font_size=12),
    )


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
n_m  = (df['CODE_GENDER'] == 'M').sum()
n_f  = (df['CODE_GENDER'] == 'F').sum()
tot  = len(df)
loan = df['AMT_CREDIT'].sum()
ecl  = df['ECL'].sum() if 'ECL' in df.columns else 0
dr   = df['TARGET'].mean()

st.markdown(f"""
<div class="hdr">
  <div class="brand">
    <div>
      <span class="brand-icon">🏛</span>
      <span class="brand-name">NOVA BANK</span>
    </div>
    <div class="brand-sub">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Credit Risk Analyst</div>
  </div>
  <div class="kpis">
    <div class="kpi">
      <div class="kpi-lbl">Total Borrower</div>
      <div class="kpi-num">{tot/1000:.0f}K</div>
      <div class="kpi-detail">Male: {n_m/1000:.0f}K<br>Female: {n_f/1000:.0f}K</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">Total Loan Amount</div>
      <div class="kpi-num">${loan/1e9:.2f}B</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">Amount At Risk (ECL)</div>
      <div class="kpi-num">${ecl/1e6:,.2f}M</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">Loan Default Rate</div>
      <div class="kpi-num">{dr:.1%}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# FILTER RIBBON
# ═══════════════════════════════════════════════════════════════
fc = st.columns([1, 1, 2, 1], gap="small")
with fc[0]:
    gender_f = st.selectbox(" ", ["All", "Male", "Female"], label_visibility="collapsed", key="gf")
with fc[1]:
    contract_f = st.selectbox(" ", ["All", "Cash loans", "Revolving loans"], label_visibility="collapsed", key="cf")
with fc[2]:
    tier_f = st.multiselect(" ", ['Very Low', 'Low', 'Medium', 'High'],
                             default=['Very Low', 'Low', 'Medium', 'High'],
                             label_visibility="collapsed", key="tf")
with fc[3]:
    edu_opts = ["All"] + sorted(df['NAME_EDUCATION_TYPE'].dropna().unique())
    edu_f = st.selectbox(" ", edu_opts, label_visibility="collapsed", key="ef")

st.markdown("<div style='height:1px;background:#1a2e46;margin:0 16px'></div>", unsafe_allow_html=True)

# Apply filters
fdf = df.copy()
if gender_f == 'Male':    fdf = fdf[fdf['CODE_GENDER'] == 'M']
elif gender_f == 'Female': fdf = fdf[fdf['CODE_GENDER'] == 'F']
if contract_f != 'All':   fdf = fdf[fdf['NAME_CONTRACT_TYPE'] == contract_f]
if tier_f:                fdf = fdf[fdf['RISK_TIER'].isin(tier_f)]
if edu_f != 'All':        fdf = fdf[fdf['NAME_EDUCATION_TYPE'] == edu_f]

# ═══════════════════════════════════════════════════════════════
# ROW 1 — Three charts
# ═══════════════════════════════════════════════════════════════
st.markdown("<div style='padding:6px 14px 0'>", unsafe_allow_html=True)
r1c1, r1c2, r1c3 = st.columns([1.1, 1.3, 1.6], gap="small")

# ── 1A: Donut — Employee Type vs Default ──
with r1c1:
    grp = (fdf[fdf['TARGET'] == 1]['NAME_INCOME_TYPE']
           .value_counts()
           .reset_index())
    grp.columns = ['Type', 'Count']

    fig1 = go.Figure(go.Pie(
        labels=grp['Type'],
        values=grp['Count'],
        hole=0.42,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(colors=PALETTE, line=dict(color=BG, width=2)),
        textfont=dict(size=10, color='#a8c0dc'),
        pull=[0.03] * len(grp),
    ))
    fig1.update_layout(**chart_layout('How Employee Type Affects Default Loans', h=298))
    st.plotly_chart(fig1, use_container_width=True)

# ── 1B: Bar — Default Loans by Loan Term Month ──
with r1c2:
    term = (fdf[fdf['TARGET'] == 1]
            .groupby('TERM_BIN', observed=True)
            .size()
            .reset_index(name='Defaults'))

    colors_bar = [GRN if i < len(term) - 1 else BLU for i in range(len(term))]

    fig2 = go.Figure(go.Bar(
        x=term['TERM_BIN'].astype(str),
        y=term['Defaults'],
        marker=dict(color=colors_bar, line=dict(color=BG, width=1.5)),
        text=[f"{v/1000:.1f}K" for v in term['Defaults']],
        textposition='outside',
        textfont=dict(color='#c0d4ec', size=11),
        width=0.5,
    ))
    lo2 = chart_layout('Default Loan by Loan Term Month', h=298)
    lo2['xaxis']['title'] = dict(text='loan_term_months', font=dict(size=11))
    lo2['yaxis']['title'] = dict(text='Total Default Loans', font=dict(size=11))
    fig2.update_layout(**lo2)
    st.plotly_chart(fig2, use_container_width=True)

# ── 1C: Combo — Amount at Risk + Default Rate by Credit Bin ──
with r1c3:
    bin_g = (fdf.groupby('CREDIT_BIN', observed=True)
             .agg(Amt=('AMT_CREDIT', 'sum'), DR=('TARGET', 'mean'))
             .reset_index())

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])

    fig3.add_trace(go.Bar(
        x=bin_g['CREDIT_BIN'].astype(str),
        y=bin_g['Amt'],
        name='Amount at risk',
        marker=dict(color=BLU, line=dict(color=BG, width=1)),
        text=[f"${v/1e6:.0f}M" for v in bin_g['Amt']],
        textposition='inside',
        textfont=dict(color='white', size=10),
    ), secondary_y=False)

    fig3.add_trace(go.Scatter(
        x=bin_g['CREDIT_BIN'].astype(str),
        y=bin_g['DR'],
        name='Loan Default Rate',
        mode='lines+markers+text',
        line=dict(color=GLD, width=2.5),
        marker=dict(size=8, color=GLD, line=dict(color=BG, width=1.5)),
        text=[f"{v:.1%}" for v in bin_g['DR']],
        textposition='top center',
        textfont=dict(color=GLD, size=10),
    ), secondary_y=True)

    lo3 = chart_layout('Amount at Risk & Default Rate by Loan Amount Bin', h=298, show_legend=True)
    fig3.update_layout(**lo3)
    fig3.update_yaxes(showticklabels=False, showgrid=True, gridcolor='#111c2c',
                      zeroline=False, secondary_y=False)
    fig3.update_yaxes(tickformat='.0%', showgrid=False, zeroline=False,
                      tickfont=dict(color=TXT, size=10), secondary_y=True)
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# ROW 2 — Four charts
# ═══════════════════════════════════════════════════════════════
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
r2c1, r2c2, r2c3, r2c4 = st.columns([1, 1, 1, 1.1], gap="small")

# ── 2A: H-Bar — Default Rate by LTI ──
with r2c1:
    lti = (fdf.groupby('LTI_BIN', observed=True)['TARGET']
           .mean().reset_index())
    lti.columns = ['Bin', 'DR']

    fig4 = go.Figure(go.Bar(
        x=lti['DR'],
        y=lti['Bin'].astype(str),
        orientation='h',
        marker=dict(color=BLU, line=dict(color=BG, width=1.5)),
        text=[f" {v:.1%}" for v in lti['DR']],
        textposition='outside',
        textfont=dict(color='#c0d4ec', size=10),
    ))
    lo4 = chart_layout('Loan Default Rate by LTI', h=265)
    lo4['xaxis']['showticklabels'] = False
    lo4['xaxis']['title'] = dict(text='TARGET', font=dict(size=11))
    lo4['yaxis']['title'] = dict(text='Loan To Income', font=dict(size=11))
    fig4.update_layout(**lo4)
    st.plotly_chart(fig4, use_container_width=True)

# ── 2B: Full Pie — Default Intensity by Education ──
with r2c2:
    edu = (fdf.groupby('NAME_EDUCATION_TYPE')['TARGET']
           .mean().reset_index())
    edu.columns = ['Edu', 'DR']
    # Shorten labels for pie
    edu['Label'] = edu['Edu'].str.replace('Secondary / secondary special', 'Secondary / sec. sp.')

    fig5 = go.Figure(go.Pie(
        labels=edu['Label'],
        values=edu['DR'],
        hole=0,
        textinfo='label+percent',
        textposition='inside',
        marker=dict(colors=PALETTE, line=dict(color=BG, width=2)),
        textfont=dict(size=9),
        insidetextorientation='radial',
    ))
    fig5.update_layout(**chart_layout('Default Intensity by Education', h=265))
    st.plotly_chart(fig5, use_container_width=True)

# ── 2C: V-Bar — Default Rate by DTI ──
with r2c3:
    dti = (fdf.groupby('DTI_BIN', observed=True)['TARGET']
           .mean().reset_index())
    dti.columns = ['Bin', 'DR']

    fig6 = go.Figure(go.Bar(
        x=dti['Bin'].astype(str),
        y=dti['DR'],
        marker=dict(color=BLU, line=dict(color=BG, width=1.5)),
        text=[f"{v:.1%}" for v in dti['DR']],
        textposition='outside',
        textfont=dict(color='#c0d4ec', size=10),
        width=0.55,
    ))
    lo6 = chart_layout('Loan Default Rate by DTI', h=265)
    lo6['xaxis']['title'] = dict(text='Debt To Income', font=dict(size=11))
    lo6['yaxis']['showticklabels'] = False
    fig6.update_layout(**lo6)
    st.plotly_chart(fig6, use_container_width=True)

# ── 2D: H-Bar — Amount at Risk by Region Rating ──
with r2c4:
    reg = (fdf.groupby('REGION_RATING_CLIENT_W_CITY')
           .agg(Amt=('AMT_CREDIT', 'sum'))
           .reset_index()
           .sort_values('Amt'))
    reg['Label'] = 'Rating ' + reg['REGION_RATING_CLIENT_W_CITY'].astype(str)
    bar_colors = [GRN, GLD, BLU][:len(reg)]

    fig7 = go.Figure(go.Bar(
        x=reg['Amt'],
        y=reg['Label'],
        orientation='h',
        marker=dict(color=bar_colors, line=dict(color=BG, width=1.5)),
        text=[f"${v/1e6:.0f}M" for v in reg['Amt']],
        textposition='inside',
        textfont=dict(color='white', size=10),
    ))
    lo7 = chart_layout('Amount at Risk by Region Rating', h=265)
    lo7['xaxis']['showticklabels'] = False
    lo7['xaxis']['title'] = dict(text='Amt', font=dict(size=11))
    lo7['yaxis']['title'] = dict(text='Region Rating', font=dict(size=11))
    fig7.update_layout(**lo7)
    st.plotly_chart(fig7, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

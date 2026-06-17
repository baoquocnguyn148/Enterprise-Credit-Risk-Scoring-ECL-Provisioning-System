"""
streamlit_app.py — Nova Bank Credit Risk Dashboard
Exact replica of the PowerBI reference image design.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Nova Bank – Credit Risk Analyst",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════
# GLOBAL CSS — exact match of reference image
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap');

/* ── Reset ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
    background-color: #080f1e !important;
    color: #c8d8f0;
}
.stApp { background-color: #080f1e !important; }

/* ── Header row ── */
.hdr {
    display: flex;
    align-items: stretch;
    background: #060c18;
    border: 1px solid #1e4272;
    border-radius: 0;
    margin: 6px 6px 0 6px;
    height: 80px;
}
.brand-box {
    background: #060c18;
    border-right: 2px solid #1e4272;
    padding: 8px 18px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-width: 200px;
    max-width: 200px;
}
.brand-icon { font-size: 24px; }
.brand-name {
    font-family: 'Oswald', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #b8d0ec;
    letter-spacing: 1px;
    line-height: 1.1;
}
.brand-sub {
    font-size: 12px;
    color: #3a80c8;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.kpis {
    display: flex;
    flex: 1;
    align-items: stretch;
}
.kpi-box {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 6px 16px;
    border-right: 1px solid #1a3a62;
    border-left: 1px solid #1a3a62;
    position: relative;
}
.kpi-box:first-child { border-left: none; }
.kpi-lbl {
    font-size: 11px;
    color: #6898cc;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
    margin-bottom: 2px;
}
.kpi-num {
    font-family: 'Oswald', sans-serif;
    font-size: 36px;
    font-weight: 700;
    color: #ddeeff;
    line-height: 1;
}
.kpi-sub {
    font-size: 11px;
    color: #5888b8;
    margin-top: 2px;
    line-height: 1.5;
}

/* ── Filter row ── */
.filt-row {
    display: flex;
    align-items: center;
    background: #060c18;
    border: 1px solid #1e4272;
    border-top: 0;
    margin: 0 6px 4px 6px;
    padding: 6px 10px;
    gap: 8px;
}

/* ── Streamlit select / multiselect overrides ── */
.stSelectbox label, .stMultiSelect label { display: none !important; }
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div {
    background: #0c1828 !important;
    border: 1px solid #1e4272 !important;
    color: #aacce8 !important;
    font-size: 12px !important;
    min-height: 32px !important;
    border-radius: 2px !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background: #1a3860 !important;
    border-radius: 2px !important;
    color: #88bbdd !important;
    font-size: 10px !important;
}
div[data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stHorizontalBlock"] { gap: 4px !important; }

/* small chart area padding */
.chart-wrap { padding: 0 6px 4px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# DATA
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
        st.error("No data found — run modeling.py first."); st.stop()

    df['TERM_M']  = df['AMT_CREDIT'] / df['AMT_ANNUITY'].clip(lower=1)
    df['TERM_BIN'] = pd.cut(df['TERM_M'], [-1,18,30,42,56,9999],
                             labels=['12','24','36','48','60+'])
    lti = df.get('CREDIT_INCOME_RATIO',
                 df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].clip(lower=1))
    df['LTI_BIN'] = pd.cut(lti, [-1,1,2,3,5,999],
                            labels=['0–1×','1–2×','2–3×','3–5×','>5×'])
    dti = df.get('ANNUITY_INCOME_RATIO',
                 df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].clip(lower=1))
    df['DTI_BIN'] = pd.cut(dti, [-1,.10,.20,.30,.50,9],
                            labels=['0–10%','10–20%','20–30%','30–50%','>50%'])
    df['CRED_BIN'] = pd.cut(df['AMT_CREDIT'],
                             [0,100e3,200e3,300e3,500e3,9e9],
                             labels=['$0–100k','$100–200k','$200–300k',
                                     '$300–500k','$500k+'])
    return df

df = load()

# ── COLOURS ─────────────────────────────────────────────────────
BG       = '#080f1e'
PANEL    = '#060c18'
BORDER   = '#1e4272'
TXT      = '#6898cc'
BAR_BLUE = '#4a7ab8'   # main steel-blue bars (matching image)
BAR_GRN  = '#1a8a3a'   # waterfall increase
BAR_RED  = '#c83232'   # waterfall decrease
LINE_GLD = '#c89a1a'   # gold line overlay
PIE_COLS = ['#4a7ab8','#c89a1a','#1a8a7a','#8a6040','#6a5090','#3a8a50','#8a3050']

def lay(title='', h=280, legend=False):
    """Shared chart layout — all kwargs are fully specified here, no conflicts."""
    return dict(
        title=dict(text=title, font=dict(size=12,color='#a0c0e0',family='Open Sans'),
                   x=0, pad=dict(l=4,t=2)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor ='rgba(0,0,0,0)',
        font=dict(color=TXT, family='Open Sans', size=10),
        margin=dict(l=6, r=10, t=40, b=24),
        height=h,
        showlegend=legend,
        legend=dict(orientation='h', y=1.12, x=.5, xanchor='center',
                    font=dict(size=9), bgcolor='rgba(0,0,0,0)',
                    itemsizing='constant'),
        xaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickfont=dict(size=9,color=TXT)),
        yaxis=dict(showgrid=True, gridcolor='#0e1e34', zeroline=False,
                   tickfont=dict(size=9,color=TXT)),
        hoverlabel=dict(bgcolor='#0c1e38',font_color='#e0f0ff',font_size=11),
    )

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
nm = (df['CODE_GENDER']=='M').sum()
nf = (df['CODE_GENDER']=='F').sum()
total_loan = df['AMT_CREDIT'].sum()
ecl  = df['ECL'].sum() if 'ECL' in df.columns else 0
dr   = df['TARGET'].mean()

st.markdown(f"""
<div class="hdr">
  <div class="brand-box">
    <div class="brand-icon">🏛</div>
    <div class="brand-name">NOVA BANK</div>
    <div class="brand-sub">Credit Risk Analyst</div>
  </div>
  <div class="kpis">
    <div class="kpi-box">
      <div class="kpi-lbl">Total Borrower</div>
      <div class="kpi-num">{len(df)//1000}K</div>
      <div class="kpi-sub">Male:&nbsp;&nbsp;{nm//1000}K<br>Female:{nf//1000}K</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-lbl">Total Loan Amount</div>
      <div class="kpi-num">${total_loan/1e9:.2f}B</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-lbl">Amount At Risk (ECL)</div>
      <div class="kpi-num">${ecl/1e6:,.0f}M</div>
    </div>
    <div class="kpi-box" style="border-right:none">
      <div class="kpi-lbl">Loan Default Rate</div>
      <div class="kpi-num">{dr:.1%}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# FILTER ROW
# ═══════════════════════════════════════════════════════════════════
fc = st.columns([1,1,2,1], gap="small")
with fc[0]:
    gf = st.selectbox("g", ["All","Male","Female"], label_visibility="collapsed")
with fc[1]:
    cf = st.selectbox("c", ["All","Cash loans","Revolving loans"], label_visibility="collapsed")
with fc[2]:
    tf = st.multiselect("t", ['Very Low','Low','Medium','High'],
                         default=['Very Low','Low','Medium','High'],
                         label_visibility="collapsed")
with fc[3]:
    edu_list = ["All"] + sorted(df['NAME_EDUCATION_TYPE'].dropna().unique())
    ef = st.selectbox("e", edu_list, label_visibility="collapsed")

st.markdown("<div style='height:1px;background:#1a3860;margin:0 6px 4px'></div>",
            unsafe_allow_html=True)

# Apply filters
fdf = df.copy()
if gf == 'Male':   fdf = fdf[fdf['CODE_GENDER']=='M']
elif gf == 'Female': fdf = fdf[fdf['CODE_GENDER']=='F']
if cf != 'All':    fdf = fdf[fdf['NAME_CONTRACT_TYPE']==cf]
if tf:             fdf = fdf[fdf['RISK_TIER'].isin(tf)]
if ef != 'All':    fdf = fdf[fdf['NAME_EDUCATION_TYPE']==ef]

# ═══════════════════════════════════════════════════════════════════
# ROW 1 — three charts (same proportions as image)
# ═══════════════════════════════════════════════════════════════════
c1, c2, c3 = st.columns([1.1, 1.3, 1.6], gap="small")

# ── Chart 1: Donut — Employee type vs default ─────────────────
with c1:
    grp = (fdf[fdf['TARGET']==1]['NAME_INCOME_TYPE']
           .value_counts().reset_index())
    grp.columns = ['Type','Count']

    fig1 = go.Figure(go.Pie(
        labels=grp['Type'], values=grp['Count'],
        hole=0.42,
        marker=dict(colors=PIE_COLS, line=dict(color='#08111e', width=2)),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=9, color='#b0c8e8'),
        rotation=60,
        pull=[0.04]*len(grp),
    ))
    lo1 = lay('How Employee Type Affecting Default Loans', h=295)
    lo1['showlegend'] = False
    fig1.update_layout(**lo1)
    st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: Bar (waterfall style) — Defaults by Term Month ──
with c2:
    term = (fdf[fdf['TARGET']==1]
            .groupby('TERM_BIN', observed=True)
            .size().reset_index(name='N'))
    term = term[term['N']>0]

    n = len(term)
    bar_colors = [BAR_GRN]*(n-1) + [BAR_BLUE]

    fig2 = go.Figure(go.Bar(
        x=term['TERM_BIN'].astype(str),
        y=term['N'],
        marker=dict(color=bar_colors, line=dict(color='#08111e', width=1.5)),
        text=[f"{v/1000:.1f}K" for v in term['N']],
        textposition='outside',
        textfont=dict(color='#c0d8f0', size=10),
        width=0.52,
    ))
    lo2 = lay('Default Loan by Loan Term Month', h=295)
    lo2['yaxis']['title'] = dict(text='Total Default Loans',
                                  font=dict(size=10, color=TXT))
    lo2['xaxis']['title'] = dict(text='loan_term_months',
                                  font=dict(size=10, color=TXT))
    # Legend-style annotation
    lo2['showlegend'] = True
    lo2['legend'] = dict(orientation='h', y=1.10, x=0.5, xanchor='center',
                          font=dict(size=9), bgcolor='rgba(0,0,0,0)')
    fig2.update_layout(**lo2)
    # Add dummy traces for legend to match reference image
    fig2.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                               marker=dict(color=BAR_GRN, size=8, symbol='square'),
                               name='Increase'))
    fig2.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                               marker=dict(color=BAR_RED, size=8, symbol='square'),
                               name='Decrease'))
    fig2.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                               marker=dict(color=BAR_BLUE, size=8, symbol='square'),
                               name='Total'))
    st.plotly_chart(fig2, use_container_width=True)

# ── Chart 3: Combo — Amount at Risk + Default Rate by Credit Bin ──
with c3:
    bg = (fdf.groupby('CRED_BIN', observed=True)
          .agg(Amt=('AMT_CREDIT','sum'), DR=('TARGET','mean'))
          .reset_index())

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])

    fig3.add_trace(go.Bar(
        x=bg['CRED_BIN'].astype(str),
        y=bg['Amt'],
        name='Amount at risk',
        marker=dict(color=BAR_BLUE, line=dict(color='#08111e', width=1)),
        text=[f"${v/1e6:.0f}M" for v in bg['Amt']],
        textposition='inside',
        textfont=dict(color='white', size=9),
        width=0.55,
    ), secondary_y=False)

    fig3.add_trace(go.Scatter(
        x=bg['CRED_BIN'].astype(str),
        y=bg['DR'],
        name='Loan Default Rate',
        mode='lines+markers+text',
        line=dict(color=LINE_GLD, width=2.5),
        marker=dict(color=LINE_GLD, size=7,
                    line=dict(color='#08111e', width=1.5)),
        text=[f"{v:.0%}" for v in bg['DR']],
        textposition='top center',
        textfont=dict(color=LINE_GLD, size=9),
    ), secondary_y=True)

    lo3 = lay('Amount at risk and Loan Default Rate by Loan Am…', h=295, legend=True)
    lo3['legend'] = dict(orientation='h', y=1.12, x=.5, xanchor='center',
                          font=dict(size=9), bgcolor='rgba(0,0,0,0)',
                          marker=dict(size=8))
    fig3.update_layout(**lo3)
    fig3.update_yaxes(showgrid=True, gridcolor='#0e1e34',
                      zeroline=False, showticklabels=False,
                      secondary_y=False)
    fig3.update_yaxes(tickformat='.0%', showgrid=False,
                      zeroline=False, tickfont=dict(size=9, color=TXT),
                      secondary_y=True)
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# ROW 2 — four charts
# ═══════════════════════════════════════════════════════════════════
st.markdown("<div style='height:3px'></div>", unsafe_allow_html=True)
d1, d2, d3, d4 = st.columns([1, 1, 1, 1.1], gap="small")

# ── Chart 4: H-Bar — Default Rate by LTI ─────────────────────
with d1:
    lti = (fdf.groupby('LTI_BIN', observed=True)['TARGET']
           .mean().reset_index())
    lti.columns = ['Bin','DR']

    fig4 = go.Figure(go.Bar(
        x=lti['DR'],
        y=lti['Bin'].astype(str),
        orientation='h',
        marker=dict(color=BAR_BLUE,
                    line=dict(color='#08111e', width=1.5)),
        text=[f" {v:.0%}" for v in lti['DR']],
        textposition='outside',
        textfont=dict(color='#c0d8f0', size=10, family='Open Sans'),
        width=0.6,
    ))
    lo4 = lay('Loan Default Rate by LTI', h=262)
    lo4['xaxis']['showticklabels'] = False
    lo4['xaxis']['title'] = dict(text='Loan Default Rate',
                                  font=dict(size=10, color=TXT))
    lo4['yaxis']['title'] = dict(text='Loan To Income',
                                  font=dict(size=10, color=TXT))
    fig4.update_layout(**lo4)
    st.plotly_chart(fig4, use_container_width=True)

# ── Chart 5: Full Pie — Default by Education ─────────────────
with d2:
    edu = (fdf.groupby('NAME_EDUCATION_TYPE')['TARGET']
           .mean().reset_index())
    edu.columns = ['Label','DR']
    # Shorter labels
    edu['Label'] = (edu['Label']
                    .str.replace('Secondary / secondary special','Secondary\nspecial')
                    .str.replace('Higher education','Higher edu')
                    .str.replace('Incomplete higher','Incomplete\nhigher'))

    fig5 = go.Figure(go.Pie(
        labels=edu['Label'], values=edu['DR'],
        hole=0,
        marker=dict(colors=PIE_COLS, line=dict(color='#08111e', width=2)),
        textinfo='label+percent',
        textposition='inside',
        textfont=dict(size=8.5, color='white'),
        insidetextorientation='radial',
    ))
    lo5 = lay('Loan Default Rate by Loan Purpose', h=262)
    lo5['showlegend'] = False
    fig5.update_layout(**lo5)
    st.plotly_chart(fig5, use_container_width=True)

# ── Chart 6: V-Bar — Default Rate by DTI ─────────────────────
with d3:
    dti = (fdf.groupby('DTI_BIN', observed=True)['TARGET']
           .mean().reset_index())
    dti.columns = ['Bin','DR']

    fig6 = go.Figure(go.Bar(
        x=dti['Bin'].astype(str),
        y=dti['DR'],
        marker=dict(color=BAR_BLUE, line=dict(color='#08111e', width=1.5)),
        text=[f"{v:.0%}" for v in dti['DR']],
        textposition='outside',
        textfont=dict(color='#c0d8f0', size=10),
        width=0.55,
    ))
    lo6 = lay('Loan Default Rate by DTI', h=262)
    lo6['xaxis']['title'] = dict(text='Debt To Income',
                                  font=dict(size=10, color=TXT))
    lo6['yaxis']['title'] = dict(text='Loan Default Rate',
                                  font=dict(size=10, color=TXT))
    lo6['yaxis']['showticklabels'] = False
    fig6.update_layout(**lo6)
    st.plotly_chart(fig6, use_container_width=True)

# ── Chart 7: H-Bar — Amount at Risk by Region Rating ─────────
with d4:
    reg = (fdf.groupby('REGION_RATING_CLIENT_W_CITY')
           .agg(Amt=('AMT_CREDIT','sum'))
           .reset_index()
           .sort_values('Amt'))
    reg['Label'] = 'Rating ' + reg['REGION_RATING_CLIENT_W_CITY'].astype(str)
    seg_colors = [BAR_GRN, LINE_GLD, BAR_BLUE][:len(reg)]

    fig7 = go.Figure(go.Bar(
        x=reg['Amt'],
        y=reg['Label'],
        orientation='h',
        marker=dict(color=seg_colors, line=dict(color='#08111e', width=1.5)),
        text=[f"${v/1e6:.0f}M" for v in reg['Amt']],
        textposition='inside',
        textfont=dict(color='white', size=10),
        width=0.55,
    ))
    lo7 = lay('Amount at risk by state and c…', h=262)
    lo7['xaxis']['showticklabels'] = False
    lo7['xaxis']['title'] = dict(text='Amt', font=dict(size=10, color=TXT))
    lo7['yaxis']['title'] = dict(text='', font=dict(size=10, color=TXT))
    fig7.update_layout(**lo7)
    st.plotly_chart(fig7, use_container_width=True)

"""
streamlit_app.py — Nova Bank Credit Risk Dashboard
Clean, error-free implementation matching the reference dark-navy style.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Nova Bank – Credit Risk",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────
# CSS — dark navy exact match
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 4px 8px !important; max-width: 100% !important; }
html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
    background-color: #060d1a !important;
    color: #b0c8e4;
}
.stApp { background-color: #060d1a !important; }

/* Give every plotly chart a visible border panel */
[data-testid="stPlotlyChart"] {
    border: 1px solid #2060a0 !important;
    border-radius: 2px !important;
    padding: 2px !important;
    background: #080f1e !important;
}
/* Column gaps have border-right to separate panels */
[data-testid="column"] > div:first-child {
    border-right: 0px solid transparent;
}

/* KPI cards */
.kcard {
    background: #070e1c;
    border: 1px solid #2060a0;
    border-left: 3px solid #2878c0;
    padding: 8px 16px;
    border-radius: 0;
    height: 78px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin-bottom: 0;
}
.kcard-lbl {
    font-size: 10px;
    color: #90b8d8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
    margin-bottom: 2px;
}
.kcard-val {
    font-family: 'Oswald', sans-serif;
    font-size: 38px;
    font-weight: 700;
    color: #e0f0ff;
    line-height: 1;
}
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
    border-left: 3px solid #2878c0;
    padding: 8px 16px;
    border-radius: 0;
    height: 78px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.brand-title {
    font-family: 'Oswald', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #a8c8e8;
    letter-spacing: 2px;
}
.brand-sub {
    font-size: 12px;
    color: #3a78c0;
    font-weight: 600;
    margin-top: 2px;
    letter-spacing: 0.5px;
}

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
.stMultiSelect [data-baseweb="tag"] {
    background: #1a4070 !important;
    border-radius: 2px !important;
    color: #80b8e0 !important;
    font-size: 10px !important;
    border: 1px solid #2a60a0 !important;
}
div[data-testid="stMultiSelect"] > div > div {
    background: #09152a !important;
    border: 1px solid #2060a0 !important;
    border-radius: 2px !important;
    font-size: 12px !important;
}
[data-testid="stHorizontalBlock"] { gap: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────
DATA_DIR = r'd:\Risk\data'

@st.cache_data
def load():
    for fn in ['results_ifrs9.parquet', 'results_df.parquet']:
        p = os.path.join(DATA_DIR, fn)
        if os.path.exists(p):
            df = pd.read_parquet(p)
            break
    else:
        st.error("No data. Run modeling.py first."); st.stop()

    df['TERM_M'] = (df['AMT_CREDIT'] / df['AMT_ANNUITY'].clip(lower=1)).round(0)
    df['TERM_BIN'] = pd.cut(df['TERM_M'], [-1,18,30,42,56,9999],
                             labels=['12','24','36','48','60+'])

    lti = df.get('CREDIT_INCOME_RATIO', df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].clip(lower=1))
    df['LTI_BIN'] = pd.cut(lti, [-1,1,2,3,5,999],
                            labels=['0–1×','1–2×','2–3×','3–5×','>5×'])

    dti = df.get('ANNUITY_INCOME_RATIO', df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].clip(lower=1))
    df['DTI_BIN'] = pd.cut(dti, [-1,.10,.20,.30,.50,9],
                            labels=['0–10%','10–20%','20–30%','30–50%','>50%'])

    df['CRED_BIN'] = pd.cut(df['AMT_CREDIT'], [0,100e3,200e3,300e3,500e3,9e9],
                             labels=['$0–100k','$100–200k','$200–300k','$300–500k','$500k+'])
    return df

df = load()

# ─────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────
BLU  = '#4a7ab8'
GRN  = '#2a8a40'
GLD  = '#c89020'
RED  = '#b83030'
TEAL = '#1a8a78'
BRWN = '#8a6040'
PIE  = [BLU, GLD, GRN, TEAL, BRWN, '#6a5090', '#3a8858']
TXT  = '#6898c8'
GRID = '#0e1e34'
BG   = 'rgba(0,0,0,0)'


def mk_layout(title='', h=280, show_legend=False):
    """Clean, conflict-free Plotly layout."""
    return dict(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=11, color='#8ab8d8', family='Open Sans'),
                   x=0, xref='paper', pad=dict(l=2, t=2)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TXT, family='Open Sans', size=10),
        margin=dict(l=4, r=10, t=38, b=22),
        height=h,
        showlegend=show_legend,
        hoverlabel=dict(bgcolor='#0c1e38', font_color='#e0f0ff', font_size=11),
    )


# ─────────────────────────────────────────────────────────────────
# HEADER ROW
# ─────────────────────────────────────────────────────────────────
nm = (df['CODE_GENDER'] == 'M').sum()
nf = (df['CODE_GENDER'] == 'F').sum()
loan_total = df['AMT_CREDIT'].sum()
ecl_total  = df['ECL'].sum() if 'ECL' in df.columns else 0
dr_total   = df['TARGET'].mean()

h0, h1, h2, h3, h4 = st.columns([1.2, 1, 1, 1, 0.8], gap="small")

with h0:
    st.markdown(f"""
    <div class="brand-card">
        <div class="brand-title">🏛 NOVA BANK</div>
        <div class="brand-sub">&nbsp;Credit Risk Analyst</div>
    </div>""", unsafe_allow_html=True)

with h1:
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Total Borrower</div>
        <div class="kcard-val">{len(df)//1000}K</div>
        <div class="kcard-sub">Male: {nm//1000}K &nbsp; Female: {nf//1000}K</div>
    </div>""", unsafe_allow_html=True)

with h2:
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Total Loan Amount</div>
        <div class="kcard-val">${loan_total/1e9:.2f}B</div>
    </div>""", unsafe_allow_html=True)

with h3:
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Amount At Risk (ECL)</div>
        <div class="kcard-val">${ecl_total/1e6:,.0f}M</div>
    </div>""", unsafe_allow_html=True)

with h4:
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Loan Default Rate</div>
        <div class="kcard-val">{dr_total:.1%}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# FILTER ROW
# ─────────────────────────────────────────────────────────────────
f0, f1, f2, f3 = st.columns([1, 1, 2, 1], gap="small")
with f0:
    gf = st.selectbox("_g", ["All","Male","Female"], label_visibility="collapsed")
with f1:
    cf = st.selectbox("_c", ["All","Cash loans","Revolving loans"], label_visibility="collapsed")
with f2:
    tf = st.multiselect("_t", ['Very Low','Low','Medium','High'],
                         default=['Very Low','Low','Medium','High'],
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

# ─────────────────────────────────────────────────────────────────
# ROW 1 ── three charts
# ─────────────────────────────────────────────────────────────────
r1a, r1b, r1c = st.columns([1.1, 1.3, 1.6], gap="small")

# Chart A: Donut — Income Type vs Defaults
with r1a:
    grp = (fdf[fdf['TARGET'] == 1]['NAME_INCOME_TYPE']
           .value_counts().reset_index())
    grp.columns = ['Type', 'Count']

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=grp['Type'], values=grp['Count'],
        hole=0.40,
        marker=dict(colors=PIE, line=dict(color='#07101e', width=2)),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=9, color='#a8c8e8'),
        rotation=45,
        pull=[0.03] * len(grp),
    ))
    lo = mk_layout('How Employee Type Affecting Default Loans', h=290)
    lo['showlegend'] = False
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True)

# Chart B: Bar — Defaults by Loan Term
with r1b:
    term = (fdf[fdf['TARGET'] == 1]
            .groupby('TERM_BIN', observed=True)
            .size().reset_index(name='N'))
    term = term[term['N'] > 0]

    n = len(term)
    colors = [GRN] * max(0, n - 1) + [BLU]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=term['TERM_BIN'].astype(str), y=term['N'],
        marker=dict(color=colors, line=dict(color='#07101e', width=1.5)),
        text=[f"{v/1000:.1f}K" for v in term['N']],
        textposition='outside',
        textfont=dict(color='#c0d8f0', size=10),
        width=0.50,
        name='',
    ))
    # Legend markers (Increase / Decrease / Total)
    for name, col in [('Increase', GRN), ('Decrease', RED), ('Total', BLU)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color=col, size=9, symbol='square'),
            name=name,
        ))

    lo = mk_layout('Default Loan by Loan Term Month', h=290, show_legend=True)
    lo['legend'] = dict(
        orientation='h', y=1.10, x=0.5, xanchor='center',
        font=dict(size=9), bgcolor='rgba(0,0,0,0)',
        traceorder='normal',
    )
    lo['xaxis'] = dict(title=dict(text='loan_term_months', font=dict(size=10, color=TXT)),
                       showgrid=False, zeroline=False,
                       tickfont=dict(size=9, color=TXT))
    lo['yaxis'] = dict(title=dict(text='Total Default Loans', font=dict(size=10, color=TXT)),
                       showgrid=True, gridcolor=GRID, zeroline=False,
                       tickfont=dict(size=9, color=TXT))
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True)

# Chart C: Combo — Amount at Risk + Default Rate
with r1c:
    bg = (fdf.groupby('CRED_BIN', observed=True)
          .agg(Amt=('AMT_CREDIT', 'sum'), DR=('TARGET', 'mean'))
          .reset_index())

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=bg['CRED_BIN'].astype(str), y=bg['Amt'],
        name='Amount at risk',
        marker=dict(color=BLU, line=dict(color='#07101e', width=1)),
        text=[f"${v/1e6:.0f}M" for v in bg['Amt']],
        textposition='inside',
        textfont=dict(color='white', size=9),
        width=0.55,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=bg['CRED_BIN'].astype(str), y=bg['DR'],
        name='Loan Default Rate',
        mode='lines+markers+text',
        line=dict(color=GLD, width=2.5),
        marker=dict(color=GLD, size=7, line=dict(color='#07101e', width=1.5)),
        text=[f"{v:.0%}" for v in bg['DR']],
        textposition='top center',
        textfont=dict(color=GLD, size=9),
    ), secondary_y=True)

    lo = mk_layout('Amount at risk and Loan Default Rate by Loan Amount', h=290, show_legend=True)
    lo['legend'] = dict(orientation='h', y=1.10, x=0.5, xanchor='center',
                         font=dict(size=9), bgcolor='rgba(0,0,0,0)')
    lo['xaxis'] = dict(showgrid=False, zeroline=False,
                        tickfont=dict(size=9, color=TXT))
    lo['yaxis'] = dict(showgrid=True, gridcolor=GRID, zeroline=False,
                        showticklabels=False,
                        tickfont=dict(size=9, color=TXT))
    lo['yaxis2'] = dict(showgrid=False, zeroline=False,
                         tickformat='.0%',
                         tickfont=dict(size=9, color=TXT))
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# ROW 2 ── four charts
# ─────────────────────────────────────────────────────────────────
st.markdown("<div style='height:3px'></div>", unsafe_allow_html=True)
r2a, r2b, r2c, r2d = st.columns([1, 1, 1, 1.1], gap="small")

# Chart D: H-Bar — Default Rate by LTI
with r2a:
    lti = (fdf.groupby('LTI_BIN', observed=True)['TARGET']
           .mean().reset_index())
    lti.columns = ['Bin', 'DR']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lti['DR'], y=lti['Bin'].astype(str),
        orientation='h',
        marker=dict(color=BLU, line=dict(color='#07101e', width=1.5)),
        text=[f" {v:.0%}" for v in lti['DR']],
        textposition='outside',
        textfont=dict(color='#c0d8f0', size=10),
        width=0.58,
    ))
    lo = mk_layout('Loan Default Rate by LTI', h=262)
    lo['xaxis'] = dict(showgrid=False, zeroline=False, showticklabels=False,
                        title=dict(text='Loan Default Rate', font=dict(size=10, color=TXT)))
    lo['yaxis'] = dict(showgrid=True, gridcolor=GRID, zeroline=False,
                        tickfont=dict(size=9, color=TXT),
                        title=dict(text='Loan To Income', font=dict(size=10, color=TXT)))
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True)

# Chart E: Full Pie — Default by Loan Purpose (Education proxy)
with r2b:
    edu = (fdf.groupby('NAME_EDUCATION_TYPE')['TARGET']
           .mean().reset_index())
    edu.columns = ['Edu', 'DR']
    edu['Short'] = (edu['Edu']
                    .str.replace('Secondary / secondary special', 'SECONDARY')
                    .str.replace('Higher education', 'HIGHER EDU')
                    .str.replace('Incomplete higher', 'INCOMPLETE')
                    .str.replace('Lower secondary', 'LOWER SEC')
                    .str.replace('Academic degree', 'ACADEMIC'))

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=edu['Short'], values=edu['DR'],
        hole=0,
        marker=dict(colors=PIE, line=dict(color='#07101e', width=2)),
        textinfo='label+percent',
        textposition='inside',
        textfont=dict(size=8.5, color='white'),
        insidetextorientation='radial',
    ))
    lo = mk_layout('Loan Default Rate by Loan Purpose', h=262)
    lo['showlegend'] = False
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True)

# Chart F: V-Bar — Default Rate by DTI
with r2c:
    dti = (fdf.groupby('DTI_BIN', observed=True)['TARGET']
           .mean().reset_index())
    dti.columns = ['Bin', 'DR']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dti['Bin'].astype(str), y=dti['DR'],
        marker=dict(color=BLU, line=dict(color='#07101e', width=1.5)),
        text=[f"{v:.0%}" for v in dti['DR']],
        textposition='outside',
        textfont=dict(color='#c0d8f0', size=10),
        width=0.55,
    ))
    lo = mk_layout('Loan Default Rate by DTI', h=262)
    lo['xaxis'] = dict(showgrid=False, zeroline=False,
                        tickfont=dict(size=9, color=TXT),
                        title=dict(text='Debt To Income', font=dict(size=10, color=TXT)))
    lo['yaxis'] = dict(showgrid=True, gridcolor=GRID, zeroline=False,
                        showticklabels=False,
                        title=dict(text='Loan Default Rate', font=dict(size=10, color=TXT)))
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True)

# Chart G: H-Bar — Amount at Risk by Region Rating
with r2d:
    reg = (fdf.groupby('REGION_RATING_CLIENT_W_CITY')
           .agg(Amt=('AMT_CREDIT', 'sum'))
           .reset_index()
           .sort_values('Amt', ascending=True))
    reg['Label'] = 'Rating ' + reg['REGION_RATING_CLIENT_W_CITY'].astype(str)
    seg_colors = [GRN, GLD, BLU][:len(reg)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=reg['Amt'], y=reg['Label'],
        orientation='h',
        marker=dict(color=seg_colors, line=dict(color='#07101e', width=1.5)),
        text=[f"${v/1e6:.0f}M" for v in reg['Amt']],
        textposition='inside',
        textfont=dict(color='white', size=10),
        width=0.55,
    ))
    lo = mk_layout('Amount at risk by Region Rating', h=262)
    lo['xaxis'] = dict(showgrid=False, zeroline=False, showticklabels=False,
                        title=dict(text='Amt', font=dict(size=10, color=TXT)))
    lo['yaxis'] = dict(showgrid=True, gridcolor=GRID, zeroline=False,
                        tickfont=dict(size=9, color=TXT))
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True)

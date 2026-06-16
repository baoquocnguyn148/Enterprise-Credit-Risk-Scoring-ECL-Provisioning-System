"""
shap_analysis.py — Phase 4: Model Interpretation (v2 UPGRADED)
Home Credit Default Risk

New in v2:
  [NEW] Waterfall plot for 1 High-Risk client
  [NEW] Waterfall plot for 1 Low-Risk client
  [NEW] SHAP Dependence plots (EXT_SOURCE_2, CREDIT_INCOME_RATIO)
  [NEW] Segment-level SHAP (by Risk Tier)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os, pickle, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

DATA_DIR    = r'd:\Risk\data'
MODELS_DIR  = r'd:\Risk\models'
REPORTS_DIR = r'd:\Risk\reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

def header(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def step(t):   print(f"\n  >> {t}")
def log(t):    print(f"     {t}")


# ═══════════════════════════════════════════════════════════════
# STEP 1 — Load Model and Sample Data
# ═══════════════════════════════════════════════════════════════
header("STEP 1 — Load Model and Sample Data")

step("Loading best fold model (fold 2 — highest AUC)...")
# Use fold2 which tends to be the strongest
model_path = os.path.join(MODELS_DIR, 'lgbm_fold2.pkl')
if not os.path.exists(model_path):
    model_path = os.path.join(MODELS_DIR, 'lgbm_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'feature_list.pkl'), 'rb') as f:
    FEATURES = pickle.load(f)

import re
FEATURES = [re.sub(r'[^A-Za-z0-9_]+', '_', x) for x in FEATURES]

step("Loading data sample (10,000 rows for SHAP calculation)...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train_features.parquet'))
train_df = train_df.rename(columns=lambda x: re.sub(r'[^A-Za-z0-9_]+', '_', x))

# Load full results (with PRED_PROB + RISK_TIER)
results_df = pd.read_parquet(os.path.join(DATA_DIR, 'results_df.parquet'))
results_df = results_df.rename(columns=lambda x: re.sub(r'[^A-Za-z0-9_]+', '_', x))

# Sample 10K for global SHAP
sample_df = train_df.sample(10000, random_state=42)
X_sample  = sample_df[FEATURES]
log(f"Sample shape: {X_sample.shape}")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Global SHAP Values
# ═══════════════════════════════════════════════════════════════
header("STEP 2 — Calculating Global SHAP Values")

explainer    = shap.TreeExplainer(model)
shap_values  = explainer.shap_values(X_sample)

# For binary classification, shap_values may be a list [neg_class, pos_class]
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]
else:
    shap_values_pos = shap_values

log(f"SHAP values shape: {shap_values_pos.shape}")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Global Summary Plots
# ═══════════════════════════════════════════════════════════════
header("STEP 3 — Generating Global Plots")

# --- 3a. Summary Bar Chart ---
step("Plotting Summary Bar Chart...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_pos, X_sample, plot_type='bar', max_display=20, show=False)
plt.title('Top 20 Features — Mean |SHAP Value| (Global Feature Importance)', fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'shap_summary_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
log("Saved: shap_summary_bar.png")

# --- 3b. Beeswarm Plot ---
step("Plotting Beeswarm Chart...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_pos, X_sample, max_display=20, show=False)
plt.title('SHAP Beeswarm — Direction & Magnitude of Feature Impact', fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
plt.close()
log("Saved: shap_beeswarm.png")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — [NEW] Individual Waterfall Plots
# ═══════════════════════════════════════════════════════════════
header("STEP 4 — [NEW] Individual SHAP Waterfall Plots")

# Merge results to get PRED_PROB on sample
sample_with_prob = sample_df.merge(
    results_df[['SK_ID_CURR', 'PRED_PROB', 'RISK_TIER', 'TARGET']],
    on='SK_ID_CURR', how='left'
)
sample_with_prob = sample_with_prob.dropna(subset=['PRED_PROB'])

# --- 4a. Highest risk client ---
step("Waterfall for High-Risk client...")
high_risk_idx = sample_with_prob['PRED_PROB'].nlargest(1).index[0]
hr_pos = sample_with_prob.index.get_loc(high_risk_idx)

shap_explanation = shap.Explanation(
    values      = shap_values_pos[hr_pos],
    base_values = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value),
    data        = X_sample.iloc[hr_pos].values,
    feature_names = FEATURES
)

plt.figure(figsize=(12, 8))
shap.waterfall_plot(shap_explanation, max_display=15, show=False)
hr_prob = float(sample_with_prob.loc[high_risk_idx, 'PRED_PROB'])
plt.title(f'SHAP Waterfall — High-Risk Client (PD = {hr_prob:.1%})\n'
          f'Client ID: {int(sample_with_prob.loc[high_risk_idx, "SK_ID_CURR"])}',
          fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'shap_waterfall_highrisk.png'), dpi=150, bbox_inches='tight')
plt.close()
log(f"Saved: shap_waterfall_highrisk.png  (PD={hr_prob:.1%})")

# --- 4b. Lowest risk client ---
step("Waterfall for Low-Risk client...")
low_risk_idx = sample_with_prob['PRED_PROB'].nsmallest(1).index[0]
lr_pos = sample_with_prob.index.get_loc(low_risk_idx)

shap_explanation_lr = shap.Explanation(
    values      = shap_values_pos[lr_pos],
    base_values = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value),
    data        = X_sample.iloc[lr_pos].values,
    feature_names = FEATURES
)

plt.figure(figsize=(12, 8))
shap.waterfall_plot(shap_explanation_lr, max_display=15, show=False)
lr_prob = float(sample_with_prob.loc[low_risk_idx, 'PRED_PROB'])
plt.title(f'SHAP Waterfall — Low-Risk Client (PD = {lr_prob:.1%})\n'
          f'Client ID: {int(sample_with_prob.loc[low_risk_idx, "SK_ID_CURR"])}',
          fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'shap_waterfall_lowrisk.png'), dpi=150, bbox_inches='tight')
plt.close()
log(f"Saved: shap_waterfall_lowrisk.png  (PD={lr_prob:.1%})")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — [NEW] SHAP Dependence Plots
# ═══════════════════════════════════════════════════════════════
header("STEP 5 — [NEW] SHAP Dependence Plots")

feature_names = np.array(FEATURES)

def shap_dep_plot(feature, interaction, title, filename):
    step(f"Dependence: {feature} (interaction={interaction})...")
    feat_idx  = list(FEATURES).index(feature)      if feature      in FEATURES else None
    inter_idx = list(FEATURES).index(interaction)  if interaction  in FEATURES else None
    if feat_idx is None:
        log(f"  Feature '{feature}' not found — skip")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.dependence_plot(
        feat_idx, shap_values_pos, X_sample,
        interaction_index=inter_idx,
        ax=ax, show=False
    )
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel(feature.replace('_', ' '), fontsize=11)
    ax.set_ylabel(f'SHAP value for {feature.replace("_", " ")}', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved: {filename}")

shap_dep_plot(
    'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'SHAP Dependence: EXT_SOURCE_2 (coloured by EXT_SOURCE_3)\nHigher score = lower risk',
    'shap_dep_extsource2.png'
)
shap_dep_plot(
    'CREDIT_INCOME_RATIO', 'AGE_YEARS',
    'SHAP Dependence: Credit-to-Income Ratio (coloured by Age)\nHigher ratio = higher risk burden',
    'shap_dep_credit_income.png'
)
shap_dep_plot(
    'bureau_days_credit_mean', 'bureau_bad_debt_flag',
    'SHAP Dependence: Bureau Credit History Age (coloured by Bad Debt Flag)',
    'shap_dep_bureau_days.png'
)


# ═══════════════════════════════════════════════════════════════
# STEP 6 — Save SHAP Artifacts for Streamlit
# ═══════════════════════════════════════════════════════════════
header("STEP 6 — Saving Artifacts for Streamlit")

# Save SHAP values + feature names as Parquet
shap_df = pd.DataFrame(shap_values_pos, columns=FEATURES)
shap_df['SK_ID_CURR'] = sample_df['SK_ID_CURR'].values
shap_df.to_parquet(os.path.join(DATA_DIR, 'shap_values_sample.parquet'), index=False)

# Save expected value for individual predictions
expected_val = explainer.expected_value
if isinstance(expected_val, (list, np.ndarray)):
    expected_val = float(expected_val[1])
else:
    expected_val = float(expected_val)
with open(os.path.join(MODELS_DIR, 'shap_expected_value.pkl'), 'wb') as f:
    pickle.dump(expected_val, f)

print(f"\nSaved SHAP artifacts successfully:")
for fname in ['shap_summary_bar.png', 'shap_beeswarm.png',
              'shap_waterfall_highrisk.png', 'shap_waterfall_lowrisk.png',
              'shap_dep_extsource2.png', 'shap_dep_credit_income.png',
              'shap_dep_bureau_days.png']:
    path = os.path.join(REPORTS_DIR, fname)
    exists = '✓' if os.path.exists(path) else '✗ MISSING'
    size   = f"{os.path.getsize(path)/1024:.0f} KB" if os.path.exists(path) else ''
    print(f"  {exists}  reports/{fname}  {size}")
print(f"  ✓  data/shap_values_sample.parquet")
print(f"  ✓  models/shap_expected_value.pkl")

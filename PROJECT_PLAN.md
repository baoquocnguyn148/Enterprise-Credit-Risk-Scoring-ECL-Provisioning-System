# Credit Risk Scoring — Project Plan (End-to-End)

> **Dataset:** Home Credit Default Risk (Kaggle)  
> **Goal:** Predict loan default probability · Feature Engineering · Model Interpretability · Business Insights  
> **Target role:** Data Analyst / Data Scientist — Banking & Fintech  

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Phase 1 — Setup & EDA](#phase-1--setup--eda)
3. [Phase 2 — Feature Engineering](#phase-2--feature-engineering)
4. [Phase 3 — Modeling](#phase-3--modeling)
5. [Phase 4 — Model Interpretation & Business Insights](#phase-4--model-interpretation--business-insights)
6. [Phase 5 — Dashboard & Storytelling](#phase-5--dashboard--storytelling)
7. [Phase 6 — Packaging & GitHub Portfolio](#phase-6--packaging--github-portfolio)
8. [Tech Stack](#tech-stack)
9. [Timeline](#timeline)
10. [Folder Structure](#folder-structure)

---

## Project Overview

| Item | Detail |
|---|---|
| **Problem** | Binary classification — predict whether a loan applicant will default |
| **Target variable** | `TARGET` (1 = defaulted, 0 = repaid) |
| **Primary metric** | AUC-ROC (industry standard for credit risk) |
| **Secondary metrics** | KS Statistic, Gini Coefficient, Calibration |
| **Data size** | ~307,511 applicants · 122 features (main table) |
| **Files used** | `application_train/test.csv`, `bureau.csv`, `previous_application.csv`, `installments_payments.csv` |
| **Target AUC** | ≥ 0.77 |

---

## Phase 1 — Setup & EDA

**Duration:** ~3–4 days  
**Output:** `notebooks/01_EDA.ipynb`

### 1.1 Environment Setup

```bash
pip install pandas numpy matplotlib seaborn lightgbm shap scikit-learn optuna pyarrow fastparquet
```

```python
# Recommended versions
pandas==2.1.0
lightgbm==4.1.0
shap==0.43.0
scikit-learn==1.3.0
```

### 1.2 Load & Inspect Data

- Load `application_train.csv` and `HomeCredit_columns_description.csv`
- **Map column descriptions** into a reference dict — use this throughout the project to answer "what does column X mean?" during interviews
- Check: `shape`, `dtypes`, `duplicated()`, `value_counts()` on key columns
- Identify numerical vs categorical columns early

```python
# Build a description lookup for any column
col_desc = pd.read_csv('HomeCredit_columns_description.csv', encoding='latin1')
desc_map = dict(zip(col_desc['Row'], col_desc['Description']))

def describe_col(col):
    return desc_map.get(col, 'No description available')
```

### 1.3 Target Variable Analysis

- Calculate default rate → expected ~8.07%
- Confirm **class imbalance** with bar chart + pie chart
- Document chosen strategy: AUC-ROC as primary metric, `scale_pos_weight` in LightGBM

> **Why not accuracy?** With 8% default rate, a model that always predicts "no default" achieves 92% accuracy but is completely useless for the bank.

### 1.4 Missing Value Analysis

- Compute `missing %` per column, sort descending
- Plot color-coded horizontal bar chart:
  - 🔴 Red: `>40%` missing → candidate for dropping
  - 🟡 Yellow: `15–40%` missing → impute carefully
  - 🔵 Blue: `<15%` missing → standard impute
- Categorise columns into `drop_list` and `impute_list`
- Document imputation strategy per group (median for numerical, mode or `'Unknown'` for categorical)

### 1.5 Numerical Feature Analysis

Analyse these key features — always split distributions by `TARGET=0` vs `TARGET=1`:

| Feature | What to look for |
|---|---|
| `AMT_INCOME_TOTAL` | Income distribution, outliers at 99th pct |
| `AMT_CREDIT` | Credit amount vs default rate |
| `AMT_ANNUITY` | Monthly payment burden |
| `AMT_GOODS_PRICE` | Goods price vs credit amount gap |
| `DAYS_BIRTH` | Convert to `AGE_YEARS`, check by age band |
| `DAYS_EMPLOYED` | Anomaly at 365243, convert to `YEARS_EMPLOYED` |
| `EXT_SOURCE_1/2/3` | External credit scores — strongest predictors |

**Key charts to produce:**
- Overlapping histograms by target (9-panel grid)
- Default rate by age group (bar chart)
- `EXT_SOURCE` score distributions — show clear separation

**Known anomalies to flag now:**
- `DAYS_EMPLOYED == 365243` → means unemployed, not actually employed 1000 years. Replace with NaN + create `EMP_ANOMALY` flag
- `CODE_GENDER == 'XNA'` → 4 rows, treat as `'Unknown'`
- `AMT_INCOME_TOTAL` extreme outliers → cap at 99th percentile for visualisation

### 1.6 Categorical Feature Analysis

Build a reusable helper and apply to all key categoricals:

```python
def plot_cat_default_rate(col, ax, top_n=10):
    stats = (
        app.groupby(col)['TARGET']
        .agg(['mean', 'count'])
        .query('count > 50')
        .sort_values('mean', ascending=False)
        .head(top_n)
    )
    # horizontal bar chart + overall average reference line
```

**Categories to analyse:**

| Column | Business question |
|---|---|
| `NAME_INCOME_TYPE` | Which income source is riskiest? |
| `OCCUPATION_TYPE` | Which jobs correlate with default? |
| `NAME_CONTRACT_TYPE` | Cash vs Revolving — different risk profiles? |
| `NAME_EDUCATION_TYPE` | Does education level predict repayment? |
| `NAME_HOUSING_TYPE` | Housing ownership as proxy for stability? |
| `NAME_FAMILY_STATUS` | Marital status and financial responsibility? |
| `CODE_GENDER` | Gender gap in default rate? |

### 1.7 Correlation & Multicollinearity

- Compute `|Pearson correlation|` with TARGET for all numerical features
- Plot top 20 features, color-coded by sign (positive = red, negative = green)
- Heatmap for top 12 features → identify multicollinear pairs (e.g. `AMT_CREDIT` ↔ `AMT_GOODS_PRICE`)
- Note highly correlated pairs for Phase 2 feature selection

### 1.8 EDA Findings Summary

End the notebook with a markdown cell summarising:

```
1. Class imbalance: ~8% default rate → use AUC-ROC
2. Top numerical predictors: EXT_SOURCE_2 > EXT_SOURCE_3 > EXT_SOURCE_1
3. Key risk segments: Age 20–30, Laborers, Secondary education, Cash loans
4. Anomalies to fix: DAYS_EMPLOYED=365243, CODE_GENDER=XNA
5. Multicollinearity: AMT_CREDIT ↔ AMT_GOODS_PRICE (r=0.99) — engineer ratio instead
6. Features to engineer in Phase 2: [list here]
```

Save processed dataframe to `app_eda.parquet` for Phase 2.

**Deliverable checklist:**
- [ ] ≥ 10 professional charts, each with English business commentary below
- [ ] All anomalies documented
- [ ] Column description lookup built
- [ ] `app_eda.parquet` saved

---

## Phase 2 — Feature Engineering

**Duration:** ~4–5 days  
**Output:** `notebooks/02_feature_engineering.ipynb`, `data/train_features.parquet`

> This is the most differentiating section of the project. Most juniors skip this. Don't.

### 2.1 Features from application_train.csv

**Financial ratios** — capture relative burden, not just absolute amounts:

```python
app['CREDIT_INCOME_RATIO']   = app['AMT_CREDIT']  / app['AMT_INCOME_TOTAL']
app['ANNUITY_INCOME_RATIO']  = app['AMT_ANNUITY'] / app['AMT_INCOME_TOTAL']
app['CREDIT_TERM']           = app['AMT_CREDIT']  / app['AMT_ANNUITY']
app['GOODS_CREDIT_RATIO']    = app['AMT_GOODS_PRICE'] / app['AMT_CREDIT']
app['CREDIT_DOWNPAYMENT']    = app['AMT_GOODS_PRICE'] - app['AMT_CREDIT']
```

**Time-based features:**

```python
app['AGE_YEARS']             = -app['DAYS_BIRTH']    / 365
app['YEARS_EMPLOYED']        = -app['DAYS_EMPLOYED'] / 365
app['DAYS_EMPLOYED_RATIO']   = app['DAYS_EMPLOYED']  / app['DAYS_BIRTH']
app['YEARS_ID_PUBLISH']      = -app['DAYS_ID_PUBLISH'] / 365
app['YEARS_LAST_PHONE_CHANGE'] = -app['DAYS_LAST_PHONE_CHANGE'] / 365
```

**Anomaly flags:**

```python
app['EMP_ANOMALY']           = (app['DAYS_EMPLOYED'] == 365243).astype(int)
app['DAYS_EMPLOYED']         = app['DAYS_EMPLOYED'].replace(365243, np.nan)
```

**Document quality indicators:**

```python
app['FLAG_DOCS_SUM']         = app[[c for c in app.columns if 'FLAG_DOCUMENT' in c]].sum(axis=1)
app['CNT_CHILDREN_RATIO']    = app['CNT_CHILDREN'] / (app['CNT_FAM_MEMBERS'] + 1e-6)
```

### 2.2 Aggregation from bureau.csv

```python
bureau = pd.read_csv('bureau.csv')

bureau_agg = bureau.groupby('SK_ID_CURR').agg(
    bureau_loan_count          = ('SK_ID_BUREAU', 'count'),
    bureau_active_count        = ('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
    bureau_closed_count        = ('CREDIT_ACTIVE', lambda x: (x == 'Closed').sum()),
    bureau_credit_sum          = ('AMT_CREDIT_SUM', 'sum'),
    bureau_credit_mean         = ('AMT_CREDIT_SUM', 'mean'),
    bureau_credit_max          = ('AMT_CREDIT_SUM', 'max'),
    bureau_debt_sum            = ('AMT_CREDIT_SUM_DEBT', 'sum'),
    bureau_overdue_mean        = ('AMT_CREDIT_SUM_OVERDUE', 'mean'),
    bureau_overdue_max         = ('AMT_CREDIT_SUM_OVERDUE', 'max'),
    bureau_days_credit_mean    = ('DAYS_CREDIT', 'mean'),
    bureau_days_enddate_max    = ('DAYS_CREDIT_ENDDATE', 'max'),
    bureau_bad_debt_count      = ('CREDIT_DAY_OVERDUE', lambda x: (x > 0).sum()),
).reset_index()

bureau_agg['bureau_bad_debt_flag']    = (bureau_agg['bureau_bad_debt_count'] > 0).astype(int)
bureau_agg['bureau_active_ratio']     = bureau_agg['bureau_active_count'] / (bureau_agg['bureau_loan_count'] + 1e-6)
bureau_agg['bureau_debt_credit_ratio'] = bureau_agg['bureau_debt_sum'] / (bureau_agg['bureau_credit_sum'] + 1e-6)
```

### 2.3 Aggregation from previous_application.csv

```python
prev = pd.read_csv('previous_application.csv')

prev_agg = prev.groupby('SK_ID_CURR').agg(
    prev_app_count             = ('SK_ID_PREV', 'count'),
    prev_approved_count        = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
    prev_refused_count         = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
    prev_credit_mean           = ('AMT_CREDIT', 'mean'),
    prev_credit_max            = ('AMT_CREDIT', 'max'),
    prev_annuity_mean          = ('AMT_ANNUITY', 'mean'),
    prev_down_payment_mean     = ('AMT_DOWN_PAYMENT', 'mean'),
    prev_days_decision_mean    = ('DAYS_DECISION', 'mean'),
    prev_consumer_count        = ('NAME_CONTRACT_TYPE', lambda x: (x == 'Consumer loans').sum()),
    prev_cash_count            = ('NAME_CONTRACT_TYPE', lambda x: (x == 'Cash loans').sum()),
).reset_index()

prev_agg['prev_refused_ratio']        = prev_agg['prev_refused_count'] / (prev_agg['prev_app_count'] + 1e-6)
prev_agg['prev_approved_ratio']       = prev_agg['prev_approved_count'] / (prev_agg['prev_app_count'] + 1e-6)
```

### 2.4 Aggregation from installments_payments.csv (optional, adds ~0.002 AUC)

```python
inst = pd.read_csv('installments_payments.csv')
inst['PAYMENT_DIFF']     = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
inst['DAYS_PAST_DUE']    = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
inst['DAYS_PAST_DUE']    = inst['DAYS_PAST_DUE'].clip(lower=0)

inst_agg = inst.groupby('SK_ID_CURR').agg(
    inst_payment_diff_mean = ('PAYMENT_DIFF', 'mean'),
    inst_payment_diff_max  = ('PAYMENT_DIFF', 'max'),
    inst_days_past_due_mean = ('DAYS_PAST_DUE', 'mean'),
    inst_days_past_due_max  = ('DAYS_PAST_DUE', 'max'),
    inst_late_payment_count = ('DAYS_PAST_DUE', lambda x: (x > 0).sum()),
).reset_index()
```

### 2.5 Merge & Validate

```python
# Merge all feature tables
df = app.copy()
df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
df = df.merge(prev_agg,   on='SK_ID_CURR', how='left')
df = df.merge(inst_agg,   on='SK_ID_CURR', how='left')

# ── VALIDATION (do not skip) ───────────────────────────────────────────────
assert df.shape[0] == app.shape[0], "Row count changed after merge — check for duplicates"

print(f"Shape after merge: {df.shape}")
print(f"New features added: {df.shape[1] - app.shape[1]}")

# Check missing rate on aggregated features
new_cols_missing = df[bureau_agg.columns.tolist() + prev_agg.columns.tolist()].isnull().mean()
print("\nMissing rate for aggregated features (NaN = no history, expected):")
print(new_cols_missing[new_cols_missing > 0])
# NaN here is valid — it means the client has no bureau/previous history
# Fill with 0 or median depending on the feature semantics
```

### 2.6 Encoding & Imputation

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Binary encoding for 2-category columns
binary_cols = [c for c in df.select_dtypes('object') if df[c].nunique() == 2]
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# One-hot encode remaining categoricals
df = pd.get_dummies(df, dummy_na=False)

# Impute remaining numerical NaN
imputer = SimpleImputer(strategy='median')
df[df.select_dtypes(include=np.number).columns] = imputer.fit_transform(
    df.select_dtypes(include=np.number)
)
```

### 2.7 Feature Selection (remove noise)

```python
# Remove columns with near-zero variance
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)

# Remove highly correlated pairs (r > 0.95)
def remove_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return to_drop

drop_corr = remove_correlated_features(df.drop(columns=['TARGET', 'SK_ID_CURR']))
print(f"Dropping {len(drop_corr)} highly correlated features: {drop_corr[:5]}...")
```

**Deliverable checklist:**
- [ ] ≥ 40 engineered features documented with business rationale
- [ ] Merge validation passed (row count unchanged)
- [ ] Missing values after merge explained (NaN = no history, not data error)
- [ ] `train_features.parquet` saved with all features + TARGET

---

## Phase 3 — Modeling

**Duration:** ~3 days  
**Output:** `notebooks/03_modeling.ipynb`, `models/lgbm_model.pkl`

### 3.1 Train/Validation Split

```python
from sklearn.model_selection import train_test_split, StratifiedKFold

FEATURES = [c for c in df.columns if c not in ['TARGET', 'SK_ID_CURR']]
X = df[FEATURES]
y = df['TARGET']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 3.2 Baseline — Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Use only top 20 correlated features for baseline
top_features = correlations.head(20).index.tolist()
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train[top_features], y_train)

baseline_auc = roc_auc_score(y_val, lr.predict_proba(X_val[top_features])[:, 1])
print(f"Baseline Logistic Regression AUC: {baseline_auc:.4f}")
# Expected: ~0.68–0.70
```

### 3.3 Main Model — LightGBM with 5-Fold CV

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

params = {
    'objective':        'binary',
    'metric':           'auc',
    'boosting_type':    'gbdt',
    'n_estimators':     2000,
    'learning_rate':    0.05,
    'num_leaves':       31,
    'max_depth':        -1,
    'min_child_samples': 20,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       0.1,
    'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),  # handle imbalance
    'random_state':     42,
    'n_jobs':           -1,
    'verbose':          -1,
}

skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
models    = []
fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
    )

    oof_preds[val_idx] = model.predict_proba(X_vl)[:, 1]
    fold_auc = roc_auc_score(y_vl, oof_preds[val_idx])
    fold_aucs.append(fold_auc)
    models.append(model)
    print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

print(f"\nOOF AUC: {roc_auc_score(y, oof_preds):.4f}")
print(f"Mean fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
```

### 3.4 Model Evaluation

```python
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve
)

# AUC-ROC curve
fpr, tpr, thresholds = roc_curve(y, oof_preds)
auc_score = roc_auc_score(y, oof_preds)

# KS Statistic (standard in banking)
ks_stat = max(tpr - fpr)
print(f"KS Statistic: {ks_stat:.4f}")

# Gini Coefficient
gini = 2 * auc_score - 1
print(f"Gini Coefficient: {gini:.4f}")

# Optimal threshold (maximise F1 or KS)
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
y_pred = (oof_preds >= optimal_threshold).astype(int)
print(classification_report(y, y_pred))
```

### 3.5 Calibration Check

> **Why this matters:** The Risk Predictor in Phase 5 outputs probability. If the model says "30% default probability" but the actual rate is 60%, the bank will misprice risk.

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

fraction_of_positives, mean_predicted_value = calibration_curve(
    y, oof_preds, n_bins=10
)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='LightGBM')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration Curve')
ax.legend()
plt.show()
# If the curve deviates significantly → apply Platt scaling or Isotonic Regression
```

### 3.6 Error Analysis — False Negatives

> **Most important for banking:** False Negatives = clients predicted to repay but actually defaulted. These are the highest-cost errors.

```python
# Identify false negatives at optimal threshold
results_df = X_val.copy()
results_df['TARGET']     = y_val.values
results_df['PRED_PROB']  = models[-1].predict_proba(X_val)[:, 1]
results_df['PRED_LABEL'] = (results_df['PRED_PROB'] >= optimal_threshold).astype(int)

fn_df = results_df[(results_df['TARGET'] == 1) & (results_df['PRED_LABEL'] == 0)]
fp_df = results_df[(results_df['TARGET'] == 0) & (results_df['PRED_LABEL'] == 1)]

print(f"False Negatives: {len(fn_df)} ({len(fn_df)/y_val.sum()*100:.1f}% of actual defaults missed)")
print(f"False Positives: {len(fp_df)} ({len(fp_df)/(y_val==0).sum()*100:.1f}% of good clients wrongly rejected)")

# Profile false negatives — who are the clients we missed?
print("\nFalse Negative profile (clients we missed):")
key_cols = ['AGE_YEARS', 'CREDIT_INCOME_RATIO', 'EXT_SOURCE_2',
            'bureau_overdue_mean', 'prev_refused_ratio']
print(fn_df[key_cols].describe())
```

### 3.7 Save Model

```python
import pickle

# Save best model (highest fold AUC)
best_model = models[np.argmax(fold_aucs)]
with open('models/lgbm_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save feature list
with open('models/feature_list.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)

print("Model saved ✓")
```

**Deliverable checklist:**
- [ ] Baseline AUC documented (Logistic Regression)
- [ ] LightGBM OOF AUC ≥ 0.75
- [ ] KS Statistic and Gini reported
- [ ] Calibration curve plotted and interpreted
- [ ] Error analysis completed — False Negative profile documented
- [ ] `lgbm_model.pkl` and `feature_list.pkl` saved

---

## Phase 4 — Model Interpretation & Business Insights

**Duration:** ~2–3 days  
**Output:** `notebooks/04_interpretation.ipynb`, `reports/business_insights.md`

### 4.1 SHAP Global Analysis

```python
import shap

explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_val)

# Summary plot — global feature importance
shap.summary_plot(shap_values, X_val, plot_type='bar', max_display=20)
plt.title('Top 20 Features by Mean |SHAP| Value')
plt.savefig('reports/shap_summary_bar.png', dpi=150, bbox_inches='tight')

# Beeswarm plot — direction of impact
shap.summary_plot(shap_values, X_val, max_display=20)
plt.savefig('reports/shap_summary_beeswarm.png', dpi=150, bbox_inches='tight')
```

### 4.2 SHAP Individual Explanations

```python
# Waterfall plot — explain a specific high-risk client
high_risk_idx = results_df['PRED_PROB'].nlargest(1).index[0]
shap.waterfall_plot(
    shap.Explanation(
        values      = shap_values[high_risk_idx],
        base_values = explainer.expected_value,
        data        = X_val.iloc[high_risk_idx],
        feature_names = FEATURES
    )
)

# Waterfall plot — explain a low-risk client
low_risk_idx = results_df['PRED_PROB'].nsmallest(1).index[0]
shap.waterfall_plot(...)

# Force plot — for Streamlit embedding
shap.force_plot(
    explainer.expected_value,
    shap_values[high_risk_idx],
    X_val.iloc[high_risk_idx],
    matplotlib=True
)
```

### 4.3 SHAP Dependence Plots

```python
# How EXT_SOURCE_2 affects risk (interacted with another feature)
shap.dependence_plot('EXT_SOURCE_2', shap_values, X_val, interaction_index='EXT_SOURCE_3')
shap.dependence_plot('CREDIT_INCOME_RATIO', shap_values, X_val, interaction_index='AGE_YEARS')
```

### 4.4 Customer Segmentation by Risk

```python
# Assign risk tier based on predicted probability
results_df['RISK_TIER'] = pd.cut(
    results_df['PRED_PROB'],
    bins   = [0, 0.05, 0.15, 0.30, 1.0],
    labels = ['Very Low', 'Low', 'Medium', 'High']
)

# Profile each segment
segment_profile = results_df.groupby('RISK_TIER').agg(
    count           = ('TARGET', 'count'),
    actual_default  = ('TARGET', 'mean'),
    avg_age         = ('AGE_YEARS', 'mean'),
    avg_credit_ratio = ('CREDIT_INCOME_RATIO', 'mean'),
    avg_ext_source2  = ('EXT_SOURCE_2', 'mean'),
)
print(segment_profile)
```

### 4.5 Business Insights Document

Write `reports/business_insights.md` with this structure:

```markdown
## Executive Summary
[2–3 sentences: what the model does, what AUC was achieved, what it enables]

## Top Risk Drivers
| Rank | Feature | Direction | Business Interpretation |
|------|---------|-----------|------------------------|
| 1 | EXT_SOURCE_2 | ↑ score → ↓ risk | Clients with strong external credit history are significantly less likely to default |
| 2 | ... | | |

## High-Risk Customer Profile
Clients with the highest predicted default probability typically share:
- Age: 20–30 years old
- Income ratio: Credit-to-income ratio > 0.5
- Employment: Laborers or low-skill occupations
- Bureau history: ≥ 1 overdue payment on record
- Previous applications: Refused ratio > 30%

## Low-Risk Customer Profile
[Mirror of above for low-risk segment]

## Policy Recommendations
1. **Credit Ceiling:** For clients with CREDIT_INCOME_RATIO > 0.5, cap approved credit at 80% of requested amount
2. **Collateral Requirement:** Clients with bureau_bad_debt_flag = 1 should provide additional collateral
3. **Monitoring:** Flag clients with prev_refused_ratio > 0.3 for manual review
4. **Early Warning:** Clients with DAYS_PAST_DUE_mean > 5 in installment history → trigger proactive outreach

## Model Limitations
- Model trained on 2016–2017 data — economic conditions may have shifted
- Does not account for macroeconomic signals (interest rates, unemployment)
- External credit scores (EXT_SOURCE) are black-box features — dependency risk
- May exhibit bias against young applicants — requires fairness audit before production
- Calibration should be re-checked quarterly as portfolio mix changes

## Next Steps for Production
- A/B test model vs existing scorecard on 5% of new applications
- Implement drift monitoring (PSI on input features monthly)
- Set up feedback loop to retrain with new default labels every 6 months
```

**Deliverable checklist:**
- [ ] SHAP summary bar + beeswarm saved as PNG
- [ ] Waterfall plots for ≥ 1 high-risk and ≥ 1 low-risk client
- [ ] Dependence plots for top 3 features
- [ ] Risk tier segmentation table
- [ ] `business_insights.md` with all 6 sections completed

---

## Phase 5 — Dashboard & Storytelling

**Duration:** ~3–4 days  
**Output:** Streamlit app (DA track: Tableau `.twbx`)

### Option A — Streamlit App (recommended for DS track)

**App structure: `app/streamlit_app.py`**

```
Tab 1: Model Overview
├── AUC-ROC curve (interactive plotly)
├── Confusion matrix at optimal threshold
├── KS Statistic, Gini, AUC side by side
└── Calibration curve

Tab 2: Risk Predictor
├── Input form: age, income, credit amount, occupation, etc.
├── → Output: default probability + risk tier badge
├── → SHAP waterfall for this specific prediction
└── → "What would lower this client's risk?" recommendation

Tab 3: Portfolio Analysis
├── Risk tier distribution of current portfolio
├── Default rate by segment (income type, occupation, age group)
└── Feature importance (SHAP bar chart)
```

**Deploy to Streamlit Cloud:**
```bash
# 1. Push to GitHub
# 2. Go to share.streamlit.io
# 3. Connect repo → select app/streamlit_app.py
# 4. Add requirements.txt → Deploy
# Free hosting, public URL to put in CV
```

### Option B — Tableau (recommended for DA track)

**Dashboard 1: Portfolio Overview**
- KPI tiles: Total applicants, Default rate, Avg credit amount
- Default rate trend (if time data available)
- Geographic breakdown (if region data available)

**Dashboard 2: Risk Driver Analysis**
- Feature importance bar chart (from SHAP output CSV)
- Default rate by income type (cross-tab)
- Default rate by occupation (sorted bar)
- Credit-to-income ratio distribution by target

**Dashboard 3: Customer Segmentation**
- Risk tier breakdown (donut chart)
- Segment profile comparison table
- Scatter: Age vs Credit Ratio, colored by default

**Connect data:** Export `results_df` and `segment_profile` as CSV from Phase 3/4, import into Tableau.

---

## Phase 6 — Packaging & GitHub Portfolio

**Duration:** ~2 days  
**Output:** Clean GitHub repo, professional README

### 6.1 Folder Structure

```
credit-risk-scoring/
├── data/
│   ├── .gitkeep
│   └── sample_100rows.csv          # Small sample for demo
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_interpretation.ipynb
├── src/
│   ├── __init__.py
│   ├── features.py                 # Feature engineering functions
│   ├── modeling.py                 # Train/predict functions
│   └── utils.py                   # Helpers (plot functions, etc.)
├── app/
│   └── streamlit_app.py
├── models/
│   ├── lgbm_model.pkl
│   └── feature_list.pkl
├── reports/
│   ├── business_insights.md
│   ├── shap_summary_bar.png
│   ├── shap_summary_beeswarm.png
│   └── model_evaluation.png
├── README.md
├── requirements.txt
└── .gitignore
```

### 6.2 README.md Structure

```markdown
# Credit Risk Scoring Model
> End-to-end ML pipeline to predict loan default probability
> for 307K+ consumer loan applicants (Home Credit Group dataset)

## Results
| Metric | Score |
|--------|-------|
| AUC-ROC | 0.77 |
| KS Statistic | 0.XX |
| Gini Coefficient | 0.XX |

## Key Insights
- EXT_SOURCE scores are the strongest predictors of default
- Clients aged 20–30 have 2× the default rate of clients aged 50+
- Credit-to-income ratio > 0.5 is a significant risk signal

## Tech Stack
Python · LightGBM · SHAP · Streamlit · pandas · scikit-learn

## Project Structure
[folder tree here]

## How to Run
[installation + run instructions]

## Live Demo
[Streamlit Cloud URL]

## Screenshots
[Dashboard screenshots here]
```

### 6.3 .gitignore

```
data/*.csv
data/*.parquet
models/*.pkl
__pycache__/
.ipynb_checkpoints/
*.egg-info/
.env
```

### 6.4 CV Bullet Points (copy-paste ready)

```
• Built end-to-end credit risk scoring pipeline on 307K+ loan records;
  engineered 40+ features from 4 data sources achieving AUC-ROC of 0.77

• Applied SHAP to identify top default drivers and translate model outputs
  into 4 actionable lending policy recommendations

• Deployed interactive Risk Predictor on Streamlit Cloud enabling real-time
  default probability scoring with individual loan-level explanations
```

---

## Tech Stack

| Category | Tools |
|---|---|
| **Data manipulation** | pandas, numpy |
| **Visualisation** | matplotlib, seaborn, plotly |
| **Modeling** | scikit-learn, LightGBM |
| **Explainability** | SHAP |
| **Hyperparameter tuning** | Optuna *(Phase 3 optional)* |
| **Dashboard** | Streamlit / Tableau |
| **Version control** | Git + GitHub |
| **Notebook** | Jupyter / Google Colab / Kaggle |
| **Deploy** | Streamlit Cloud *(free)* |

---

## Timeline

| Phase | Part-time (~2–3h/day) | Full-time (focused) |
|---|---|---|
| 1 — EDA | 3–4 days | 2 days |
| 2 — Feature Engineering | 4–5 days | 3 days |
| 3 — Modeling | 3 days | 2 days |
| 4 — Interpretation | 2–3 days | 1–2 days |
| 5 — Dashboard | 3–4 days | 2 days |
| 6 — Packaging | 2 days | 1 day |
| **Total** | **~17–21 days** | **~11–12 days** |

> **Note:** Phase 2 is most frequently underestimated. Merging multiple tables cleanly without data leaks or duplicate rows takes longer than expected. Budget an extra day as buffer.

---

## Folder Structure

```
credit-risk-scoring/
├── data/                           # gitignored (add sample only)
├── notebooks/                      # One notebook per phase
├── src/                            # Reusable Python modules
├── app/                            # Streamlit app
├── models/                         # Saved model artifacts
├── reports/                        # Charts + business_insights.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

*Last updated: 2026 · Home Credit Default Risk · Kaggle Dataset*

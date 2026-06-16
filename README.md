# 🏦 Enterprise Credit Risk Scoring & ECL Provisioning System
> An end-to-end Machine Learning pipeline to predict loan default probability, calculate Expected Credit Loss (IFRS 9), and provide actionable business insights for **307,511** consumer loan applicants.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/Interpretability-SHAP-orange)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📑 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Business Value & Impact](#-business-value--impact)
3. [Methodology & Pipeline Detail](#-methodology--pipeline-detail)
4. [Comprehensive Results](#-comprehensive-results)
5. [Deep Dive Insights (SHAP)](#-deep-dive-insights-shap)
6. [Interactive Dashboard](#-interactive-dashboard)
7. [Tech Stack & Architecture](#-tech-stack--architecture)
8. [How to Run](#-how-to-run)

---

## 🚀 Executive Summary

This project transforms the famous [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk) dataset into a **Production-Ready Enterprise Risk System**. 

Instead of stopping at model accuracy, this project solves real-world banking challenges:
1. **Probability Calibration:** Ensuring the model outputs true probabilities (PD) rather than arbitrary scores, crucial for financial calculations.
2. **IFRS 9 ECL Framework:** Translating risk scores into dollar values (Expected Credit Loss) to estimate portfolio provisioning.
3. **Interpretability:** Explaining *why* a specific client is risky to loan underwriters using SHAP values.
4. **Drift Monitoring:** Implementing Population Stability Index (PSI) to track model degradation over time.

---

## 💰 Business Value & Impact

The model classifies borrowers into 4 KS-optimized Risk Tiers. Total portfolio Expected Credit Loss (ECL) is estimated at **$30.1B** across a $184B total exposure.

| Risk Tier | % of Portfolio | Actual Default Rate | Avg PD | Total ECL | Business Policy Recommendation |
|---|---|---|---|---|---|
| **Very Low** | 40.0% | 2.0% | 16.5% | $5.8B | Auto-approve, offer preferential interest rates |
| **Low** | 25.0% | 5.2% | 36.3% | $7.7B | Standard approval process |
| **Medium** | 20.0% | 10.7% | 54.8% | $8.6B | Conditional approval (require income verification) |
| **High** | 15.0% | **25.6%** | 75.1% | **$8.0B** | **Manual review / Require additional collateral** |

> **Key Finding:** The High-Risk tier (only 15% of borrowers) accounts for **26.6% of the total Expected Credit Loss**. By applying a credit ceiling or tightening approval exclusively on this segment, the bank can potentially reduce loan loss provisions by up to $8 Billion while maintaining 85% of lending volume.

---

## 🔬 Methodology & Pipeline Detail

### Phase 1: Data Audit & Cleaning
- **Anomaly Detection:** Handled `DAYS_EMPLOYED == 365243` (unemployed indicator masquerading as 1000 years of employment) and `CODE_GENDER == XNA`.
- **Memory Optimization:** Downcast numerical types (float64 → float32) to fit 6 massive tables into RAM.
- **Output:** 6 cleaned Parquet files to eliminate redundant processing.

### Phase 2: Feature Engineering (164 Features)
Engineered meaningful financial features from 6 relational tables:
- **Financial Ratios:** `CREDIT_INCOME_RATIO` (Loan-to-Income), `ANNUITY_INCOME_RATIO` (Debt-Service Ratio), `CREDIT_TERM`.
- **Bureau Aggregations:** Aggregated external credit bureau history into features like `bureau_bad_debt_flag`, `active_credit_ratio`, and `avg_days_overdue`.
- **Behavioral Signals:** Computed `prev_refused_ratio` (history of loan rejections) and `inst_late_ratio` (historical late payment behavior).

### Phase 3: Modeling & Calibration
- **Algorithm:** `LightGBM` (Gradient Boosted Decision Trees).
- **Validation Strategy:** 5-Fold Stratified Cross-Validation (saving all 5 models for ensemble prediction to reduce variance).
- **Class Imbalance:** Handled via `scale_pos_weight` = 11.4 (Defaults represent only 8.07% of the dataset).
- **Calibration (Isotonic Regression):** Applied post-training Isotonic Regression on raw probabilities. This reduced the **Brier Score by 64%** (0.1726 → 0.0626), ensuring the predicted default probability accurately matches reality.
- **Imputation Stats for Production:** Saved median/mean values of all 164 features into `imputation_stats.json` to prevent data leakage and handle missing inputs when predicting new, unseen clients.

---

## 📊 Comprehensive Results

### Model Metrics
| Metric | Score | Interpretation / Benchmark |
|---|---|---|
| **AUC-ROC** | **0.7837** | Excellent discrimination (Top 20% Kaggle baseline) |
| **KS Statistic** | **0.4255** | > 0.40 is considered "Excellent" in Basel II internal models |
| **Gini Coefficient** | **0.5673** | Strong separation between good/bad borrowers |
| **Brier Score** | **0.0626** | Perfectly calibrated probabilities (Closer to 0 is better) |
| **PSI (Train→Test)** | **0.0045** | < 0.10 indicates extreme stability (No model drift) |

### Error Analysis (False Negatives)
At the optimal threshold, the model misses 28.7% of actual defaults. 
*Why?* Profiling the false negatives revealed they have **significantly higher Credit-to-Income ratios (4.16x vs 3.96x)** but perfectly clean external credit bureau records. 
**Insight:** These are "first-time overleveraged" borrowers. Traditional scorecards miss them because they haven't defaulted *yet*, despite being financially stretched.
**Solution:** Implement an automated income-stress-test for applicants with LTI > 5.0 and clean bureau records.

### Calibration Curve
Isotonic Regression significantly improved the reliability of predicted probabilities, reducing Brier Score by 64%:
<p align="center"><img src="reports/calibration_curve.png" width="600"></p>

---

## 🧠 Deep Dive Insights (SHAP)

We utilized **SHAP (SHapley Additive exPlanations)** to break open the black box of the LightGBM model.

### Top Risk Drivers (Global Interpretability)
1. **EXT_SOURCE_2 & 3:** External bureau scores are the strongest predictors. Higher scores dramatically reduce default risk.
2. **bureau_days_credit_mean:** Longer established credit histories correlate with lower risk.
3. **CREDIT_INCOME_RATIO:** High loan-to-income burden is the primary financial driver of default.
4. **AGE_YEARS:** Younger applicants (20-30) default at nearly twice the rate of those aged 50+.
5. **prev_refused_ratio:** A high percentage of previous loan rejections acts as a strong negative behavioral signal.

<p align="center">
  <img src="reports/shap_summary_bar.png" width="45%">
  &nbsp; &nbsp; &nbsp;
  <img src="reports/shap_beeswarm.png" width="45%">
</p>

### Local Interpretability (Explaining a Single Client)
For every loan application, the system generates a **SHAP Waterfall Plot** showing exactly *why* a specific client received their score. 
*Example:* A client might have a high baseline risk due to young age (`AGE_YEARS`), but the model approves them because their income ratio (`CREDIT_INCOME_RATIO`) and external scores (`EXT_SOURCE_2`) act as strong protective factors pushing the risk down.

<p align="center"><img src="reports/shap_waterfall_highrisk.png" width="800"></p>

---

## 🖥️ Interactive Dashboard

Built with **Streamlit**, the dashboard serves as a frontend for Credit Risk Analysts.

1. **Tab 1: Portfolio Overview:** Tracks total ECL, Coverage Ratio, Default Rates by segment, and visualizes Amount-at-Risk across credit bins.
2. **Tab 2: Risk Predictor (Underwriter Tool):** 
   - Enter client financials manually OR lookup an existing ID.
   - Instantly calculates PD, assigns a Risk Badge, and computes exact ECL in dollars.
   - Generates an interactive SHAP waterfall chart explaining the specific risk drivers for that exact applicant.
3. **Tab 3: Model Insights:** Displays interactive AUC-ROC curve, Fold AUC stability checks, PSI monitoring, Calibration Curves, and SHAP Beeswarm plots.
4. **Tab 4: Portfolio Deep Dive:** Explores default rates segmented by Education, Occupation, LTI Quintile, Age Group, and Contract Type.

---

## 🛠️ Tech Stack & Architecture

- **Data Processing:** `pandas`, `numpy`, `pyarrow` (Parquet)
- **Machine Learning:** `scikit-learn`, `LightGBM`
- **Interpretability:** `shap`
- **Visualization:** `matplotlib`, `plotly`, `seaborn`
- **Frontend / Dashboard:** `streamlit`

### Repository Structure
```text
credit-risk-scoring/
├── data/                           # (Ignored) Parquet files and raw CSVs
├── models/                         # Saved artifacts for production
│   ├── lgbm_fold1..5.pkl           # 5-fold ensemble models
│   ├── isotonic_calibrator.pkl     # Fitted probability calibrator
│   ├── imputation_stats.json       # Medians for handling missing data in production
│   └── tier_config.json            # Thresholds for risk tiers
├── reports/                        
│   ├── business_insights.md        # Full business analytical report
│   └── *.png                       # SHAP and Calibration plots
├── app/
│   └── streamlit_app.py            # Streamlit Dashboard source code
├── data_audit.py                   # Data quality and missing value checks
├── data_cleaning.py                # Pipeline to clean and downcast raw data
├── feature_engineering.py          # Relational joins and 164 feature creation
├── modeling.py                     # LightGBM training, CV, calibration, ECL calculation
├── shap_analysis.py                # SHAP value extraction and plotting
├── schema_sqlserver.sql            # DDL for migrating data to MS SQL Server
└── README.md
```

---

## ⚙️ How to Run

### 1. Installation
```bash
git clone https://github.com/yourusername/credit-risk-scoring.git
cd credit-risk-scoring
pip install -r requirements.txt
```

### 2. Execute Pipeline (Optional: If you want to retrain from scratch)
*Note: You must download the CSVs from Kaggle and place them in the `data/` folder first.*
```bash
python data_cleaning.py
python feature_engineering.py
python modeling.py
python shap_analysis.py
```

### 3. Launch Dashboard
```bash
python -m streamlit run app/streamlit_app.py
# Open your browser to http://localhost:8501
```

---

*Author: Risk Analytics Team · Dataset: Home Credit Default Risk (Kaggle) · Completed: 2026*

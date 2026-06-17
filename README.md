# 🏦 Enterprise Credit Risk Scoring & ECL Provisioning System (v2.0)
> An end-to-end Machine Learning pipeline to predict loan default probability, calculate Expected Credit Loss under **IFRS 9**, and provide actionable business insights for **307,511** consumer loan applicants.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-blue)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📑 Table of Contents
1. [Executive Summary (v2 Upgrades)](#-executive-summary)
2. [Business Value & ROI](#-business-value--roi)
3. [IFRS 9 ECL Framework & Basel II](#-ifrs-9-ecl-framework--basel-ii)
4. [Methodology & MLOps Pipeline](#-methodology--mlops-pipeline)
5. [Model Interpretability (SHAP & PDP)](#-model-interpretability-shap--pdp)
6. [Tech Stack & Architecture](#-tech-stack--architecture)
7. [How to Run](#-how-to-run)

---

## 🚀 Executive Summary

This project transforms the famous [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset into a **Production-Ready Enterprise Risk System** compliant with Basel II and IFRS 9 standards. 

**🔥 What's New in v2.0 (The Enterprise Upgrade):**
1. **Regulatory Compliance:** Implemented WoE/IV feature screening (Basel II IRB), Dynamic LGD/EAD models, and IFRS 9 Stage 1/2/3 classification with a 4-scenario macroeconomic overlay.
2. **MLOps & API:** Integrated **MLflow** for experiment tracking, **Optuna** for Bayesian Hyperparameter Optimization, and **FastAPI** for production scoring endpoints.
3. **Advanced Calibration:** Anti-leakage Isotonic Regression on strict Out-Of-Fold (OOF) predictions, dropping Brier Score by 64%.
4. **ROI Optimization:** Business module to calculate the absolute profit-maximizing decision threshold.
5. **Governance:** Added a comprehensive [Model Governance Document](reports/model_governance_doc.md) (MRM standard) and Characteristic Stability Index (CSI) monitoring.

---

## 💰 Business Value & ROI

By implementing the `business_roi_analysis.py` threshold optimizer, the bank maximizes net profit by balancing *margin earned from good clients* against *losses from defaults* and *opportunity costs of rejected clients*.

### Profit-Optimal Threshold
At the optimized decision threshold:
- **Net Profit Projected:** **$1.8B**
- **Approval Rate:** **86.1%**
- **Precision:** Good at catching high-risk without over-penalizing safe borrowers.

<p align="center"><img src="reports/roi_threshold_analysis.png" width="800"></p>

---

## 🏦 IFRS 9 ECL Framework & Basel II

Instead of just predicting 0 or 1, this system calculates true dollar-value **Expected Credit Loss (ECL)** using the formula:
`ECL = Probability of Default (PD) × Loss Given Default (LGD) × Exposure At Default (EAD)`

### Core IFRS 9 Upgrades
- **Stage Classification:** Borrows are mapped to Stage 1 (Performing), Stage 2 (Underperforming), and Stage 3 (Impaired) based on their PD and behavioral signals.
- **Lifetime PD modeling:** Uses exponential survival decay to estimate multi-year default probabilities for Stage 2.
- **Dynamic LGD & EAD:** LGD is segment-based (collateralized vs unsecured, cash vs revolving), rather than a static 45%. EAD uses Credit Conversion Factors (CCF) for undrawn limits.
- **Macroeconomic Overlay:** Applies a probability-weighted overlay across 4 macro scenarios (Optimistic, Base, Adverse, Severe).

**Portfolio Result:** Total portfolio ECL is estimated at **$68.7B** across a $184B total exposure (Coverage: 37.3%).

---

## 🔬 Methodology & MLOps Pipeline

### 1. Data Processing & WoE/IV (Basel II)
- Cleaned 6 massive relational tables, engineered 164 features, and created 9 advanced interaction composites (e.g., `STRESS_AGE_X_CREDIT`).
- Passed all features through **Weight of Evidence (WoE)** and **Information Value (IV)** screening to drop 112 useless features, keeping only the 52 most robust predictive signals.

### 2. Bayesian Tuning & LightGBM
- Hyperparameters optimized using **Optuna (TPE Sampler)**.
- 5-Fold Stratified Cross-Validation LightGBM ensemble.

### 3. Unleaked Calibration
- Isotonic Regression fitted strictly on Out-Of-Fold (OOF) CV predictions to prevent data leakage.
- **Brier Score:** Improved by 64% (0.1726 → 0.0626).

### 4. MLOps Tracking
- **MLflow** tracks `auc`, `gini`, `ks_stat`, `brier`, and `ap_score` across runs.
- **CSI (Characteristic Stability Index)** implemented to monitor feature-level drift between train and test datasets.

---

## 🧠 Model Interpretability (SHAP & PDP)

We break open the black box to give underwriters exactly what they need: **Explainability**.

### Partial Dependence Plots (PDP)
PDPs isolate the marginal effect of a single feature on the overall default probability. For example, as `CREDIT_INCOME_RATIO` crosses 4.0, the risk spikes non-linearly.
<p align="center"><img src="reports/pdp_top_features.png" width="700"></p>

### Local Explanations (SHAP Waterfalls)
Every API score comes with the top 3 risk factors. Using SHAP, we can generate a waterfall plot for an exact client showing why they were approved or flagged for review.
<p align="center"><img src="reports/shap_waterfall_highrisk.png" width="700"></p>

---

## 🛠️ Tech Stack & Architecture

- **Machine Learning:** `LightGBM`, `scikit-learn`, `Optuna`
- **Interpretability:** `SHAP`, `sklearn.inspection.PartialDependenceDisplay`
- **MLOps & Serving:** `MLflow`, `FastAPI`, `Uvicorn`
- **Frontend:** `Streamlit`
- **Data Engineering:** `pandas`, `pyarrow`

### Key Files
```text
credit-risk-scoring/
├── app/
│   ├── api.py                      # FastAPI REST Endpoints (/score, /score/batch)
│   └── streamlit_app.py            # Streamlit Interactive Dashboard
├── reports/                        
│   ├── model_governance_doc.md     # Full MRM Model Validation Document
│   └── *.png                       # Plots (SHAP, PDP, ROI, Calibration)
├── feature_engineering.py          # 164 feature creation + interaction terms
├── woe_iv_scorecard.py             # Basel II feature screening
├── optuna_tuning.py                # Bayesian Hyperparameter search
├── modeling.py                     # LGBM CV, Calibration, MLflow, CSI
├── ifrs9_ecl_engine.py             # IFRS 9 Staging, Lifetime PD, Macro overlay
├── business_roi_analysis.py        # Profit curve threshold optimizer
├── shap_analysis.py                # SHAP and PDP generation
└── README.md
```

---

## ⚙️ How to Run

### 1. Installation
```bash
git clone https://github.com/baoquocnguyn148/Enterprise-Credit-Risk-Scoring-ECL-Provisioning-System.git
cd Enterprise-Credit-Risk-Scoring-ECL-Provisioning-System
pip install -r requirements.txt
```

### 2. Start the Production API
```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
# View swagger docs at http://localhost:8000/docs
```

### 3. Launch Dashboard
```bash
python -m streamlit run app/streamlit_app.py
# Open your browser to http://localhost:8501
```

### 4. Execute Full Pipeline (Optional)
*Note: Ensure raw Kaggle CSVs are in the `data/` folder.*
```bash
python data_cleaning.py
python feature_engineering.py
python woe_iv_scorecard.py
python optuna_tuning.py
python modeling.py
python ifrs9_ecl_engine.py
python business_roi_analysis.py
python shap_analysis.py
```

---
*Author: Risk Analytics Team · Dataset: Home Credit Default Risk · Completed: 2026*

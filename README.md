# 🏦 Enterprise Credit Risk Scoring & ECL Provisioning System

> **End-to-end Machine Learning pipeline** dự đoán xác suất vỡ nợ (Probability of Default), tính toán Expected Credit Loss theo **IFRS 9**, và cung cấp actionable insights cho **307,511** hồ sơ vay tiêu dùng — được chuẩn hóa theo Basel II IRB và có khả năng triển khai Production thực tế.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-02569B?logo=lightgbm)](https://lightgbm.readthedocs.io)
[![AUC](https://img.shields.io/badge/OOF%20AUC-0.784-success)](models/model_metrics.json)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2?logo=mlflow)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📊 Dashboard Preview

<p align="center">
  <img src="images/streamlit_dashboard.png" width="100%" alt="Credit Risk Analyst Dashboard">
</p>

> *Dashboard tương tác với Dark Theme, filter động, KPI phản hồi theo filter và 7 biểu đồ phân tích rủi ro chuyên sâu.*

---

## 📑 Mục Lục

1. [Tổng Quan Dự Án](#-tổng-quan-dự-án)
2. [Business Value & ROI](#-business-value--roi)
3. [Model Performance](#-model-performance)
4. [IFRS 9 & Basel II Framework](#-ifrs-9--basel-ii-framework)
5. [Pipeline & Architecture](#-pipeline--architecture)
6. [MLOps & Tracking](#-mlops--tracking)
7. [Model Interpretability (SHAP)](#-model-interpretability-shap)
8. [Interactive Dashboard](#-interactive-dashboard)
9. [Tech Stack](#-tech-stack)
10. [Cách Chạy Dự Án](#️-cách-chạy-dự-án)
11. [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)

---

## 🎯 Tổng Quan Dự Án

| Hạng mục | Chi tiết |
|---|---|
| **Bài toán** | Binary Classification — Dự đoán khả năng vỡ nợ của khách hàng vay |
| **Target variable** | `TARGET` (1 = vỡ nợ, 0 = trả đúng hạn) |
| **Primary metric** | AUC-ROC (chuẩn mực ngành tín dụng) |
| **Secondary metrics** | KS Statistic, Gini Coefficient, Brier Score (Calibration) |
| **Data size** | ~307,511 hồ sơ · 6 bảng quan hệ · 164+ features sau FE |
| **Dataset** | [Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk) |
| **Mục tiêu AUC** | ≥ 0.77 ✅ *Đạt: 0.784* |
| **Target role** | Data Analyst / Data Scientist — Banking & Fintech |

### 🔥 What's New in v2.0 (Enterprise Upgrade)

Đây không chỉ là một notebook Kaggle thông thường. Dự án được nâng cấp lên **Production-Grade** với đầy đủ các tiêu chuẩn của một hệ thống risk scoring doanh nghiệp thực tế:

| # | Upgrade | Ý nghĩa |
|---|---------|---------|
| 1 | **WoE/IV Feature Screening (Basel II)** | Loại bỏ 112/164 features không có giá trị dự báo, giữ lại 52 signals mạnh nhất |
| 2 | **Dynamic LGD & EAD Models** | Không dùng LGD cố định 45% như textbook, mà tính LGD động theo loại tài sản thế chấp |
| 3 | **IFRS 9 Staging + Macro Overlay** | Phân loại Stage 1/2/3, Lifetime PD, và điều chỉnh theo 4 kịch bản kinh tế vĩ mô |
| 4 | **Anti-Leakage Calibration** | Isotonic Regression fit trên Out-Of-Fold predictions thuần túy → Brier Score giảm 64% |
| 5 | **Bayesian Hyperparameter Optimization** | Optuna TPE Sampler với 3-fold CV trong tuning, final training 5-fold |
| 6 | **MLOps Tracking** | MLflow theo dõi AUC, Gini, KS, Brier, AP Score, và CSI drift across experiments |
| 7 | **Business ROI Optimizer** | Tính ngưỡng quyết định tối ưu hóa lợi nhuận ròng cho ngân hàng |
| 8 | **Governance Document** | Model Risk Management (MRM) doc chuẩn ngân hàng |
| 9 | **FastAPI Scoring API** | REST endpoint `/score` và `/score/batch` cho real-time và batch inference |
| 10 | **Interactive BI Dashboard** | Dark-theme Streamlit Dashboard với filter động, KPI phản hồi real-time |

---

## 💰 Business Value & ROI

Bằng cách tối ưu hóa ngưỡng quyết định phê duyệt thay vì dùng 0.5 mặc định, hệ thống tối đa hóa lợi nhuận ròng của ngân hàng:

```
Net Profit  ≈ Revenue từ khách hàng tốt − Credit Loss từ vỡ nợ − Opportunity cost từ từ chối
```

### Kết quả Threshold Optimization
- **Net Profit Projected:** ~$1.8B
- **Approval Rate:** 86.1% (giữ được khách hàng tốt)
- **Precision:** Bắt được phần lớn high-risk mà không over-penalize khách hàng an toàn

<p align="center">
  <img src="reports/ifrs9_ecl_by_stage.png" width="700" alt="ECL by IFRS9 Stage">
</p>

---

## 📈 Model Performance

| Metric | Baseline (LogReg) | LightGBM v2 | Improvement |
|--------|-------------------|-------------|-------------|
| **OOF AUC** | 0.740 | **0.784** | +4.4 pp |
| **Gini Coefficient** | — | **0.567** | — |
| **KS Statistic** | — | **0.426** | — |
| **Brier Score** | 0.1726 | **0.0626** | **-64%** ✅ |
| **Fold Stability** | — | 0.781–0.788 | Rất ổn định |

> **Brier Score** đo độ chính xác xác suất (Calibration). Càng thấp càng tốt. Giảm 64% sau khi áp dụng Isotonic Calibration chống leakage.

<p align="center">
  <img src="reports/calibration_curve.png" width="600" alt="Calibration Curve">
</p>

---

## 🏦 IFRS 9 & Basel II Framework

### Công thức ECL (IFRS 9)
```
ECL = PD × LGD × EAD
```

| Thành phần | Phương pháp |
|-----------|-------------|
| **PD** (Probability of Default) | Calibrated LightGBM 5-fold OOF + Isotonic Regression |
| **LGD** (Loss Given Default) | Segment-based: Cash Loan (45%), Revolving (35%), Collateralized (20%) |
| **EAD** (Exposure at Default) | AMT_CREDIT × Credit Conversion Factor (CCF) |

### IFRS 9 Stage Classification
| Stage | Tiêu chí | Xử lý ECL |
|-------|----------|-----------|
| **Stage 1** (Performing) | PD < 20% | 12-month ECL |
| **Stage 2** (Watch List) | 20% ≤ PD < 50% | Lifetime ECL (Exponential Survival Decay) |
| **Stage 3** (Impaired/NPL) | PD ≥ 50% | Lifetime ECL, full provisioning |

### Macroeconomic Overlay (4 Scenarios)
| Kịch bản | Trọng số | PD Multiplier |
|---------|---------|--------------|
| Optimistic | 20% | 0.85× |
| Base | 50% | 1.00× |
| Adverse | 20% | 1.25× |
| Severe | 10% | 1.60× |

**Portfolio Result:**
- Total EAD: **$184.21B**
- Total ECL: **$68,657M** (Coverage: **37.3%**)
- Portfolio Default Rate: **8.1%**

<p align="center">
  <img src="reports/ifrs9_macro_scenarios.png" width="600" alt="IFRS9 Macro Scenarios">
</p>

---

## 🔬 Detailed Workflow & Business Insights (8-Step Pipeline)

Quy trình chuẩn hóa dữ liệu và xây dựng mô hình rủi ro tín dụng được thiết kế theo chuẩn Enterprise thông qua 8 bước pipeline liên tục, từ Raw Data đến lúc có thể dùng để trích lập dự phòng rủi ro.

### Bước 1: Data Cleaning (`data_cleaning.py`)
- **Workflow**: Quét qua 6 file CSV gốc (train, test, bureau, prev_app, POS, credit_card). Thay thế các missing values bằng median/mode tuỳ thuộc phân phối.
- **Data Insights**: 
  - *Anomaly Detection*: Phát hiện lỗi hệ thống rất phổ biến: `DAYS_EMPLOYED = 365243`. Code đã xử lý đây là cờ (flag) cho "Unemployed" (Thất nghiệp) và map lại giá trị rỗng để tránh nhiễu thuật toán.
  - *Outlier Capping*: Thu nhập (`AMT_INCOME_TOTAL`) được cắt tại p99 để tránh các hồ sơ ảo làm bóp méo mô hình học máy.
- **Output**: Các file được nén lại dưới định dạng `.parquet` giúp giảm 80% dung lượng RAM khi load.

### Bước 2: Feature Engineering (`feature_engineering.py`)
- **Workflow**: Kết hợp dữ liệu từ bảng chính và 5 bảng lịch sử tín dụng (Bureau, Previous Applications, Installments...). Sử dụng hàm aggregate đa luồng để tính toán.
- **Risk Insights (164 Features)**:
  - Khám phá ra **9 Biến Tương Tác (Interaction Features)** mang lại độ chính xác cực cao, điển hình như:
    - `STRESS_AGE_X_CREDIT`: Nhân tố tuổi tác x tỷ lệ đòn bẩy.
    - `DEBT_COVERAGE_RATIO`: Tỷ lệ trả nợ trên tổng thu nhập dư ra.
    - `BUREAU_RISK_COMPOSITE`: Điểm rủi ro tổng hợp dựa trên lịch sử CIC.

### Bước 3: Basel II Feature Screening (`woe_iv_scorecard.py`)
- **Workflow**: Áp dụng Weight of Evidence (WoE) và Information Value (IV) để lọc biến theo chuẩn Basel II Internal Ratings-Based (IRB).
- **Insights**:
  - Trong tổng số 164 features, hệ thống phát hiện ra **112 features là nhiễu (noise) / không có năng lực dự báo (IV < 0.02)**.
  - Điểm EXT_SOURCE (từ các tổ chức chấm điểm tín dụng bên ngoài) có mức IV > 0.30 (Strong Predictors).
  - Kết quả: Giữ lại 52 biến cốt lõi, giúp mô hình chạy nhanh hơn và chống Overfitting hiệu quả.

<p align="center">
  <img src="reports/iv_ranking.png" width="700" alt="IV Ranking">
</p>

### Bước 4: Bayesian Hyperparameter Optimization (`optuna_tuning.py`)
- **Workflow**: Sử dụng Tree-structured Parzen Estimator (TPE) của Optuna để tìm thông số tối ưu cho LightGBM trong không gian 12 chiều (100 trials).
- **Optimization Strategy**: Thay vì dùng 5-fold CV rất tốn thời gian, tuning phase dùng 3-fold CV. Optuna được cấu hình với Hyperband Pruner để tự động cắt đứt các trial kém chất lượng.

### Bước 5: Final Modeling & Calibration (`modeling.py`)
- **Workflow**: Huấn luyện 5 mô hình LightGBM (5-Fold Stratified CV).
- **Model Insights (Anti-Leakage)**:
  - Thông thường các model cho ra xác suất chưa chuẩn (uncalibrated). Code áp dụng **Isotonic Regression**.
  - *Insight Quan Trọng*: Để tránh data leakage, calibration chỉ được fit trên **Out-Of-Fold (OOF) predictions**. Việc này giúp giảm Brier Score tới 64% (0.1726 xuống 0.0626), tức là mô hình phán đoán "tỷ lệ vỡ nợ 15%" thì thực tế đúng là 15% sẽ vỡ nợ.

### Bước 6: IFRS 9 Expected Credit Loss Engine (`lgd_ead_model.py` & `ifrs9_ecl_engine.py`)
- **Workflow**: Chuyển đổi xác suất vỡ nợ (PD) thành Giá trị Lỗ Dự kiến (ECL) thông qua 3 mảng: Staging, Term Structure, và Macro Overlay.
- **Business Insights**:
  - *Dynamic LGD*: LGD không cố định ở 45%. Khách mua hàng tiêu dùng (Consumer Loan) có tài sản thế chấp (Goods) sẽ có LGD = 45%, trong khi vay tiền mặt (Cash Loan) không thế chấp có LGD lên tới 75%.
  - *Macroeconomic Stress Test*: Nếu nền kinh tế rơi vào kịch bản Suy thoái (Severe Scenario), hệ số rủi ro sẽ nhân lên 1.75x, bắt buộc ngân hàng phải trích lập thêm tiền phòng hờ.

### Bước 7: Profit Threshold Optimization (`business_roi_analysis.py`)
- **Workflow**: Quét qua mọi mức độ cắt điểm (Threshold từ 0.01 đến 0.99) để tìm điểm Giao cắt Lợi Nhuận cao nhất.
- **Business Insights**:
  - Theo mặc định, các model ML thường dùng ngưỡng 0.5 để quyết định Đậu/Rớt. Tuy nhiên, phân tích ROI chỉ ra rằng: việc hạ ngưỡng xuống ~0.48 sẽ giúp ngân hàng tối đa hóa Lợi nhuận Ròng lên mức $1.8 Tỷ USD, do tối ưu hóa sự cân bằng giữa Khách Tốt và Rủi Ro Bùng Nợ.

### Bước 8: Explainable AI & Governance (`shap_analysis.py`)
- **Workflow**: Tạo hệ thống Explainability sử dụng SHAP values và Partial Dependence Plots (PDP).
- **Insights**:
  - Không chỉ đưa ra kết quả, hệ thống có khả năng xuất ra *SHAP Waterfall Chart* cho từng khách hàng, trả lời chính xác câu hỏi: "Vì sao anh A bị từ chối tín dụng?" (Ví dụ: do tỷ lệ LTI cao, hoặc lịch sử CIC xấu). Thỏa mãn yêu cầu khắt khe của bộ phận Kiểm toán & Quản trị Rủi ro (MRM).

---

## 🤖 MLOps & Tracking

```bash
# Xem MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Mở http://localhost:5000
```

**Các metrics được track:**
| Metric | Giá trị đạt được |
|--------|-----------------|
| `oof_auc` | 0.7837 |
| `gini` | 0.5673 |
| `ks_statistic` | 0.4255 |
| `brier_score` (raw) | 0.1766 |
| `brier_calibrated` | 0.0626 |
| `psi_score` | 0.0045 (PSI < 0.10 = Ổn định) |

**Monitoring:**
- **PSI (Population Stability Index)**: 0.0045 — Dataset rất ổn định giữa train/test
- **CSI (Characteristic Stability Index)**: Theo dõi drift từng feature

---

## 🧠 Model Interpretability (SHAP)

### Top Features theo SHAP
<p align="center">
  <img src="reports/shap_summary_bar.png" width="650" alt="SHAP Summary Bar">
</p>

### Beeswarm — Tác động từng feature đến xác suất vỡ nợ
<p align="center">
  <img src="reports/shap_beeswarm.png" width="700" alt="SHAP Beeswarm">
</p>

### Waterfall — Giải thích cá nhân hồ sơ High Risk
<p align="center">
  <img src="reports/shap_waterfall_highrisk.png" width="650" alt="SHAP Waterfall High Risk">
</p>

> Mỗi API response có thể đi kèm top 3 yếu tố rủi ro của khách hàng cụ thể, phục vụ việc giải thích quyết định cho underwriter và comply với mô hình quản trị rủi ro.

---

## 📊 Interactive Dashboard

Dashboard được xây dựng bằng **Streamlit** với Dark Theme chuyên nghiệp, gồm 2 tabs:

### Tab 1: Credit Risk Analyst
| Biểu đồ | Loại | Insight |
|---------|------|---------|
| Who Defaults by Employment | Horizontal Bar | Nhóm nghề nào sinh ra nhiều nợ xấu nhất |
| Default by Loan Term | Column Chart | Kỳ hạn nào rủi ro nhất |
| Exposure & Default Rate by Credit Band | Combo (Bar+Line) | EAD theo dải khoản vay vs tỷ lệ vỡ nợ |
| Vintage Analysis — Default Trend | Line + Trendline | Xu hướng nợ xấu theo thời gian |
| PD Distribution — Risk Concentration | Histogram (Gradient) | Phân bổ rủi ro toàn danh mục |
| ECL & Default Rate by DTI | Combo (Bar+Line) | Tác động Debt-to-Income đến ECL |
| IFRS 9 Staging — EAD by Stage | Donut | Phân bổ EAD theo Stage 1/2/3 |

### Tab 2: Insights
- **3 Actionable Insight Cards** (tự động cập nhật theo filter)
- **Stage 1/2/3 Detail Cards** — Số liệu chi tiết từng stage
- **Exposure & ECL by Region Rating** — Combo Bar+Scatter

### Filter Controls
- **Family Status** (Segmented control: Married / Single / Divorced)
- **Region Rating** (Dropdown: 1/2/3)
- **Risk Tier** (Multi-select: Very Low / Low / Medium / High)
- **Loan Term** (Multi-select: 12 / 24 / 36 / 60+)
- **⚡ FILTERED VIEW** badge + Delta indicators khi filter đang bật

---

## 🗄️ Enterprise Data Warehouse (SQL Server) & Strategic Business Use Cases

Hệ thống không chỉ dừng ở việc huấn luyện Model ML, mà còn tự động hóa luồng dữ liệu (Data Pipeline) đẩy kết quả Scoring của 307,511 hồ sơ sang Data Warehouse (SQL Server). Từ đây, các phòng ban nghiệp vụ có thể trực tiếp khai thác dữ liệu để đưa ra các **quyết định chiến lược (Actionable Insights)**.

### 1. Bộ phận Thu hồi nợ (Collections Department)
**Tình huống:** Đợi khách hàng trễ hạn mới gọi điện đòi nợ là phương pháp bị động và tỷ lệ thành công thấp.
**Chiến lược đề xuất (Early Warning System):** 
- Sử dụng SQL Query lọc ra Top những khách hàng có `Xác suất vỡ nợ > 48%` và sắp đến kỳ thanh toán.
- **Hành động:** Phân bổ danh sách này cho đội Telesale/SMS tự động nhắn tin nhắc nhở nhẹ nhàng **trước 3-5 ngày**. Tập trung các nhân sự thu hồi nợ "cứng" (nhiều kinh nghiệm) để theo sát nhóm khách hàng thuộc IFRS9 Stage 2 và Stage 3 nhằm giảm thiểu tỷ lệ nợ xấu (NPL - Non-Performing Loan).
<p align="center">
  <img src="images/sql_query_collections.png" width="100%" alt="Collections Query">
</p>

### 2. Bộ phận Phê duyệt Tín dụng (Underwriting / Credit Approval)
**Tình huống:** Phê duyệt thủ công hàng ngàn hồ sơ mỗi ngày gây quá tải, nghẽn cổ chai (Bottleneck) và cảm tính.
**Chiến lược đề xuất (Auto-Decisioning & XAI):**
- Áp dụng phân luồng tự động (Decision Engine) dựa trên điểm Risk Tier và Probability.
- **Hành động:** 
  - **AUTO-APPROVE:** Cấp tín dụng ngay lập tức (giải ngân trong 5 phút) cho nhóm `Very Low Risk` (Xác suất < 15%) để tăng trải nghiệm khách hàng, cạnh tranh với các Fintech khác.
  - **MANUAL REVIEW:** Đẩy nhóm `Medium Risk` sang cho chuyên viên con người xem xét kỹ hơn giấy tờ (chứng minh thu nhập, tài sản).
  - **AUTO-REJECT:** Từ chối thẳng nhóm `High Risk` (Xác suất > 48%). Đặc biệt, kết hợp dùng biểu đồ giải thích mô hình **SHAP Waterfall** để giải thích lý do từ chối một cách minh bạch cho khách hàng (ví dụ: do tỷ lệ nợ DTI quá cao), giúp tránh rủi ro pháp lý và khiếu nại.
<p align="center">
  <img src="images/sql_query_underwriting.png" width="100%" alt="Underwriting Query">
</p>

### 3. Bộ phận Quản trị Rủi ro & Tài chính (Risk Management & Finance)
**Tình huống:** Trích lập dự phòng rủi ro theo kiểu cào bằng (quy định cũ) gây giam vốn vô ích của Ngân hàng hoặc trích lập thiếu khi có khủng hoảng.
**Chiến lược đề xuất (Dynamic IFRS 9 Provisioning):**
- Theo dõi Dư nợ chịu rủi ro (EAD) và Tiền dự phòng (ECL) trượt theo thời gian thực phân bổ theo 3 Nhóm nợ (Stage 1, 2, 3).
- **Hành động:** 
  - Báo cáo định kỳ lên Ban Giám đốc và Ngân hàng Nhà nước con số trích lập chính xác tới từng đồng dựa trên rủi ro thực tế của danh mục.
  - Thực hiện **Stress Testing**: Chạy mô phỏng kịch bản vĩ mô suy thoái (Severe Scenario) để tính toán trước xem Ngân hàng có cần chuẩn bị thêm vốn đệm (Capital Buffer) hay không.
<p align="center">
  <img src="images/sql_query_ifrs9.png" width="600" alt="IFRS9 Query">
</p>

### 4. Bộ phận Kinh doanh & Tiếp thị (Sale & Marketing)
**Tình huống:** Chạy quảng cáo đại trà tốn kém chi phí CPA nhưng lại thu hút toàn tệp khách hàng "bùng nợ" vào đăng ký.
**Chiến lược đề xuất (Risk-Based Marketing):**
- Đảo ngược bài toán rủi ro thành bài toán tìm kiếm cơ hội kinh doanh.
- **Hành động:** 
  - Lọc tệp "Khách hàng VIP" (Thu nhập cao > 200,000, Rủi ro siêu thấp < 5%) để chạy chiến dịch **Cross-sell / Upsell** (mời mở Thẻ tín dụng hạn mức cao, vay thêm vốn đầu tư). Nhóm này rủi ro thấp nên biên lợi nhuận (Profit Margin) mang lại sẽ cực kỳ lớn.
  - Loại bỏ (Blacklist) ngay nhóm High Risk ra khỏi tệp Custom Audience trên Facebook/Google Ads để không lãng phí ngân sách Marketing vô ích.
<p align="center">
  <img src="images/sql_query_marketing.png" width="100%" alt="Marketing Query">
</p>

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Core** | `LightGBM 4.x`, `scikit-learn 1.3+` |
| **Optimization** | `Optuna 3.3+` (TPE Sampler) |
| **Calibration** | `IsotonicRegression` (sklearn) |
| **Interpretability** | `SHAP 0.43+`, PDP |
| **MLOps** | `MLflow` (SQLite backend) |
| **API** | `FastAPI`, `Uvicorn` |
| **Dashboard** | `Streamlit 1.25+`, `Plotly 5.15+` |
| **Data** | `pandas 2.0+`, `pyarrow`, `numpy` |
| **Compliance** | IFRS 9, Basel II IRB |

---

## ⚙️ Cách Chạy Dự Án

### 1. Cài đặt Dependencies

```bash
git clone https://github.com/baoquocnguyn148/Enterprise-Credit-Risk-Scoring-ECL-Provisioning-System.git
cd Enterprise-Credit-Risk-Scoring-ECL-Provisioning-System
pip install -r requirements.txt
```

### 2. Chạy Dashboard (Nếu đã có data)

Nếu đã có file `data/results_df.parquet` hoặc `data/results_ifrs9.parquet`:

```bash
python -m streamlit run app/streamlit_app.py
# Mở trình duyệt: http://localhost:8501
```

### 3. Chạy Full Pipeline từ đầu

> *Đảm bảo raw Kaggle CSVs đã được đặt trong thư mục `data/`*

```bash
# Bước 1: Làm sạch và join dữ liệu
python src/data_cleaning.py

# Bước 2: Feature Engineering (164 features)
python src/feature_engineering.py

# Bước 3: WoE/IV Screening (Basel II)
python src/woe_iv_scorecard.py

# Bước 4: Bayesian Hyperparameter Tuning (~90-120 phút)
python src/optuna_tuning.py

# Bước 5: Training LightGBM + Calibration + MLflow
python src/modeling.py

# Bước 6: IFRS 9 ECL Calculation
python src/ifrs9_ecl_engine.py

# Bước 7: Business ROI Analysis
python src/business_roi_analysis.py

# Bước 8: SHAP & Interpretability
python src/shap_analysis.py
```

### 4. Khởi động Production API

```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI: http://localhost:8000/docs
```

**API Endpoints:**
```http
POST /score           # Single applicant scoring
POST /score/batch     # Batch scoring (JSON array)
GET  /health          # Health check
GET  /model/info      # Model version & metrics
```

### 5. Xem MLflow Experiment Dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Mở: http://localhost:5000
```

---

## 📁 Cấu Trúc Thư Mục

```text
Enterprise-Credit-Risk-Scoring/
│
├── 📊 app/
│   ├── streamlit_app.py          # Dashboard Streamlit (Dark Theme, Filter, 7 Charts)
│   └── api.py                    # FastAPI REST Scoring API
│
├── 🤖 models/
│   ├── lgbm_fold1-5.pkl          # 5 LightGBM fold models (ensemble)
│   ├── isotonic_calibrator.pkl   # Anti-leakage probability calibrator
│   ├── feature_list.pkl          # 52 selected features (post-WoE/IV)
│   ├── tier_config.json          # Risk Tier thresholds
│   └── model_metrics.json        # Evaluation metrics (AUC, Gini, KS, Brier...)
│
├── 📈 reports/
│   ├── model_governance_doc.md   # MRM Model Validation Document
│   ├── business_insights.md      # Business Insights Report
│   ├── shap_beeswarm.png         # SHAP global importance
│   ├── shap_waterfall_*.png      # SHAP local explanations
│   ├── calibration_curve.png     # Probability calibration chart
│   ├── ifrs9_ecl_by_stage.png   # ECL by IFRS9 Stage
│   ├── iv_ranking.png            # WoE/IV feature ranking
│   └── dashboard_screenshot.png  # Dashboard preview
│
├── 🐍 src/ (Pipeline Scripts)
│   ├── data_cleaning.py          # Data cleaning & joining 6 tables
│   ├── feature_engineering.py    # 164 features + 9 interaction composites
│   ├── woe_iv_scorecard.py       # Basel II WoE/IV feature screening
│   ├── optuna_tuning.py          # Bayesian HPO (Optuna TPE, 100 trials)
│   ├── modeling.py               # LGBM CV, Calibration, MLflow tracking
│   ├── lgd_ead_model.py          # Dynamic LGD & EAD calculation
│   ├── ifrs9_ecl_engine.py       # IFRS 9 Staging, Lifetime PD, Macro overlay
│   ├── business_roi_analysis.py  # Profit curve & threshold optimization
│   └── shap_analysis.py          # SHAP + PDP generation
│
├── 🗄️ schema_sqlserver.sql        # SQL Server schema (8 tables, views, indexes)
├── 📋 requirements.txt
├── 📖 PROJECT_PLAN.md            # Chi tiết kế hoạch từng Phase
└── 📖 README.md
```

---

## 🎓 Key Learnings & Interview Talking Points

Dự án này được thiết kế để **nổi bật trong portfolio và interview** cho vị trí Data Scientist/Analyst trong ngành Banking & Fintech:

| Chủ đề | Điểm thể hiện |
|--------|--------------|
| **Regulatory Compliance** | IFRS 9 Stage 1/2/3, Basel II IRB WoE/IV, MRM Governance |
| **MLOps Maturity** | MLflow, Calibration, PSI/CSI Monitoring, API serving |
| **Business Acumen** | ROI Optimization, Risk Appetite, ECL Provisioning |
| **Technical Depth** | Anti-leakage design, Ensemble CV, Bayesian Tuning |
| **Communication** | SHAP Explainability, BI Dashboard cho C-level |

---

## 📜 License

MIT License — Tự do sử dụng, học tập, và tham khảo với attribution.

---

<p align="center">
  <i>Tác giả: <strong>Bao Quoc Nguyen</strong> · Dataset: Home Credit Default Risk (Kaggle) · Hoàn thành: 2026</i><br>
  <i>Xây dựng với mục tiêu: Production-Ready · Regulation-Compliant · Interview-Ready</i>
</p>

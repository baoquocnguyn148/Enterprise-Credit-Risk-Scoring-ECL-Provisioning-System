"""
modeling.py — Phase 3: Modeling (UPGRADED v2.1)
===============================================
Home Credit Default Risk — Enterprise Grade

Nâng cấp trong phiên bản này:
  [NEW] Tích hợp MLflow tracking (experiment: credit_risk_lgbm)
  [NEW] Bayesian Hyperparameters (load từ optuna best_params.json)
  [NEW] Precision-Recall Curve (AP score)
  [FIX] Calibration Leakage: Fit Isotonic Regression strictly trên OOF predictions
        (không dùng validation set trùng lắp với train set).
  [NEW] CSI (Characteristic Stability Index) cho feature-level monitoring.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os, time, warnings, pickle, re, json
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mlflow
import mlflow.lightgbm

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             precision_recall_curve, average_precision_score)
import lightgbm as lgb

from pathlib import Path
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = str(ROOT_DIR / 'data')
MODELS_DIR  = str(ROOT_DIR / 'models')
REPORTS_DIR = str(ROOT_DIR / 'reports')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def header(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def step(t):   print(f"\n  >> {t}")
def log(t):    print(f"     {t}")

# MLflow setup
mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR}/mlflow.db")
mlflow.set_experiment("credit_risk_lgbm")

# ═══════════════════════════════════════════════════════════════
# STEP 1 — Load Data
# ═══════════════════════════════════════════════════════════════
header("STEP 1 — Loading Feature Data")
t_total = time.time()

train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train_features.parquet'))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, 'test_features.parquet'))

# Clean column names cho LightGBM
train_df = train_df.rename(columns=lambda x: re.sub(r'[^A-Za-z0-9_]+', '_', x))
test_df  = test_df.rename(columns=lambda x: re.sub(r'[^A-Za-z0-9_]+', '_', x))

FEATURES = [c for c in train_df.columns if c not in ['TARGET', 'SK_ID_CURR']]
X = train_df[FEATURES]
y = train_df['TARGET']

log(f"Train Shape: {train_df.shape} | Test Shape: {test_df.shape}")
log(f"Features   : {len(FEATURES)}")
log(f"Default Rate: {y.mean()*100:.2f}%  |  Class Imbalance Ratio: {(y==0).sum()/(y==1).sum():.1f}:1")

step("Saving imputation parameters for production deployment")
imputation_stats = {}
for col in FEATURES:
    col_data = X[col]
    if str(col_data.dtype) == 'bool': col_data = col_data.astype(float)
    try:
        imputation_stats[col] = {
            'median': float(col_data.median()),
            'mean':   float(col_data.mean()),
            'std':    float(col_data.std()),
            'p01':    float(col_data.quantile(0.01)),
            'p99':    float(col_data.quantile(0.99)),
        }
    except Exception:
        imputation_stats[col] = {'median':0.0, 'mean':0.0, 'std':0.0, 'p01':0.0, 'p99':1.0}

with open(os.path.join(MODELS_DIR, 'imputation_stats.json'), 'w') as f:
    json.dump(imputation_stats, f, indent=2)

with mlflow.start_run(run_name="LGBM_5Fold_Calibrated"):
    mlflow.log_param("n_features", len(FEATURES))
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2 — Baseline Logistic Regression
    # ═══════════════════════════════════════════════════════════════
    header("STEP 2 — Baseline Logistic Regression")
    
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    top20 = corr.head(20).index.tolist()
    
    # Đánh giá baseline nhanh bằng 3-Fold (tránh data leakage)
    skf_base = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    base_oof = np.zeros(len(X))
    for tr_idx, vl_idx in skf_base.split(X, y):
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(X.iloc[tr_idx][top20].fillna(0), y.iloc[tr_idx])
        base_oof[vl_idx] = lr.predict_proba(X.iloc[vl_idx][top20].fillna(0))[:, 1]
    
    baseline_auc = roc_auc_score(y, base_oof)
    log(f"Baseline Logistic Regression AUC: {baseline_auc:.4f}")
    mlflow.log_metric("baseline_auc", baseline_auc)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3 — LightGBM 5-Fold Stratified CV
    # ═══════════════════════════════════════════════════════════════
    header("STEP 3 — LightGBM 5-Fold Stratified CV (with Optuna params)")
    
    # Load best params from Optuna if exists
    optuna_file = os.path.join(MODELS_DIR, 'best_params.json')
    if os.path.exists(optuna_file):
        with open(optuna_file) as f:
            best_params = json.load(f)['params']
        log("Loaded best hyperparameters from Optuna tuning.")
        params = best_params
        params.update({'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
                       'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    else:
        log("Optuna params not found. Using defaults.")
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 31,
            'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
    
    mlflow.log_params(params)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(test_df))
    models     = []
    fold_aucs  = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
    
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    
        fold_oof = model.predict_proba(X_vl)[:, 1]
        oof_preds[val_idx] = fold_oof
        test_preds += model.predict_proba(test_df[FEATURES])[:, 1] / skf.n_splits
    
        fold_auc = roc_auc_score(y_vl, fold_oof)
        fold_aucs.append(fold_auc)
        models.append(model)
        log(f"Fold {fold+1} AUC: {fold_auc:.4f}  |  Best iter: {model.best_iteration_}")
    
    oof_auc = roc_auc_score(y, oof_preds)
    log(f"\nOOF AUC  : {oof_auc:.4f}")
    log(f"Mean±Std : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    mlflow.log_metric("oof_auc", oof_auc)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 4 — Evaluation: AUC, KS, Gini, PR-Curve
    # ═══════════════════════════════════════════════════════════════
    header("STEP 4 — Evaluation Metrics & PR-Curve")
    
    fpr, tpr, thresholds = roc_curve(y, oof_preds)
    ks_values  = tpr - fpr
    ks_stat    = ks_values.max()
    ks_idx     = ks_values.argmax()
    optimal_threshold = thresholds[ks_idx]
    gini       = 2 * oof_auc - 1
    brier      = brier_score_loss(y, oof_preds)
    
    precision, recall, pr_thresh = precision_recall_curve(y, oof_preds)
    avg_precision = average_precision_score(y, oof_preds)
    
    log(f"AUC-ROC        : {oof_auc:.4f}")
    log(f"KS Statistic   : {ks_stat:.4f}  (threshold={optimal_threshold:.4f})")
    log(f"Gini Coefficient: {gini:.4f}")
    log(f"Average Precision: {avg_precision:.4f} (PR-AUC)")
    
    mlflow.log_metrics({"ks_stat": ks_stat, "gini": gini, "brier_raw": brier, "ap_score": avg_precision})
    
    # PR Curve Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='#2ECC71', lw=2, label=f'LightGBM (AP={avg_precision:.4f})')
    ax.axhline(y.mean(), color='gray', linestyle='--', label=f'Baseline (AP={y.mean():.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'pr_curve.png'), dpi=150)
    plt.close()
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 5 — Calibration (Anti-Leakage)
    # ═══════════════════════════════════════════════════════════════
    header("STEP 5 — Model Calibration (Isotonic Regression - Calibration/Validation Split)")
    
    step("Splitting OOF into calibration fit (80%) and held-out validation (20%)...")
    # Best practice: fit calibrator on 80% of OOF, measure Brier on held-out 20%
    # This avoids optimistic calibration where we measure on the same data used to fit.
    from sklearn.model_selection import train_test_split
    oof_cal_idx, oof_val_idx = train_test_split(
        np.arange(len(oof_preds)), test_size=0.20, random_state=42, stratify=y
    )
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_preds[oof_cal_idx], y.iloc[oof_cal_idx])
    
    # Measure Brier on the held-out 20% (uncontaminated)
    cal_val_preds  = calibrator.transform(oof_preds[oof_val_idx])
    brier_cal_val  = brier_score_loss(y.iloc[oof_val_idx], cal_val_preds)
    
    # Now transform all OOF and test for downstream use
    cal_oof_preds  = calibrator.transform(oof_preds)
    cal_test_preds = calibrator.transform(test_preds)
    
    frac_pos_raw, mean_pred_raw = calibration_curve(y, oof_preds, n_bins=10)
    frac_pos_cal, mean_pred_cal = calibration_curve(y, cal_oof_preds, n_bins=10)
    
    # Plot calibration curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')
    ax.plot(mean_pred_raw, frac_pos_raw, 's-', color='#E74C3C', lw=2, label='LightGBM (raw)')
    ax.plot(mean_pred_cal, frac_pos_cal, 'o-', color='#2ECC71', lw=2, label='LightGBM + Isotonic')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Unleaked Model Calibration Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'calibration_curve.png'), dpi=150)
    plt.close()
    
    brier_cal = brier_score_loss(y, cal_oof_preds)
    log(f"Brier BEFORE cal (full OOF)     : {brier:.4f}")
    log(f"Brier AFTER  cal (held-out 20%) : {brier_cal_val:.4f}  <-- conservative, uncontaminated estimate")
    log(f"Brier AFTER  cal (full OOF)     : {brier_cal:.4f}  <-- for reference only")
    mlflow.log_metric("brier_calibrated", brier_cal_val)  # log the conservative estimate
    
    # Use calibrated predictions moving forward
    oof_preds  = cal_oof_preds
    test_preds = cal_test_preds
    
    # Recalculate optimal threshold post-calibration
    fpr, tpr, thresholds = roc_curve(y, oof_preds)
    optimal_threshold = thresholds[(tpr - fpr).argmax()]
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 6 — Risk Tier Assignment
    # ═══════════════════════════════════════════════════════════════
    header("STEP 6 — Risk Tier Assignment")
    prob_df = pd.DataFrame({'PRED_PROB': oof_preds, 'TARGET': y.values}).sort_values('PRED_PROB')
    deciles = [prob_df['PRED_PROB'].quantile(q) for q in [0, 0.4, 0.65, 0.85, 1.0]]
    bins    = [0.0, deciles[1], deciles[2], deciles[3], 1.0]
    labels  = ['Very Low', 'Low', 'Medium', 'High']
    
    train_df['PRED_PROB'] = oof_preds
    test_df['PRED_PROB']  = test_preds
    train_df['RISK_TIER'] = pd.cut(train_df['PRED_PROB'], bins=bins, labels=labels, include_lowest=True)
    test_df['RISK_TIER']  = pd.cut(test_df['PRED_PROB'],  bins=bins, labels=labels, include_lowest=True)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 7 — CSI Feature Level Monitoring (NEW)
    # ═══════════════════════════════════════════════════════════════
    header("STEP 7 — CSI (Characteristic Stability Index)")
    
    def compute_csi(train_feat, test_feat, bins=10):
        # Numerical CSI using decile bins
        try:
            _, cuts = pd.qcut(train_feat, q=bins, retbins=True, duplicates='drop')
            cuts[0], cuts[-1] = -np.inf, np.inf
            tr_pct = pd.cut(train_feat, bins=cuts).value_counts(normalize=True).sort_index() + 1e-6
            ts_pct = pd.cut(test_feat,  bins=cuts).value_counts(normalize=True).sort_index() + 1e-6
            return np.sum((ts_pct - tr_pct) * np.log(ts_pct / tr_pct))
        except:
            return 0.0
            
    csi_scores = {}
    for f in top20[:10]: # Check top 10 features
        csi = compute_csi(train_df[f], test_df[f])
        csi_scores[f] = csi
        
    log("Top 10 Feature CSI (Train vs Test):")
    for f, csi in sorted(csi_scores.items(), key=lambda x: x[1], reverse=True):
        log(f"  {f:<40} {csi:.4f} {'(Stable)' if csi < 0.1 else '(Drift!)'}")
        
    # Population PSI
    psi_score = compute_csi(train_df['PRED_PROB'], test_df['PRED_PROB'])
    log(f"\nModel Output PSI : {psi_score:.4f}")
    mlflow.log_metric("psi_score", psi_score)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 8 — Save All Artifacts
    # ═══════════════════════════════════════════════════════════════
    header("STEP 8 — Save All Artifacts")
    
    for i, m in enumerate(models):
        with open(os.path.join(MODELS_DIR, f'lgbm_fold{i+1}.pkl'), 'wb') as f: pickle.dump(m, f)
        
    with open(os.path.join(MODELS_DIR, 'isotonic_calibrator.pkl'), 'wb') as f:
        pickle.dump(calibrator, f)
        
    with open(os.path.join(MODELS_DIR, 'feature_list.pkl'), 'wb') as f:
        pickle.dump(FEATURES, f)
        
    tier_config = {'bins': bins, 'labels': labels, 'optimal_threshold': float(optimal_threshold)}
    with open(os.path.join(MODELS_DIR, 'tier_config.json'), 'w') as f:
        json.dump(tier_config, f, indent=2)
        
    metrics = {
        'oof_auc': float(oof_auc), 'gini': float(gini), 'ks_statistic': float(ks_stat),
        'optimal_threshold': float(optimal_threshold), 'ap_score': float(avg_precision),
        'brier_score': float(brier),
        'brier_calibrated': float(brier_cal_val),  # held-out 20% conservative estimate
        'brier_calibrated_full_oof': float(brier_cal),  # optimistic reference
        'psi_score': float(psi_score),
        'fold_aucs': [float(x) for x in fold_aucs],
        'baseline_auc': float(baseline_auc),
    }
    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    train_df.to_parquet(os.path.join(DATA_DIR, 'results_df.parquet'), index=False)
    
    submission = test_df[['SK_ID_CURR']].copy()
    submission['TARGET'] = test_preds
    submission.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
    
    log(f"Total runtime: {time.time()-t_total:.0f}s")
    print(f"\nPhase 3 Modeling Complete (MLflow Tracked).")

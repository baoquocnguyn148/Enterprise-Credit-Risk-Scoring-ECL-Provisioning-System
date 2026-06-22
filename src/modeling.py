"""
modeling.py — Phase 3: Modeling (UPGRADED v3.0)
===============================================
Home Credit Default Risk — Enterprise Grade

Nâng cấp trong phiên bản này:
  [NEW] Multi-Model Benchmark (LightGBM, XGBoost, CatBoost, RandomForest)
  [NEW] MLflow logging cho tất cả các mô hình.
  [NEW] Báo cáo model_comparison_leaderboard.csv
  [NEW] Calibration Isotonic chỉ trên Winner Model.
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

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             precision_recall_curve, average_precision_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import seaborn as sns

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

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
mlflow.set_experiment("credit_risk_multimodel")

# ═══════════════════════════════════════════════════════════════
# STEP 1 — Load Data
# ═══════════════════════════════════════════════════════════════
header("STEP 1 — Loading Feature Data")
t_total = time.time()

train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train_features.parquet'))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, 'test_features.parquet'))

# Clean column names
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

# ═══════════════════════════════════════════════════════════════
# STEP 2 — Define Models for Benchmark
# ═══════════════════════════════════════════════════════════════
header("STEP 2 — Defining Multi-Model Pipeline")

scale_weight = (y == 0).sum() / (y == 1).sum()

models_dict = {
    'LightGBM': lgb.LGBMClassifier(
        objective='binary', metric='auc', boosting_type='gbdt',
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        scale_pos_weight=scale_weight, random_state=42, n_jobs=-1, verbose=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        scale_pos_weight=scale_weight, random_state=42, n_jobs=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        auto_class_weights='Balanced', eval_metric='AUC',
        random_seed=42, verbose=False, thread_count=-1
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
}

# ═══════════════════════════════════════════════════════════════
# STEP 3 — Multi-Model Training (5-Fold CV)
# ═══════════════════════════════════════════════════════════════
header("STEP 3 — Training Benchmark (5-Fold Stratified CV)")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

benchmark_results = []
trained_models = {}
oof_predictions_all = {}
test_predictions_all = {}

plt.figure(figsize=(10, 8))

for model_name, model in models_dict.items():
    step(f"Training {model_name}...")
    
    with mlflow.start_run(run_name=f"{model_name}_Benchmark"):
        mlflow.log_param("model_type", model_name)
        
        oof_preds  = np.zeros(len(X))
        test_preds = np.zeros(len(test_df))
        fold_models = []
        
        fold_metrics = {'auc': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'spec': []}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
            
            # RandomForest không nhận NaN
            if model_name == 'RandomForest':
                X_tr = X_tr.fillna(0)
                X_vl = X_vl.fillna(0)
                test_df_clean = test_df[FEATURES].fillna(0)
            else:
                test_df_clean = test_df[FEATURES]
                
            if model_name == 'LightGBM':
                model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=[lgb.early_stopping(30, verbose=False)])
            elif model_name == 'XGBoost':
                model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            elif model_name == 'CatBoost':
                model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), early_stopping_rounds=30, verbose=False)
            else:
                model.fit(X_tr, y_tr)
                
            fold_oof = model.predict_proba(X_vl)[:, 1]
            oof_preds[val_idx] = fold_oof
            test_preds += model.predict_proba(test_df_clean)[:, 1] / skf.n_splits
            fold_models.append(model)
            
            # Calculate metrics for THIS fold
            f_auc = roc_auc_score(y_vl, fold_oof)
            
            # Find optimal threshold using Youden's J for the fold
            f_fpr, f_tpr, f_thres = roc_curve(y_vl, fold_oof)
            f_opt_th = f_thres[np.argmax(f_tpr - f_fpr)]
            f_pred_bin = (fold_oof >= f_opt_th).astype(int)
            
            f_acc = accuracy_score(y_vl, f_pred_bin)
            f_prec = precision_score(y_vl, f_pred_bin, zero_division=0)
            f_rec = recall_score(y_vl, f_pred_bin)
            f_f1 = f1_score(y_vl, f_pred_bin)
            
            tn, fp, fn, tp = confusion_matrix(y_vl, f_pred_bin).ravel()
            f_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            fold_metrics['auc'].append(f_auc)
            fold_metrics['acc'].append(f_acc)
            fold_metrics['prec'].append(f_prec)
            fold_metrics['rec'].append(f_rec)
            fold_metrics['f1'].append(f_f1)
            fold_metrics['spec'].append(f_spec)
            
        oof_auc = roc_auc_score(y, oof_preds)
        fpr, tpr, _ = roc_curve(y, oof_preds)
        ks_stat = (tpr - fpr).max() * 100
        gini = (2 * oof_auc - 1)
        brier = brier_score_loss(y, oof_preds)
        
        log(f"{model_name} OOF AUC: {oof_auc:.4f} | KS: {ks_stat:.2f} | Gini: {gini:.4f}")
        
        # Cross Validation Mean & Std
        cv_auc_m, cv_auc_s = np.mean(fold_metrics['auc']), np.std(fold_metrics['auc'])
        cv_acc_m, cv_acc_s = np.mean(fold_metrics['acc']), np.std(fold_metrics['acc'])
        cv_prec_m, cv_prec_s = np.mean(fold_metrics['prec']), np.std(fold_metrics['prec'])
        cv_rec_m, cv_rec_s = np.mean(fold_metrics['rec']), np.std(fold_metrics['rec'])
        cv_f1_m, cv_f1_s = np.mean(fold_metrics['f1']), np.std(fold_metrics['f1'])
        cv_spec_m, cv_spec_s = np.mean(fold_metrics['spec']), np.std(fold_metrics['spec'])
        
        # OOF Confusion Matrix
        f_fpr_all, f_tpr_all, f_thres_all = roc_curve(y, oof_preds)
        oof_opt_th = f_thres_all[np.argmax(f_tpr_all - f_fpr_all)]
        oof_bin = (oof_preds >= oof_opt_th).astype(int)
        cm = confusion_matrix(y, oof_bin)
        
        log(f"CV AUC: {cv_auc_m:.4f}±{cv_auc_s:.4f} | F1: {cv_f1_m:.4f}±{cv_f1_s:.4f}")
        log(f"Confusion Matrix (OOF):\n{cm}")
        
        mlflow.log_metrics({
            "oof_auc": oof_auc,
            "cv_auc_mean": cv_auc_m, "cv_auc_std": cv_auc_s,
            "cv_acc_mean": cv_acc_m, "cv_acc_std": cv_acc_s,
            "cv_prec_mean": cv_prec_m, "cv_prec_std": cv_prec_s,
            "cv_rec_mean": cv_rec_m, "cv_rec_std": cv_rec_s,
            "cv_f1_mean": cv_f1_m, "cv_f1_std": cv_f1_s,
            "cv_spec_mean": cv_spec_m, "cv_spec_std": cv_spec_s,
            "ks_statistic": ks_stat,
            "gini": gini,
            "brier_score": brier
        })
        
        # Draw and save Confusion Matrix plot
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, f'confusion_matrix_{model_name}.png'), dpi=150)
        plt.close()
        
        benchmark_results.append({
            'Model': model_name,
            'OOF_AUC': round(oof_auc, 4),
            'CV_AUC (Mean±Std)': f"{cv_auc_m:.4f} ± {cv_auc_s:.4f}",
            'CV_Accuracy': f"{cv_acc_m:.4f} ± {cv_acc_s:.4f}",
            'CV_Precision': f"{cv_prec_m:.4f} ± {cv_prec_s:.4f}",
            'CV_Recall': f"{cv_rec_m:.4f} ± {cv_rec_s:.4f}",
            'CV_F1_Score': f"{cv_f1_m:.4f} ± {cv_f1_s:.4f}",
            'CV_Specificity': f"{cv_spec_m:.4f} ± {cv_spec_s:.4f}",
            'Gini_Coefficient': round(gini, 4),
            'KS_Statistic': round(ks_stat, 2),
            'Brier_Score': round(brier, 5)
        })
        
        trained_models[model_name] = fold_models
        oof_predictions_all[model_name] = oof_preds
        test_predictions_all[model_name] = test_preds
        
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {oof_auc:.4f})")

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Model ROC Comparison')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'model_benchmark_roc.png'), dpi=150)
plt.close()

# Save leaderboard
leaderboard = pd.DataFrame(benchmark_results).sort_values(by='OOF_AUC', ascending=False)
leaderboard.to_csv(os.path.join(REPORTS_DIR, 'model_comparison_leaderboard.csv'), index=False)
step("Leaderboard saved to reports/model_comparison_leaderboard.csv")
print(leaderboard.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# STEP 4 — Select Best Model & Calibrate
# ═══════════════════════════════════════════════════════════════
header("STEP 4 — Select Winner Model & Calibrate")

winner_name = leaderboard.iloc[0]['Model']
log(f"WINNER MODEL: {winner_name}")

best_oof_preds = oof_predictions_all[winner_name]
best_test_preds = test_predictions_all[winner_name]
best_models = trained_models[winner_name]

# Anti-leakage Calibration
from sklearn.model_selection import train_test_split
oof_cal_idx, oof_val_idx = train_test_split(
    np.arange(len(best_oof_preds)), test_size=0.20, random_state=42, stratify=y
)

calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(best_oof_preds[oof_cal_idx], y.iloc[oof_cal_idx])

cal_oof_preds = calibrator.transform(best_oof_preds)
cal_test_preds = calibrator.transform(best_test_preds)

brier_cal = brier_score_loss(y, cal_oof_preds)
log(f"Calibrated Brier: {brier_cal:.5f}")

# Tính optimal threshold trên calibrated probs
fpr_cal, tpr_cal, thresholds_cal = roc_curve(y, cal_oof_preds)
optimal_threshold = thresholds_cal[(tpr_cal - fpr_cal).argmax()]

# Lưu winner
with open(os.path.join(MODELS_DIR, 'best_production_model.pkl'), 'wb') as f:
    pickle.dump(best_models[0], f)  # Lưu fold 1 làm đại diện hoặc luân phiên
with open(os.path.join(MODELS_DIR, 'isotonic_calibrator.pkl'), 'wb') as f:
    pickle.dump(calibrator, f)
with open(os.path.join(MODELS_DIR, 'feature_list.pkl'), 'wb') as f:
    pickle.dump(FEATURES, f)

# ═══════════════════════════════════════════════════════════════
# STEP 5 — Risk Tier & Export
# ═══════════════════════════════════════════════════════════════
header("STEP 5 — Risk Tier Assignment")
prob_df = pd.DataFrame({'PRED_PROB': cal_oof_preds, 'TARGET': y.values}).sort_values('PRED_PROB')
deciles = [prob_df['PRED_PROB'].quantile(q) for q in [0, 0.4, 0.65, 0.85, 1.0]]
bins    = [0.0, deciles[1], deciles[2], deciles[3], 1.0]
labels  = ['Very Low', 'Low', 'Medium', 'High']

train_df['PRED_PROB'] = cal_oof_preds
test_df['PRED_PROB']  = cal_test_preds
train_df['RISK_TIER'] = pd.cut(train_df['PRED_PROB'], bins=bins, labels=labels, include_lowest=True)
test_df['RISK_TIER']  = pd.cut(test_df['PRED_PROB'],  bins=bins, labels=labels, include_lowest=True)

tier_config = {'bins': bins, 'labels': labels, 'optimal_threshold': float(optimal_threshold), 'winner_model': winner_name}
with open(os.path.join(MODELS_DIR, 'tier_config.json'), 'w') as f:
    json.dump(tier_config, f, indent=2)

train_df.to_parquet(os.path.join(DATA_DIR, 'results_df.parquet'), index=False)

submission = test_df[['SK_ID_CURR']].copy()
submission['TARGET'] = cal_test_preds
submission.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)

log(f"Total runtime: {time.time()-t_total:.0f}s")
print(f"\nPhase 3 Modeling Complete. Multi-Model Benchmark generated.")

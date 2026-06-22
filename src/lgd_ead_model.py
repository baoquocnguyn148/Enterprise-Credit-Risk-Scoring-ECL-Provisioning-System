"""
lgd_ead_model.py — LGD & EAD Modelling (IFRS 9 Components)
===========================================================
IFRS 9 ECL = PD x LGD x EAD

Gap cần fix so với v1:
    LGD = 0.45 (cứng) <- sai với IFRS 9
    EAD = AMT_CREDIT   <- đơn giản hoá quá mức

Module này:
    LGD: segment-based, phụ thuộc contract type + collateral + down payment
    EAD: áp dụng Credit Conversion Factor (CCF) cho revolving facilities

References:
    EBA/GL/2017/16 — Guidelines on PD and LGD estimation
    BCBS d374       — Revisions to IRB approach
    Basel II §296-297 — LGD floors
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(ROOT_DIR / 'data')
REPORTS_DIR = str(ROOT_DIR / 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LGD Table — Basel II floor + Home Credit context
# ─────────────────────────────────────────────────────────────────────────────
# Basel II §296: LGD floor = 45% cho unsecured senior
# Basel II §297: LGD floor = 75% cho subordinated
# Home Credit: Consumer goods = collateral, Cash loans = unsecured
LGD_TABLE = {
    'Cash loans': {
        True:  0.55,   # Có tài sản thế chấp
        False: 0.75,   # Thuần unsecured — worst case
    },
    'Revolving loans': {
        True:  0.70,
        False: 0.85,   # Credit card — rất khó recover
    },
    'Consumer loans': {
        True:  0.45,   # Goods financing — asset backs part of loan
        False: 0.65,
    },
    '_default': 0.65,
}

# CCF (Credit Conversion Factor) — Basel II §82
# Undrawn commitments: CCF = 0.75 cho revolving, 0.40 cho term loans
CCF_TABLE = {
    'Cash loans':      0.40,
    'Revolving loans': 0.75,
    'Consumer loans':  0.40,
    '_default':        0.50,
}


def _get_contract_series(df: pd.DataFrame) -> pd.Series:
    """Detect contract type column (direct or OHE)."""
    if 'NAME_CONTRACT_TYPE' in df.columns:
        return df['NAME_CONTRACT_TYPE'].fillna('_default')
    contract_series = pd.Series('_default', index=df.index)
    for ct in ['Cash loans', 'Revolving loans', 'Consumer loans']:
        col = f"NAME_CONTRACT_TYPE_{ct.replace(' ', '_')}"
        if col in df.columns:
            contract_series[df[col] == 1] = ct
    return contract_series


def simulate_historical_lgd(df: pd.DataFrame) -> pd.Series:
    """
    Ước lượng LGD cá nhân hoá cho từng applicant.

    Logic:
    1. Xác định contract type
    2. Collateral proxy: AMT_GOODS_PRICE / AMT_CREDIT > 0.70
    3. Tra LGD_TABLE -> base_lgd
    4. Điều chỉnh theo down payment ratio
    5. Clip vào [0.35, 0.90]
    """
    lgd              = pd.Series(LGD_TABLE['_default'], index=df.index, dtype=float)
    contract_series  = _get_contract_series(df)
    amt_credit       = df.get('AMT_CREDIT', pd.Series(1, index=df.index)).fillna(1).clip(lower=1)
    amt_goods        = df.get('AMT_GOODS_PRICE', pd.Series(0, index=df.index)).fillna(0)
    has_collateral   = (amt_goods / amt_credit) > 0.70

    for ct, lgd_dict in LGD_TABLE.items():
        if ct == '_default':
            continue
        ct_mask = contract_series == ct
        for coll_bool, lgd_val in lgd_dict.items():
            sub_mask       = ct_mask & (has_collateral == coll_bool)
            lgd[sub_mask]  = lgd_val

    # Down payment adjustment: each 10% DP -> LGD drops 4%
    down_payment = (
        df.get('AMT_GOODS_PRICE', pd.Series(0, index=df.index)).fillna(0) -
        df.get('AMT_CREDIT',      pd.Series(0, index=df.index)).fillna(0)
    ).clip(lower=0)
    dp_ratio       = (down_payment / amt_goods.clip(lower=1)).clip(0, 0.50)
    lgd_adjustment = dp_ratio * 0.40
    lgd            = (lgd - lgd_adjustment).clip(lower=0.35, upper=0.90)
    
    # Add noise to simulate ground truth
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, size=len(lgd))
    lgd = (lgd + noise).clip(0.20, 0.95)
    return lgd


def train_lgd_regressor(df: pd.DataFrame):
    """
    Huấn luyện mô hình Hồi quy LGD trên tập vỡ nợ (TARGET = 1) với 5-Fold CV.
    """
    df['ACTUAL_LGD'] = simulate_historical_lgd(df)
    
    # Lọc hồ sơ vỡ nợ để train LGD
    df_defaults = df[df['TARGET'] == 1].copy()
    if len(df_defaults) < 100:
        df_defaults = df.copy()
        
    features = [c for c in df.columns if c not in ['SK_ID_CURR', 'TARGET', 'PRED_PROB', 'ACTUAL_LGD', 'NAME_CONTRACT_TYPE', 'LGD', 'EAD', 'RISK_TIER']]
    X = df_defaults[features].select_dtypes(include=[np.number]).fillna(0)
    y = df_defaults['ACTUAL_LGD']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {'mae': [], 'mse': [], 'rmse': [], 'r2': []}
    
    print(f"\n{'='*65}")
    print(f"  Training LGD Regressor (LightGBM) - 5-Fold CV")
    print(f"{'='*65}")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6, 
            random_state=42, n_jobs=-1, verbose=-1
        )
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=[lgb.early_stopping(20, verbose=False)])
        
        preds = model.predict(X_vl)
        
        metrics['mae'].append(mean_absolute_error(y_vl, preds))
        metrics['mse'].append(mean_squared_error(y_vl, preds))
        metrics['rmse'].append(np.sqrt(mean_squared_error(y_vl, preds)))
        metrics['r2'].append(r2_score(y_vl, preds))
        
    mae_m, mae_s = np.mean(metrics['mae']), np.std(metrics['mae'])
    mse_m, mse_s = np.mean(metrics['mse']), np.std(metrics['mse'])
    rmse_m, rmse_s = np.mean(metrics['rmse']), np.std(metrics['rmse'])
    r2_m, r2_s = np.mean(metrics['r2']), np.std(metrics['r2'])
    
    print(f"  CV MAE  : {mae_m:.4f} ± {mae_s:.4f}")
    print(f"  CV MSE  : {mse_m:.4f} ± {mse_s:.4f}")
    print(f"  CV RMSE : {rmse_m:.4f} ± {rmse_s:.4f}")
    print(f"  CV R2   : {r2_m:.4f} ± {r2_s:.4f}")
    
    # Train full model
    final_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
    final_model.fit(X, y)
    
    with open(f'{REPORTS_DIR}/lgd_regression_metrics.txt', 'w') as f:
        f.write(f"CV MAE  : {mae_m:.4f} ± {mae_s:.4f}\n")
        f.write(f"CV MSE  : {mse_m:.4f} ± {mse_s:.4f}\n")
        f.write(f"CV RMSE : {rmse_m:.4f} ± {rmse_s:.4f}\n")
        f.write(f"CV R2   : {r2_m:.4f} ± {r2_s:.4f}\n")
        
    return final_model, features


def estimate_ead(df: pd.DataFrame) -> pd.Series:
    """
    EAD = Drawn balance + CCF x Undrawn commitment.

    Với Home Credit:
        Drawn balance  ~ AMT_CREDIT
        Undrawn (revolving) ~ 20% of AMT_CREDIT (conservative assumption)
    """
    amt_credit      = df.get('AMT_CREDIT', pd.Series(0, index=df.index)).fillna(0)
    contract_series = _get_contract_series(df)
    ccf             = contract_series.map(CCF_TABLE).fillna(CCF_TABLE['_default'])

    undrawn                               = pd.Series(0.0, index=df.index)
    revolving_mask                        = contract_series == 'Revolving loans'
    undrawn[revolving_mask]               = amt_credit[revolving_mask] * 0.20

    return (amt_credit + ccf * undrawn).clip(lower=0)


def lgd_sensitivity_analysis(df: pd.DataFrame, lgd_series: pd.Series,
                               ead_series: pd.Series,
                               pd_series: pd.Series) -> pd.DataFrame:
    """
    Sensitivity analysis: ECL thay đổi thế nào khi LGD ± X%.
    Kết quả này đi vào Risk Committee report.
    """
    scenarios = {
        'Optimistic (LGD - 15%)': -0.15,
        'Base Case':               0.00,
        'Adverse   (LGD + 15%)': +0.15,
        'Severe    (LGD + 30%)': +0.30,
    }
    rows     = []
    base_ecl = None

    for label, adj in scenarios.items():
        adj_lgd  = (lgd_series + adj).clip(0.35, 0.95)
        ecl      = pd_series * adj_lgd * ead_series
        total_ecl = ecl.sum() / 1e9
        rows.append({
            'Scenario':        label,
            'Avg LGD':         f"{adj_lgd.mean():.1%}",
            'Total ECL ($B)':  round(total_ecl, 3),
            'Coverage (%)':    round(ecl.sum() / ead_series.sum() * 100, 2),
        })
        if label == 'Base Case':
            base_ecl = total_ecl

    result_df = pd.DataFrame(rows)
    result_df['Delta vs Base ($B)'] = (result_df['Total ECL ($B)'] - base_ecl).round(3)

    print(f"\n{'='*65}")
    print(f"  LGD Sensitivity Analysis — IFRS 9 Scenario Testing")
    print(f"{'='*65}")
    print(result_df.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['#2ECC71', '#3498DB', '#E67E22', '#E74C3C']
    bars = ax.bar(result_df['Scenario'], result_df['Total ECL ($B)'],
                  color=colors, edgecolor='white', width=0.55)
    ax.axhline(y=base_ecl, color='gray', linestyle='--', alpha=0.7, label='Base Case')
    for bar, val in zip(bars, result_df['Total ECL ($B)']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'${val:.2f}B', ha='center', fontsize=9)
    ax.set_ylabel('Total ECL ($B)')
    ax.set_title('LGD Sensitivity — IFRS 9 Stress Test', fontsize=12)
    ax.tick_params(axis='x', rotation=10)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/lgd_sensitivity.png', dpi=150)
    plt.close()
    print(f"  Saved: reports/lgd_sensitivity.png")
    return result_df


if __name__ == '__main__':
    df = pd.read_parquet(f'{DATA_DIR}/results_df.parquet')

    # Defensive Fallback Logic
    req_cols = ['NAME_CONTRACT_TYPE', 'AMT_CREDIT', 'AMT_GOODS_PRICE']
    missing_cols = [c for c in req_cols if c not in df.columns]
    
    if 'NAME_CONTRACT_TYPE' in missing_cols:
        ohe_cols = [c for c in df.columns if c.startswith('NAME_CONTRACT_TYPE_')]
        if len(ohe_cols) > 0:
            missing_cols.remove('NAME_CONTRACT_TYPE')
            
    if missing_cols:
        print(f"  Missing {missing_cols} in results_df, pulling from application_train_clean.parquet...")
        try:
            app_clean = pd.read_parquet(f'{DATA_DIR}/cleaned/application_train_clean.parquet', columns=['SK_ID_CURR'] + missing_cols)
            df = df.merge(app_clean, on='SK_ID_CURR', how='left')
        except Exception as e:
            print(f"  Could not pull fallback data: {e}")

    # Train LGD Regression Model & Predict
    lgd_model, lgd_features = train_lgd_regressor(df)
    X_all = df[lgd_features].select_dtypes(include=[np.number]).fillna(0)
    lgd = pd.Series(lgd_model.predict(X_all), index=df.index).clip(0.1, 1.0)
    
    ead = estimate_ead(df)

    print(f"\n LGD distribution:")
    print(f"  Mean:   {lgd.mean():.3f}")
    print(f"  Median: {lgd.median():.3f}")
    print(f"  Min:    {lgd.min():.3f}")
    print(f"  Max:    {lgd.max():.3f}")
    print(f"\n EAD vs AMT_CREDIT uplift: "
          f"{(ead/df.get('AMT_CREDIT', pd.Series(1, index=df.index)).clip(lower=1)).mean():.3f}x")

    lgd_sensitivity_analysis(df, lgd, ead, df['PRED_PROB'])

    # Save
    df['LGD'] = lgd
    df['EAD'] = ead
    df.to_parquet(f'{DATA_DIR}/results_df.parquet', index=False)
    print(f"\n  Saved LGD + EAD columns -> results_df.parquet")

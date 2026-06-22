"""
business_roi_analysis.py — Phase E: Business ROI & Threshold Optimization
=========================================================================
Trả lời câu hỏi: "Ở threshold nào thì ngân hàng tối đa hoá lợi nhuận?"

Framework:
    Revenue per loan approved (if repaid)    = margin x AMT_CREDIT
    Cost per default (if approved, defaulted) = LGD x AMT_CREDIT
    Cost per false negative (missed default)  = included in above
    Cost per false positive (denied good cust)= opportunity cost

Output:
    reports/roi_threshold_analysis.png  — ROI vs threshold curve
    reports/confusion_profit_matrix.png — Profit breakdown at optimal threshold
    reports/roi_summary.csv
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = str(ROOT_DIR / 'data')
REPORTS_DIR = str(ROOT_DIR / 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Business parameters — tunable per bank's P&L assumptions
# ─────────────────────────────────────────────────────────────────────────────
PARAMS = {
    'net_margin_rate':     0.035,  # 3.5% net interest margin trên khoản vay được duyệt
    'lgd_default':         0.65,   # Loss given default (dùng LGD dynamic nếu có)
    'opex_per_application': 50,    # Chi phí xét duyệt mỗi hồ sơ ($)
    'false_pos_opp_cost':   0.01,  # 1% opportunity cost khi từ chối khách tốt (lost margin)
}


def compute_profit_at_threshold(df: pd.DataFrame, threshold: float,
                                  params: dict = PARAMS) -> dict:
    """
    Tại một threshold nhất định:
        - TP (True Positive) : đúng là default, correctly rejected → tránh loss
        - FP (False Positive): thực ra tốt, bị reject → mất revenue
        - TN (True Negative) : đúng là tốt, approved → earn revenue
        - FN (False Negative): bị miss, approved default → incur loss
    """
    pred     = (df['PRED_PROB'] >= threshold).astype(int)
    actual   = df['TARGET']
    amt      = df.get('AMT_CREDIT', df.get('EAD', pd.Series(100000, index=df.index)))

    if 'LGD' in df.columns:
        lgd = df['LGD']
    else:
        lgd = pd.Series(params['lgd_default'], index=df.index)

    tp_mask = (pred == 1) & (actual == 1)   # correctly flagged as bad
    fp_mask = (pred == 1) & (actual == 0)   # incorrectly flagged (good client denied)
    tn_mask = (pred == 0) & (actual == 0)   # correctly approved good client
    fn_mask = (pred == 0) & (actual == 1)   # missed default — approved bad client

    # Revenue from approved good clients
    revenue_tn = (amt[tn_mask] * params['net_margin_rate']).sum()

    # Opportunity cost from wrongly rejecting good clients
    opp_cost_fp = (amt[fp_mask] * params['false_pos_opp_cost']).sum()

    # Loss from missed defaults
    loss_fn = (amt[fn_mask] * lgd[fn_mask]).sum()

    # Operating cost (applied to all reviewed applications)
    opex = len(df) * params['opex_per_application']

    net_profit = revenue_tn - loss_fn - opp_cost_fp - opex
    approval_rate = (pred == 0).mean()

    return {
        'threshold':     threshold,
        'approval_rate': approval_rate,
        'precision':     tp_mask.sum() / (pred == 1).sum() if (pred == 1).sum() > 0 else 0,
        'recall':        tp_mask.sum() / actual.sum() if actual.sum() > 0 else 0,
        'TP': int(tp_mask.sum()), 'FP': int(fp_mask.sum()),
        'TN': int(tn_mask.sum()), 'FN': int(fn_mask.sum()),
        'revenue_tn_M':    revenue_tn   / 1e6,
        'opp_cost_fp_M':   opp_cost_fp  / 1e6,
        'loss_fn_M':       loss_fn       / 1e6,
        'opex_M':          opex          / 1e6,
        'net_profit_M':    net_profit    / 1e6,
    }


def run_threshold_optimization(df: pd.DataFrame,
                                thresholds: np.ndarray = None) -> pd.DataFrame:
    """Sweep thresholds 0.01 → 0.99 và tìm threshold tối ưu hoá profit."""
    if thresholds is None:
        thresholds = np.arange(0.02, 0.95, 0.01)

    results = [compute_profit_at_threshold(df, t) for t in thresholds]
    df_res  = pd.DataFrame(results)

    best_row    = df_res.loc[df_res['net_profit_M'].idxmax()]
    ks_threshold = df.get('KS_threshold', None)

    print(f"\n{'='*65}")
    print(f"  Business ROI — Threshold Optimization")
    print(f"{'='*65}")
    print(f"  Optimal threshold (max profit):  {best_row['threshold']:.3f}")
    print(f"  Net profit at optimal:           ${best_row['net_profit_M']:.1f}M")
    print(f"  Approval rate at optimal:        {best_row['approval_rate']:.1%}")
    print(f"  Precision at optimal:            {best_row['precision']:.4f}")
    print(f"  Recall at optimal:               {best_row['recall']:.4f}")
    print(f"\n  Revenue (approved good):         ${best_row['revenue_tn_M']:.1f}M")
    print(f"  Opportunity cost (denied good):  -${best_row['opp_cost_fp_M']:.1f}M")
    print(f"  Loan loss (missed defaults):     -${best_row['loss_fn_M']:.1f}M")
    print(f"  Operating expense:               -${best_row['opex_M']:.1f}M")

    # ── Plot 1: Profit + approval rate vs threshold ────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(df_res['threshold'], df_res['net_profit_M'],
            color='#2ECC71', lw=2.5, label='Net Profit ($M)')
    ax.axvline(best_row['threshold'], color='red', linestyle='--',
               label=f"Optimal={best_row['threshold']:.3f}")
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Net Profit ($M)')
    ax.set_title('Net Profit vs Decision Threshold')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df_res['threshold'], df_res['approval_rate'] * 100,
            color='#3498DB', lw=2)
    ax.axvline(best_row['threshold'], color='red', linestyle='--')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Approval Rate (%)')
    ax.set_title('Approval Rate vs Threshold')
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df_res['threshold'], df_res['revenue_tn_M'],   label='Revenue', color='#2ECC71')
    ax.plot(df_res['threshold'], -df_res['loss_fn_M'],     label='Loan Loss', color='#E74C3C')
    ax.plot(df_res['threshold'], -df_res['opp_cost_fp_M'], label='Opp Cost', color='#E67E22')
    ax.axvline(best_row['threshold'], color='red', linestyle='--')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Amount ($M)')
    ax.set_title('P&L Decomposition by Threshold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Confusion profit matrix at optimal threshold
    ax = axes[1, 1]
    cm_data = np.array([
        [best_row['TN'] * PARAMS['net_margin_rate'] * df['AMT_CREDIT'].mean() / 1e6,
         -best_row['FP'] * PARAMS['false_pos_opp_cost'] * df['AMT_CREDIT'].mean() / 1e6],
        [-best_row['FN'] * PARAMS['lgd_default'] * df['AMT_CREDIT'].mean() / 1e6,
         best_row['TP'] * PARAMS['lgd_default'] * df['AMT_CREDIT'].mean() / 1e6],
    ])
    im = ax.imshow(cm_data, cmap='RdYlGn', aspect='auto')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted: Approve', 'Predicted: Reject'])
    ax.set_yticklabels(['Actual: Good', 'Actual: Default'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'${cm_data[i,j]:.0f}M', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if abs(cm_data[i,j]) > abs(cm_data).max()*0.5 else 'black')
    ax.set_title(f'Profit Matrix at Optimal Threshold={best_row["threshold"]:.3f}')
    plt.colorbar(im, ax=ax, label='Profit Impact ($M)')

    plt.suptitle('Business ROI & Decision Threshold Optimization', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/roi_threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: reports/roi_threshold_analysis.png")

    df_res.to_csv(f'{REPORTS_DIR}/roi_summary.csv', index=False)
    print(f"  Saved: reports/roi_summary.csv")

    return df_res, best_row


if __name__ == '__main__':
    df = pd.read_parquet(f'{DATA_DIR}/results_df.parquet')
    if 'PRED_PROB' not in df.columns:
        print("ERROR: results_df.parquet missing PRED_PROB. Run modeling.py first.")
        sys.exit(1)

    df_roi, best = run_threshold_optimization(df)

    print(f"\n  === Summary at Optimal Threshold {best['threshold']:.3f} ===")
    print(f"  Approval Rate:  {best['approval_rate']:.1%}")
    print(f"  Net Profit:     ${best['net_profit_M']:.1f}M")
    print(f"  vs KS threshold ({0.4806:.3f}) profit: "
          f"${compute_profit_at_threshold(df, 0.4806)['net_profit_M']:.1f}M")

    # ── Enrich Data cho Streamlit Dashboard ─────────────────────────
    print("\n  Enriching data for BI Dashboard...")
    try:
        raw_csv_path = os.path.join(DATA_DIR, 'raw', 'application_train.csv')
        if os.path.exists(raw_csv_path):
            raw_df = pd.read_csv(raw_csv_path, usecols=['SK_ID_CURR', 'OCCUPATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'CODE_GENDER'])
            df = df.merge(raw_df, on='SK_ID_CURR', how='left')
            print("    [+] Joined text columns from raw CSV.")
        else:
            print(f"    [-] Raw CSV not found at {raw_csv_path}. Skipping text join.")
    except Exception as e:
        print(f"    [-] Error joining raw CSV: {e}")

    # Gán nhãn Decision_Status
    df['Decision_Status'] = np.where(df['PRED_PROB'] < best['threshold'], 'Approved', 'Rejected')
    
    # Gán Credit_Band
    if 'RISK_TIER' in df.columns:
        df['Credit_Band'] = df['RISK_TIER'].replace({
            'Very Low': 'Excellent',
            'Low': 'Good',
            'Medium': 'Fair',
            'High': 'Poor'
        })
    else:
        df['Credit_Band'] = 'Unknown'

    # Lưu lại
    df.to_parquet(f'{DATA_DIR}/results_df.parquet', index=False)
    print("  [+] Saved enriched data to results_df.parquet.")

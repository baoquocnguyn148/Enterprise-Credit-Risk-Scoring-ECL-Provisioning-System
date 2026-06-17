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
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

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


def estimate_lgd(df: pd.DataFrame) -> pd.Series:
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
    return lgd


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
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = str(ROOT_DIR / 'data')
    df       = pd.read_parquet(f'{DATA_DIR}/results_df.parquet')

    lgd = estimate_lgd(df)
    ead = estimate_ead(df)

    print(f"\n LGD distribution:")
    print(f"  Mean:   {lgd.mean():.3f}")
    print(f"  Median: {lgd.median():.3f}")
    print(f"  Min:    {lgd.min():.3f}")
    print(f"  Max:    {lgd.max():.3f}")
    print(f"\n EAD vs AMT_CREDIT uplift: "
          f"{(ead/df['AMT_CREDIT'].clip(lower=1)).mean():.3f}x")

    lgd_sensitivity_analysis(df, lgd, ead, df['PRED_PROB'])

    # Save
    df['LGD'] = lgd
    df['EAD'] = ead
    df.to_parquet(f'{DATA_DIR}/results_df.parquet', index=False)
    print(f"\n  Saved LGD + EAD columns -> results_df.parquet")

"""
src/time_series_engine.py
=========================
Trend and scenario analytics for the credit-risk portfolio.

Important data note
-------------------
The public Home Credit dataset does not provide a true application calendar,
default date, or observation-by-month performance table. This module therefore
separates two concepts:

1. Observed cross-sectional analytics, such as cohort default rate by age or
   application hour.
2. Forward-looking scenario analytics, such as ECL projection and stressed
   stage migration.

Where a true calendar date is unavailable, the module creates a deterministic
proxy date and marks it with `date_quality = "proxy"`. This keeps the dashboard
honest while still giving analysts useful trend-like views for portfolio review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


MACRO_PATHS = {
    "Optimistic": [0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87],
    "Base": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "Adverse": [1.05, 1.08, 1.10, 1.12, 1.13, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20, 1.20],
    "Severe": [1.10, 1.18, 1.25, 1.32, 1.38, 1.45, 1.50, 1.54, 1.57, 1.60, 1.62, 1.65],
}

MACRO_WEIGHTS = {"Optimistic": 0.20, "Base": 0.50, "Adverse": 0.20, "Severe": 0.10}
MACRO_COLORS = {"Optimistic": "#2ecc71", "Base": "#3498db", "Adverse": "#e67e22", "Severe": "#e74c3c"}

REF_DATE = pd.Timestamp("2016-06-01")
HIGH_RISK_THRESHOLD = 0.48


@dataclass(frozen=True)
class DateDerivation:
    date_col: str
    quality: str
    note: str


def _require_columns(df: pd.DataFrame, cols: Iterable[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{context} requires missing columns: {missing}")


def _series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _ead(df: pd.DataFrame) -> pd.Series:
    return _series(df, "EAD", np.nan).fillna(_series(df, "AMT_CREDIT", 0.0)).clip(lower=0)


def _lgd(df: pd.DataFrame) -> pd.Series:
    return _series(df, "LGD", 0.65).fillna(0.65).clip(0.05, 0.95)


def _pd(df: pd.DataFrame) -> pd.Series:
    return _series(df, "PRED_PROB", 0.10).fillna(0.10).clip(0.001, 0.999)


def _extend_path(path: list[float], months: int) -> list[float]:
    if months <= len(path):
        return path[:months]
    return path + [path[-1]] * (months - len(path))


def derive_application_date(df: pd.DataFrame) -> tuple[pd.Series, DateDerivation]:
    """
    Return the best available application/disbursement date.

    Preferred inputs are explicit date columns. If none exist, use
    `DAYS_ID_PUBLISH` only as a deterministic cohort proxy. Since
    `DAYS_ID_PUBLISH` is measured as days before application, the proxy date is
    `REF_DATE - abs(DAYS_ID_PUBLISH)`, not plus.
    """
    explicit_candidates = [
        "APPLICATION_DATE",
        "APP_DATE",
        "DISBURSEMENT_DATE",
        "ORIGINATION_DATE",
    ]
    for col in explicit_candidates:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.90:
                derivation = DateDerivation(col, "actual", f"Using explicit date column `{col}`.")
                return parsed, derivation

    if "DAYS_ID_PUBLISH" in df.columns:
        days = _series(df, "DAYS_ID_PUBLISH", -365).fillna(-365).abs().clip(0, 1460)
        dates = REF_DATE - pd.to_timedelta(days, unit="D")
        derivation = DateDerivation(
            "DAYS_ID_PUBLISH",
            "proxy",
            "Proxy date from REF_DATE - abs(DAYS_ID_PUBLISH); not a true application date.",
        )
        return dates, derivation

    dates = pd.Series(REF_DATE, index=df.index)
    derivation = DateDerivation("constant_ref_date", "fallback", "No date proxy available; all rows assigned REF_DATE.")
    return dates, derivation


def _classify_stage(pd_series: pd.Series, bureau_bad: pd.Series | None = None, late_ratio: pd.Series | None = None) -> pd.Series:
    """Classify IFRS 9 stage from observable model/behavioral signals only."""
    pd_series = pd_series.fillna(0.10).clip(0.001, 0.999)
    stage = pd.Series(1, index=pd_series.index, dtype=np.int8)

    stage[pd_series > 0.70] = 3
    s2 = pd_series > 0.20
    if bureau_bad is not None:
        s2 = s2 | (pd.to_numeric(bureau_bad, errors="coerce").fillna(0) == 1)
    if late_ratio is not None:
        s2 = s2 | (pd.to_numeric(late_ratio, errors="coerce").fillna(0) > 0.30)
    stage[s2 & (stage < 3)] = 2
    return stage


def _lifetime_pd(pd_12m: pd.Series, term_months: pd.Series) -> pd.Series:
    term_years = term_months.fillna(36).clip(6, 120) / 12.0
    life_pd = 1.0 - (1.0 - pd_12m.clip(0.001, 0.999)) ** term_years
    return life_pd.clip(lower=pd_12m, upper=0.999)


def build_vintage_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build cohort-style vintage data.

    If true application dates are absent, `COHORT` is proxy-based and
    `MOB_BUCKET` is a term bucket, not a real months-on-book curve. The returned
    metadata columns make that distinction explicit for downstream views.
    """
    _require_columns(df, ["SK_ID_CURR", "PRED_PROB"], "build_vintage_data")
    dfc = df.copy()
    app_date, derivation = derive_application_date(dfc)

    dfc["APP_DATE_DERIVED"] = app_date
    dfc["COHORT"] = dfc["APP_DATE_DERIVED"].dt.to_period("Q").astype(str)
    dfc["date_quality"] = derivation.quality
    dfc["date_note"] = derivation.note

    term_months = _series(dfc, "CREDIT_TERM", 24).fillna(24).clip(6, 60)
    dfc["MOB_BUCKET"] = pd.cut(
        term_months,
        bins=[0, 12, 18, 24, 36, 60],
        labels=["0-12m", "12-18m", "18-24m", "24-36m", "36m+"],
        include_lowest=True,
    ).astype(str)
    dfc["mob_quality"] = "term_bucket_proxy"

    default_source = "TARGET" if "TARGET" in dfc.columns else "PRED_PROB"
    vintage = (
        dfc.groupby(["COHORT", "MOB_BUCKET"], observed=True)
        .agg(
            default_rate=(default_source, "mean"),
            count=("SK_ID_CURR", "count"),
            avg_pd=("PRED_PROB", "mean"),
            total_ead=("EAD", "sum") if "EAD" in dfc.columns else ("AMT_CREDIT", "sum"),
        )
        .reset_index()
    )

    vintage["default_rate_pct"] = vintage["default_rate"] * 100
    vintage["avg_pd_pct"] = vintage["avg_pd"] * 100
    vintage["total_ead_B"] = vintage["total_ead"] / 1e9
    vintage["date_quality"] = derivation.quality
    vintage["date_note"] = derivation.note
    vintage["mob_quality"] = "term_bucket_proxy"
    vintage["default_source"] = default_source
    return vintage.sort_values(["COHORT", "MOB_BUCKET"]).reset_index(drop=True)


def build_migration_matrix(df: pd.DataFrame, macro_scalar: float = 1.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a stressed 3x3 IFRS 9 stage migration matrix by EAD.

    The migration is a point-in-time stress simulation: current stage is based
    on observed PD/behavioral signals; T+1 stage is recalculated after applying
    `macro_scalar` to PD.
    """
    _require_columns(df, ["PRED_PROB"], "build_migration_matrix")
    dfc = df.copy()
    ead = _ead(dfc)
    bureau_bad = dfc.get("bureau_bad_debt_flag")
    late_ratio = dfc.get("inst_late_ratio")

    dfc["Stage_T"] = _classify_stage(_pd(dfc), bureau_bad, late_ratio)
    dfc["Stage_T1"] = _classify_stage((_pd(dfc) * macro_scalar).clip(0.001, 0.999), bureau_bad, late_ratio)
    dfc["_EAD"] = ead

    matrix = dfc.groupby(["Stage_T", "Stage_T1"], observed=True)["_EAD"].sum().reset_index()
    matrix["EAD_B"] = matrix["_EAD"] / 1e9
    pivot = matrix.pivot(index="Stage_T", columns="Stage_T1", values="EAD_B").fillna(0.0)

    for s in [1, 2, 3]:
        if s not in pivot.index:
            pivot.loc[s] = 0.0
        if s not in pivot.columns:
            pivot[s] = 0.0

    pivot = pivot.sort_index().sort_index(axis=1)
    row_totals = pivot.sum(axis=1).replace(0, np.nan)
    pivot_pct = pivot.div(row_totals, axis=0).fillna(0.0) * 100
    return pivot, pivot_pct


def build_ecl_projection(df: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """
    Project ECL under macro paths for the next N months.

    This is a deterministic scenario engine rather than an ARIMA/Prophet style
    forecast. It recalculates stage and ECL from stressed PD, LGD, EAD runoff,
    and remaining term assumptions each month.
    """
    _require_columns(df, ["PRED_PROB"], "build_ecl_projection")
    if months < 1:
        raise ValueError("months must be >= 1")

    dfc = df.copy()
    base_pd = _pd(dfc)
    lgd = _lgd(dfc)
    ead0 = _ead(dfc)
    term0 = _series(dfc, "CREDIT_TERM", 36).fillna(36).clip(6, 120)
    bureau_bad = dfc.get("bureau_bad_debt_flag")
    late_ratio = dfc.get("inst_late_ratio")

    current_ecl = dfc["ECL"].sum() if "ECL" in dfc.columns else (base_pd * lgd * ead0).sum()
    current_ead = ead0.sum()
    records: list[dict] = []

    for scenario, path in MACRO_PATHS.items():
        for month, scalar in enumerate(_extend_path(path, months), start=1):
            runoff = 0.985 ** month
            ead_t = ead0 * runoff
            pd_t = (base_pd * scalar).clip(0.001, 0.999)
            term_t = (term0 - month).clip(1, 120)
            life_pd_t = _lifetime_pd(pd_t, term_t)
            stage_t = _classify_stage(pd_t, bureau_bad, late_ratio)

            ecl_t = pd.Series(0.0, index=dfc.index)
            s1, s2, s3 = stage_t == 1, stage_t == 2, stage_t == 3
            ecl_t[s1] = pd_t[s1] * lgd[s1] * ead_t[s1]
            ecl_t[s2] = life_pd_t[s2] * lgd[s2] * ead_t[s2]
            ecl_t[s3] = lgd[s3] * ead_t[s3]

            records.append(
                {
                    "Month": month,
                    "Scenario": scenario,
                    "ECL_B": round(ecl_t.sum() / 1e9, 3),
                    "EAD_B": round(ead_t.sum() / 1e9, 3),
                    "Coverage_pct": round(ecl_t.sum() / max(ead_t.sum(), 1.0) * 100, 3),
                    "Stage1_EAD_B": round(ead_t[s1].sum() / 1e9, 3),
                    "Stage2_EAD_B": round(ead_t[s2].sum() / 1e9, 3),
                    "Stage3_EAD_B": round(ead_t[s3].sum() / 1e9, 3),
                    "Color": MACRO_COLORS[scenario],
                    "Weight": MACRO_WEIGHTS[scenario],
                    "ECL_current_B": current_ecl / 1e9,
                    "EAD_current_B": current_ead / 1e9,
                    "projection_method": "deterministic_pd_lgd_ead_stage_scenario",
                }
            )

    proj = pd.DataFrame(records)
    weighted = (
        proj.assign(weighted_ecl=lambda x: x["ECL_B"] * x["Weight"])
        .groupby("Month", as_index=False)
        .agg(ECL_B=("weighted_ecl", "sum"), EAD_B=("EAD_B", "mean"), Coverage_pct=("Coverage_pct", "mean"))
    )
    weighted["Scenario"] = "Probability Weighted"
    weighted["Color"] = "#bdc3c7"
    weighted["Weight"] = 1.0
    weighted["ECL_current_B"] = current_ecl / 1e9
    weighted["EAD_current_B"] = current_ead / 1e9
    weighted["projection_method"] = "weighted_average_of_macro_scenarios"

    return pd.concat([proj, weighted], ignore_index=True, sort=False)


def build_cohort_default(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute observed default/PD by age band and job-tenure band.

    This is a cohort segmentation view, not a chronological time series.
    """
    _require_columns(df, ["SK_ID_CURR", "PRED_PROB"], "build_cohort_default")
    dfc = df.copy()

    if "AGE_YEARS" in dfc.columns:
        age_years = _series(dfc, "AGE_YEARS", 35)
    else:
        age_years = (-_series(dfc, "DAYS_BIRTH", -35 * 365) / 365)
    dfc["AGE_YEARS_TS"] = age_years.round(0).clip(18, 70)
    dfc["AGE_BAND"] = pd.cut(
        dfc["AGE_YEARS_TS"],
        bins=[17, 25, 35, 45, 55, 70],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
        include_lowest=True,
    ).astype(str)

    if "YEARS_EMPLOYED" in dfc.columns:
        tenure_years = _series(dfc, "YEARS_EMPLOYED", 0)
    else:
        tenure_years = (-_series(dfc, "DAYS_EMPLOYED", 0) / 365)
    dfc["JOB_TENURE_YRS"] = tenure_years.fillna(0).clip(0, 20)
    dfc["TENURE_BAND"] = pd.cut(
        dfc["JOB_TENURE_YRS"],
        bins=[-0.01, 1, 3, 5, 10, 20],
        labels=["<1yr", "1-3yr", "3-5yr", "5-10yr", ">10yr"],
        include_lowest=True,
    ).astype(str)

    default_source = "TARGET" if "TARGET" in dfc.columns else "PRED_PROB"
    cohort = (
        dfc.groupby(["AGE_BAND", "TENURE_BAND"], observed=True)
        .agg(
            default_rate=(default_source, "mean"),
            count=("SK_ID_CURR", "count"),
            total_ead=("EAD", "sum") if "EAD" in dfc.columns else ("AMT_CREDIT", "sum"),
            avg_ecl=("ECL", "mean") if "ECL" in dfc.columns else ("PRED_PROB", "mean"),
            avg_pd=("PRED_PROB", "mean"),
        )
        .reset_index()
    )

    cohort["default_rate_pct"] = cohort["default_rate"] * 100
    cohort["total_ead_B"] = cohort["total_ead"] / 1e9
    cohort["avg_ecl_k"] = cohort["avg_ecl"] / 1e3
    cohort["default_source"] = default_source

    age_order = ["18-25", "26-35", "36-45", "46-55", "56+"]
    cohort["AGE_BAND"] = pd.Categorical(cohort["AGE_BAND"], categories=age_order, ordered=True)
    return cohort.sort_values(["AGE_BAND", "TENURE_BAND"]).reset_index(drop=True)


def build_intraday_risk(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute high-risk rate by application hour and weekday.

    This uses application process fields. It is an operational pattern view,
    useful for staffing/manual-review rules, not a forecast.
    """
    _require_columns(df, ["SK_ID_CURR", "PRED_PROB"], "build_intraday_risk")
    dfc = df.copy()
    dfc["IS_HIGH_RISK"] = (_pd(dfc) > HIGH_RISK_THRESHOLD).astype(int)

    day_order = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_map = dict(zip(day_order, day_labels))

    if "WEEKDAY_APPR_PROCESS_START" in dfc.columns:
        dfc["WEEKDAY"] = dfc["WEEKDAY_APPR_PROCESS_START"].astype(str).str.upper().map(day_map).fillna("Unknown")
    else:
        weekday_cols = [c for c in dfc.columns if c.startswith("WEEKDAY_APPR_PROCESS_START_")]
        if weekday_cols:
            raw = dfc[weekday_cols].idxmax(axis=1).str.replace("WEEKDAY_APPR_PROCESS_START_", "", regex=False)
            dfc["WEEKDAY"] = raw.str.upper().map(day_map).fillna("Unknown")
        else:
            dfc["WEEKDAY"] = "Unknown"

    if "HOUR_APPR_PROCESS_START" not in dfc.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    dfc["HOUR_APPR_PROCESS_START"] = _series(dfc, "HOUR_APPR_PROCESS_START", np.nan).round()
    dfc = dfc[dfc["HOUR_APPR_PROCESS_START"].between(0, 23)]
    if dfc.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pivot = (
        dfc.pivot_table(
            values="IS_HIGH_RISK",
            index="WEEKDAY",
            columns="HOUR_APPR_PROCESS_START",
            aggfunc="mean",
            fill_value=0.0,
        )
        * 100
    )
    pivot = pivot.reindex([d for d in day_labels if d in pivot.index])

    hourly = (
        dfc.groupby("HOUR_APPR_PROCESS_START")
        .agg(high_risk_rate=("IS_HIGH_RISK", "mean"), volume=("SK_ID_CURR", "count"))
        .reset_index()
        .rename(columns={"HOUR_APPR_PROCESS_START": "Hour"})
    )
    hourly["Hour"] = hourly["Hour"].astype(int)
    hourly["high_risk_pct"] = hourly["high_risk_rate"] * 100

    daily = (
        dfc.groupby("WEEKDAY")
        .agg(high_risk_rate=("IS_HIGH_RISK", "mean"), volume=("SK_ID_CURR", "count"))
        .reset_index()
    )
    daily["high_risk_pct"] = daily["high_risk_rate"] * 100
    daily = daily[daily["WEEKDAY"].isin(day_labels)]
    daily["WEEKDAY"] = pd.Categorical(daily["WEEKDAY"], categories=day_labels, ordered=True)
    daily = daily.sort_values("WEEKDAY").reset_index(drop=True)

    return pivot, hourly, daily

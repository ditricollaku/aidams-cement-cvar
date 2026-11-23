"""
Data ingestion, raw Pandera schema validation, GEM equity-share parsing,
NGFS carbon price extraction, and cached Yahoo Finance corporate financials.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openpyxl
import pandas as pd
import yfinance as yf

try:
    import pandera.pandas as pa
except ImportError:
    import pandera as pa

from src import config as cfg


# ---------------------------------------------------------------------------
# 1. RAW SCHEMA VALIDATION WITH PANDERA
# ---------------------------------------------------------------------------
def validate_raw_datasets(
    gem_raw: pd.DataFrame, ct_raw: pd.DataFrame
) -> pd.DataFrame:
    """
    Validate raw GEM and Climate TRACE tables BEFORE cleaning (lazy=True)
    to document real data-quality failures prior to coercion/imputation.
    """
    records = []

    gem_schema = pa.DataFrameSchema(
        {
            "GEM plant ID": pa.Column(str, nullable=False, unique=True),
            "Country/area": pa.Column(str, nullable=False),
            "Operating status": pa.Column(
                str,
                pa.Check.isin(
                    [
                        "operating",
                        "operating pre-retirement",
                        "retired",
                        "mothballed",
                        "announced",
                        "construction",
                        "cancelled",
                        "unknown",
                    ]
                ),
                nullable=False,
            ),
            "Cement capacity (million metric tonnes per annum)": pa.Column(
                float, pa.Check.ge(0), nullable=True, coerce=False
            ),
            "Clinker capacity (million metric tonnes per annum)": pa.Column(
                float, pa.Check.ge(0), nullable=True, coerce=False
            ),
            "Clay calcination": pa.Column(
                str,
                pa.Check.isin(["yes", "no", "unknown"]),
                nullable=True,
            ),
            "Coordinates": pa.Column(str, nullable=False),
        },
        strict=False,
    )

    try:
        gem_schema.validate(gem_raw, lazy=True)
        records.append(
            {
                "dataset": "GEM_raw",
                "column": "ALL",
                "check": "schema_pass",
                "failure_count": 0,
                "sample_failure": "None",
            }
        )
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases
        for (col, chk), grp in fc.groupby(["column", "check"], dropna=False):
            records.append(
                {
                    "dataset": "GEM_raw",
                    "column": str(col),
                    "check": str(chk),
                    "failure_count": int(len(grp)),
                    "sample_failure": str(grp["failure_case"].iloc[0])[:60],
                }
            )

    ct_schema = pa.DataFrameSchema(
        {
            "source_id": pa.Column(int, nullable=False),
            "iso3_country": pa.Column(
                str, pa.Check.str_length(min_value=3, max_value=3), nullable=False
            ),
            "emissions_quantity": pa.Column(float, pa.Check.ge(0), nullable=False),
            "capacity_factor": pa.Column(
                float, pa.Check.in_range(0.0, 1.5), nullable=True
            ),
        },
        strict=False,
    )

    try:
        ct_schema.validate(ct_raw, lazy=True)
        records.append(
            {
                "dataset": "ClimateTRACE_raw",
                "column": "ALL",
                "check": "schema_pass",
                "failure_count": 0,
                "sample_failure": "None",
            }
        )
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases
        for (col, chk), grp in fc.groupby(["column", "check"], dropna=False):
            records.append(
                {
                    "dataset": "ClimateTRACE_raw",
                    "column": str(col),
                    "check": str(chk),
                    "failure_count": int(len(grp)),
                    "sample_failure": str(grp["failure_case"].iloc[0])[:60],
                }
            )

    report = pd.DataFrame(records)
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    report.to_csv(cfg.DATA_PROCESSED / "pandera_validation_report.csv", index=False)
    return report


# ---------------------------------------------------------------------------
# 2. GEM CLEANING & EQUITY-SHARE ENTITY RESOLUTION
# ---------------------------------------------------------------------------
def _parse_equity_share(parent_str: str, regex_pattern: str) -> float:
    """
    Parse equity share (0.0-1.0) for a target company from GEM 'Parent' strings like:
      'Heidelberg Materials AG; Holcim Ltd [50.0%]'
      'CRH PLC [57.0%]; Barro Group Pty Ltd'
    """
    if not isinstance(parent_str, str) or not parent_str.strip():
        return 0.0

    parts = [p.strip() for p in parent_str.split(";") if p.strip()]
    matched_shares: List[float] = []
    known_total = 0.0
    unknown_count = 0
    matched_unknown = 0

    for part in parts:
        m_pct = re.search(r"\[(\d+(?:\.\d+)?)%\]", part)
        is_match = bool(re.search(regex_pattern, part, flags=re.IGNORECASE))
        if m_pct:
            pct = float(m_pct.group(1)) / 100.0
            known_total += pct
            if is_match:
                matched_shares.append(pct)
        else:
            unknown_count += 1
            if is_match:
                matched_unknown += 1

    # Allocate residual share equally across unknown-percentage co-owners
    if matched_unknown > 0 and unknown_count > 0:
        residual = max(0.0, 1.0 - known_total)
        share_per_unknown = residual / unknown_count if residual > 0 else (1.0 / len(parts))
        matched_shares.append(matched_unknown * share_per_unknown)

    return float(min(1.0, sum(matched_shares)))


def clean_gem_data(gem_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean GEM Cement Tracker and map plants to our 18-company universe."""
    df = gem_raw.copy()

    # Parse coordinates into numeric lat / lon
    coords = df["Coordinates"].astype(str).str.split(",", expand=True)
    df["lat"] = pd.to_numeric(coords[0].str.strip(), errors="coerce")
    df["lon"] = pd.to_numeric(coords[1].str.strip(), errors="coerce")

    # Coerce numeric columns
    df["cement_cap_mtpa"] = pd.to_numeric(
        df["Cement capacity (million metric tonnes per annum)"], errors="coerce"
    )
    df["clinker_cap_mtpa"] = pd.to_numeric(
        df["Clinker capacity (million metric tonnes per annum)"], errors="coerce"
    )
    df["prod_2425_mt"] = pd.to_numeric(
        df["2024-2025 Cement production (million metric tonnes)"], errors="coerce"
    )
    df["plant_age_yrs"] = pd.to_numeric(df["Plant age"], errors="coerce")

    # Standardize categorical technology columns
    for col in [
        "Operating status",
        "Plant type",
        "Clinker production method",
        "CCS/CCUS",
        "Alternative fuel for kiln",
        "Clay calcination",
        "Majority cement type",
    ]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df["is_operating"] = df["Operating status"].isin(
        ["operating", "operating pre-retirement"]
    )
    df["has_ccs"] = df["CCS/CCUS"] == "yes"
    df["has_alt_fuel"] = df["Alternative fuel for kiln"] == "yes"

    # Build plant-to-company equity-weighted table (a JV plant can appear under both partners with split shares)
    company_rows = []
    for comp_name, meta in cfg.COMPANY_UNIVERSE.items():
        mask = df["Parent"].fillna("").str.contains(
            meta["gem_regex"], case=False, regex=True
        )
        if meta["exclude_name"]:
            mask &= ~df["Plant name (English)"].fillna("").str.contains(
                meta["exclude_name"], case=False, regex=True
            )
            mask &= ~df["Parent"].fillna("").str.contains(
                meta["exclude_name"], case=False, regex=True
            )
        sub = df.loc[mask].copy()
        sub["company"] = comp_name
        sub["equity_share"] = sub["Parent"].apply(
            lambda s: _parse_equity_share(s, meta["gem_regex"])
        )
        # Keep rows with positive equity share
        sub = sub[sub["equity_share"] > 0].copy()
        sub["eq_cement_cap_mtpa"] = sub["cement_cap_mtpa"] * sub["equity_share"]
        sub["eq_clinker_cap_mtpa"] = sub["clinker_cap_mtpa"] * sub["equity_share"]
        company_rows.append(sub)

    gem_companies = pd.concat(company_rows, ignore_index=True)
    keep_cols = [
        "company",
        "GEM plant ID",
        "Plant name (English)",
        "Parent",
        "equity_share",
        "Country/area",
        "Region",
        "lat",
        "lon",
        "Operating status",
        "is_operating",
        "Plant type",
        "Clinker production method",
        "Majority cement type",
        "Clinker substitution rate",
        "cement_cap_mtpa",
        "clinker_cap_mtpa",
        "eq_cement_cap_mtpa",
        "eq_clinker_cap_mtpa",
        "prod_2425_mt",
        "plant_age_yrs",
        "has_ccs",
        "has_alt_fuel",
    ]
    gem_companies = gem_companies[keep_cols]
    gem_companies.to_parquet(
        cfg.DATA_PROCESSED / "gem_plants_clean.parquet", index=False
    )
    return gem_companies


# ---------------------------------------------------------------------------
# 3. NGFS DOWNSCALED CARBON PRICE EXTRACTION (US$2010 -> US$2026)
# ---------------------------------------------------------------------------
def extract_ngfs_carbon_prices(force_refresh: bool = False) -> pd.DataFrame:
    """
    Read NGFS downscaled REMIND-MAgPIE carbon prices (195 ISO3 countries, annual),
    inflate US$2010 -> US$2026, and cache as a lightweight CSV.
    """
    out_path = cfg.DATA_PROCESSED / "ngfs_carbon_prices.csv"
    if out_path.exists() and not force_refresh:
        return pd.read_csv(out_path)

    wb = openpyxl.load_workbook(
        cfg.NGFS_DOWNSCALED_FILE, read_only=True, data_only=True
    )
    ws = wb["data"]
    rows_iter = ws.iter_rows(values_only=True)
    header = [str(c).strip() if c is not None else "" for c in next(rows_iter)]

    col_idx = {c: i for i, c in enumerate(header)}
    year_cols = [
        (int(c), i)
        for c, i in col_idx.items()
        if c.isdigit() and 2020 <= int(c) <= 2050
    ]

    target_vars = {cfg.NGFS_CARBON_VAR_PRIMARY, cfg.NGFS_CARBON_VAR_FALLBACK}
    target_scenarios = set(cfg.NGFS_SCENARIOS)

    extracted = []
    for row in rows_iter:
        if row[col_idx["Model"]] != cfg.NGFS_MODEL:
            continue
        scen = row[col_idx["Scenario"]]
        if scen not in target_scenarios:
            continue
        var = row[col_idx["Variable"]]
        if var not in target_vars:
            continue
        iso3 = str(row[col_idx["Region"]]).strip()
        for yr, yi in year_cols:
            val = row[yi]
            if val is not None and val != "":
                extracted.append(
                    {
                        "scenario": scen,
                        "iso3_country": iso3,
                        "variable": var,
                        "year": yr,
                        "price_usd2010": float(val),
                        "price_usd2026": float(val) * cfg.CPI_2010_TO_2026,
                    }
                )
    wb.close()

    df_ngfs = pd.DataFrame(extracted)
    # Prefer Price|Carbon|Demand|Industry; fall back to Price|Carbon where missing
    piv = df_ngfs.pivot_table(
        index=["scenario", "iso3_country", "year"],
        columns="variable",
        values="price_usd2026",
        aggfunc="first",
    ).reset_index()

    if cfg.NGFS_CARBON_VAR_PRIMARY in piv.columns:
        piv["carbon_price_usd2026"] = piv[cfg.NGFS_CARBON_VAR_PRIMARY].fillna(
            piv.get(cfg.NGFS_CARBON_VAR_FALLBACK)
        )
    else:
        piv["carbon_price_usd2026"] = piv[cfg.NGFS_CARBON_VAR_FALLBACK]

    result = piv[["scenario", "iso3_country", "year", "carbon_price_usd2026"]].dropna()
    result.to_csv(out_path, index=False)
    return result


# ---------------------------------------------------------------------------
# 4. YAHOO FINANCE CORPORATE FINANCIALS + WACC CACHE
# ---------------------------------------------------------------------------
def _get_fx_to_usd(currency: str, cache: Dict[str, float]) -> float:
    if not currency or currency.upper() == "USD":
        return 1.0
    ccy = currency.upper()
    if ccy not in cache:
        pair = f"{ccy}USD=X"
        info = yf.Ticker(pair).info or {}
        rate = info.get("regularMarketPrice") or info.get("previousClose")
        if rate is None:
            # Try inverse pair USD{ccy}=X
            inv_info = yf.Ticker(f"USD{ccy}=X").info or {}
            inv_rate = inv_info.get("regularMarketPrice") or inv_info.get("previousClose")
            rate = (1.0 / inv_rate) if inv_rate else np.nan
        cache[ccy] = float(rate) if rate is not None else np.nan
        time.sleep(0.25)
    return cache[ccy]


def fetch_or_load_financials(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch corporate financials & valuation metrics from Yahoo Finance for all
    18 companies, convert cleanly to USD, estimate WACC, and cache to CSV.
    """
    if cfg.FINANCIALS_CACHE_FILE.exists() and not force_refresh:
        return pd.read_csv(cfg.FINANCIALS_CACHE_FILE)

    rows = []
    fx_cache: Dict[str, float] = {}

    for comp_name, meta in cfg.COMPANY_UNIVERSE.items():
        symbol = meta["ticker"]
        t = yf.Ticker(symbol)
        info = t.info or {}

        mkt_ccy = info.get("currency") or "USD"
        fin_ccy = info.get("financialCurrency") or mkt_ccy
        fx_mkt = _get_fx_to_usd(mkt_ccy, fx_cache)
        fx_fin = _get_fx_to_usd(fin_ccy, fx_cache)

        rev = info.get("totalRevenue")
        ebitda = info.get("ebitda")
        mcap = info.get("marketCap")
        debt = info.get("totalDebt")
        cash = info.get("totalCash")
        beta = info.get("beta")

        rev_usd = (rev * fx_fin) if (rev is not None and pd.notna(fx_fin)) else np.nan
        ebitda_usd = (ebitda * fx_fin) if (ebitda is not None and pd.notna(fx_fin)) else np.nan
        mcap_usd = (mcap * fx_mkt) if (mcap is not None and pd.notna(fx_mkt)) else np.nan
        debt_usd = (debt * fx_fin) if (debt is not None and pd.notna(fx_fin)) else 0.0
        cash_usd = (cash * fx_fin) if (cash is not None and pd.notna(fx_fin)) else 0.0

        # Compute EV directly in USD to avoid Yahoo dual-currency EV bugs (e.g. HKD vs CNY)
        ev_usd = (mcap_usd + debt_usd - cash_usd) if pd.notna(mcap_usd) else np.nan
        if pd.notna(ev_usd) and ev_usd <= 0:
            ev_usd = mcap_usd

        # Estimate company WACC via CAPM
        if pd.notna(beta) and pd.notna(mcap_usd) and (mcap_usd + debt_usd) > 0:
            beta_clamped = float(np.clip(beta, 0.5, 1.8))
            ke = cfg.RISK_FREE_RATE + beta_clamped * cfg.EQUITY_RISK_PREMIUM
            kd_after_tax = cfg.COST_OF_DEBT_PRETAX * (1.0 - cfg.TAX_RATE_DEFAULT)
            e_wt = mcap_usd / (mcap_usd + debt_usd)
            d_wt = debt_usd / (mcap_usd + debt_usd)
            wacc = float(np.clip(e_wt * ke + d_wt * kd_after_tax, 0.06, 0.13))
        else:
            wacc = cfg.WACC_DEFAULT

        rows.append(
            {
                "company": comp_name,
                "ticker": symbol,
                "hq_region": meta["hq_region"],
                "business_mix": meta["business_mix"],
                "core_example": meta["core_example"],
                "market_currency": mkt_ccy,
                "financial_currency": fin_ccy,
                "fx_mkt_to_usd": fx_mkt,
                "fx_fin_to_usd": fx_fin,
                "revenue_usd_m": rev_usd / 1e6 if pd.notna(rev_usd) else np.nan,
                "ebitda_usd_m": ebitda_usd / 1e6 if pd.notna(ebitda_usd) else np.nan,
                "market_cap_usd_m": mcap_usd / 1e6 if pd.notna(mcap_usd) else np.nan,
                "net_debt_usd_m": (debt_usd - cash_usd) / 1e6,
                "enterprise_value_usd_m": ev_usd / 1e6 if pd.notna(ev_usd) else np.nan,
                "ev_to_ebitda": (ev_usd / ebitda_usd)
                if (pd.notna(ev_usd) and pd.notna(ebitda_usd) and ebitda_usd > 0)
                else np.nan,
                "ebitda_margin": (ebitda_usd / rev_usd)
                if (pd.notna(ebitda_usd) and pd.notna(rev_usd) and rev_usd > 0)
                else np.nan,
                "beta": beta,
                "wacc": wacc,
            }
        )
        time.sleep(0.35)

    df_fin = pd.DataFrame(rows)
    df_fin.to_csv(cfg.FINANCIALS_CACHE_FILE, index=False)
    return df_fin


def run_data_processing(
    refresh_ngfs: bool = False, refresh_financials: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute full Step 3 processing pipeline and return key validated tables."""
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    gem_raw = pd.read_excel(cfg.GEM_FILE, sheet_name="Final data")
    ct_raw = pd.read_csv(cfg.CT_EMISSIONS_FILE, low_memory=False)

    val_report = validate_raw_datasets(gem_raw, ct_raw)
    gem_clean = clean_gem_data(gem_raw)
    ngfs_prices = extract_ngfs_carbon_prices(force_refresh=refresh_ngfs)
    financials = fetch_or_load_financials(force_refresh=refresh_financials)

    return val_report, gem_clean, ngfs_prices, financials


if __name__ == "__main__":
    val_rep, gem_df, ngfs_df, fin_df = run_data_processing()

    print("=== 1. PANDERA RAW VALIDATION REPORT ===")
    print(val_rep.to_string(index=False))

    print("\n=== 2. CLEAN GEM EQUITY-WEIGHTED OPERATING CAPACITY (18 COMPANIES) ===")
    op = gem_df[gem_df["is_operating"]]
    gem_sum = (
        op.groupby("company")
        .agg(
            op_plants=("GEM plant ID", "count"),
            countries=("Country/area", "nunique"),
            eq_cem_mtpa=("eq_cement_cap_mtpa", "sum"),
            eq_clink_mtpa=("eq_clinker_cap_mtpa", "sum"),
            ccs_plants=("has_ccs", "sum"),
        )
        .sort_values("eq_cem_mtpa", ascending=False)
    )
    print(gem_sum.round(2).to_string())

    print("\n=== 3. NGFS CARBON PRICES SAMPLE (US$2026/tCO2, 2030 & 2040) ===")
    sample_iso = ["DEU", "USA", "CHN", "IND", "IDN", "MEX"]
    sub_ngfs = ngfs_df[
        ngfs_df["iso3_country"].isin(sample_iso) & ngfs_df["year"].isin([2026, 2030, 2035, 2040])
    ]
    piv = sub_ngfs.pivot_table(
        index=["scenario", "iso3_country"], columns="year", values="carbon_price_usd2026"
    )
    print(piv.round(1).to_string())

    print("\n=== 4. YAHOO FINANCE FINANCIALS AUDIT (USD MILLIONS) ===")
    cols = [
        "company",
        "ticker",
        "revenue_usd_m",
        "ebitda_usd_m",
        "enterprise_value_usd_m",
        "ev_to_ebitda",
        "ebitda_margin",
        "wacc",
    ]
    print(fin_df[cols].round(2).to_string(index=False))

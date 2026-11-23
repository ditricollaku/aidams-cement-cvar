"""
Climate Value-at-Risk (Climate VaR) & Portfolio Risk Engine (Climate Risk Analyst Core).
Combines World Bank 2026 implemented carbon prices with NGFS scenario pathways,
jurisdictional free-allocation rules, cost pass-through, and CAPM WACC discounting.
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src import config as cfg


EU27_PLUS_ISO3 = {
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "NOR", "ISL", "LIE",
}

WB_JURISDICTION_TO_ISO3 = {
    "Switzerland": "CHE",
    "United Kingdom": "GBR",
    "China": "CHN",
    "Canada": "CAN",
    "Australia": "AUS",
    "Korea, Rep.": "KOR",
    "New Zealand": "NZL",
    "Singapore": "SGP",
    "South Africa": "ZAF",
    "Colombia": "COL",
    "Chile": "CHL",
    "Argentina": "ARG",
    "Mexico": "MEX",
    "Japan": "JPN",
    "Ukraine": "UKR",
    "Serbia": "SRB",
    "Montenegro": "MNE",
    "Kazakhstan": "KAZ",
}


# ---------------------------------------------------------------------------
# 1. WORLD BANK 2026 IMPLEMENTED CARBON PRICE FLOOR
# ---------------------------------------------------------------------------
def load_wb_carbon_price_floors() -> Dict[str, float]:
    """
    Parse 2026 implemented national/regional industrial carbon prices (USD/tCO2)
    from World Bank Carbon Pricing Dashboard CSV. ETS rates take precedence
    over overlapping national non-ETS carbon taxes.
    """
    wb = pd.read_csv(cfg.WB_CARBON_FILE)
    imp = wb[
        (wb["Status"] == "Implemented")
        & (wb["Scope"].isin(["National", "Regional"]))
        & (wb["Sectors"].fillna("").str.contains("industry", case=False))
    ].copy()

    def _extract_usd(val: str) -> float:
        if not isinstance(val, str):
            return 0.0
        m = re.search(r"US\$\s*(\d+(?:\.\d+)?)", val)
        return float(m.group(1)) if m else 0.0

    imp["usd_rate"] = imp["Main Price Rate"].apply(_extract_usd)
    imp = imp[imp["usd_rate"] > 0]

    floors: Dict[str, float] = {}
    for inst_type in ["Carbon tax", "ETS"]:
        for _, row in imp[imp["Type"] == inst_type].iterrows():
            jur = str(row["Jurisdiction"]).strip()
            rate = float(row["usd_rate"])
            if jur == "EU27+":
                for iso in EU27_PLUS_ISO3:
                    floors[iso] = rate
            elif jur in WB_JURISDICTION_TO_ISO3:
                floors[WB_JURISDICTION_TO_ISO3[jur]] = rate
    return floors


# ---------------------------------------------------------------------------
# 2. JURISDICTIONAL FREE-ALLOCATION SCHEDULE
# ---------------------------------------------------------------------------
def get_free_allocation_share(iso3: str, year: int, scenario: str) -> float:
    """
    Return the fraction (0.0-1.0) of cement emissions exempt from carbon pricing
    via free allowances or intensity-based benchmarks in (iso3, year, scenario).
    """
    # EU / UK / Swiss ETS: official CBAM free-allocation phase-out (97.5% -> 0% by 2034)
    if iso3 in cfg.EU_UK_CH_ETS_ISO3:
        if year <= 2026:
            return cfg.EU_FREE_ALLOCATION_SCHEDULE[2026]
        if year >= 2034:
            return 0.0
        return cfg.EU_FREE_ALLOCATION_SCHEDULE[year]

    progress = min(1.0, max(0.0, (year - 2026) / (2040 - 2026)))
    strict_policy = scenario in {"Delayed transition", "Net Zero 2050"}

    # China National ETS: starts at 85% intensity benchmark, tightens under transition scenarios
    if iso3 == "CHN":
        target_end = 0.25 if strict_policy else 0.65
        return cfg.CHINA_FREE_ALLOCATION_BASE - progress * (
            cfg.CHINA_FREE_ALLOCATION_BASE - target_end
        )

    # India CCTS: starts at 90% intensity benchmark, tightens post-2030
    if iso3 == "IND":
        target_end = 0.35 if strict_policy else 0.75
        return cfg.INDIA_FREE_ALLOCATION_BASE - progress * (
            cfg.INDIA_FREE_ALLOCATION_BASE - target_end
        )

    # Other jurisdictions: 50% benchmark tightening to 20%-35% by 2040
    target_end = 0.20 if strict_policy else 0.35
    return cfg.OTHER_FREE_ALLOCATION_BASE - progress * (
        cfg.OTHER_FREE_ALLOCATION_BASE - target_end
    )


# ---------------------------------------------------------------------------
# 3. DETERMINISTIC CLIMATE VaR & ANNUAL CARBON COST TRAJECTORIES
# ---------------------------------------------------------------------------
def compute_climate_var(
    pass_through: float = cfg.PASS_THROUGH_DEFAULT,
    decarb_rate: float = cfg.ANNUAL_DECARB_RATE_BASE,
    emissions_mult: float = 1.0,
    price_mult: float = 1.0,
    wacc_shift: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute annual unpassed carbon costs (2026-2040) and discounted Climate VaR
    (% of Enterprise Value) + EBITDA-at-Risk across all 18 companies and 4 NGFS scenarios.
    """
    ct_ccy = pd.read_csv(cfg.DATA_PROCESSED / "ct_company_country_year.csv")
    geo_df = pd.read_csv(cfg.DATA_PROCESSED / "company_geo_profile.csv")
    reality_df = pd.read_csv(cfg.DATA_PROCESSED / "emissions_reality_check.csv")
    ngfs_df = pd.read_csv(cfg.DATA_PROCESSED / "ngfs_carbon_prices.csv")
    fin_df = pd.read_csv(cfg.FINANCIALS_CACHE_FILE)

    wb_floors = load_wb_carbon_price_floors()

    # Base-year (2025) equity-weighted emissions by (company, iso3_country)
    base_em = ct_ccy[ct_ccy["year"] == cfg.BASE_YEAR][
        ["company", "iso3_country", "eq_emissions_mt"]
    ].copy()
    base_em["base_emissions_mt"] = base_em["eq_emissions_mt"] * emissions_mult

    # Prepare lookup dictionary for NGFS prices: (scenario, iso3, year) -> price_usd2026
    ngfs_sub = ngfs_df[
        ngfs_df["year"].between(cfg.PROJECTION_START, cfg.PROJECTION_END)
    ].copy()
    ngfs_lookup = (
        ngfs_sub.set_index(["scenario", "iso3_country", "year"])["carbon_price_usd2026"]
        .to_dict()
    )
    # Global median price per (scenario, year) as fallback for any small missing ISO3
    ngfs_global_med = (
        ngfs_sub.groupby(["scenario", "year"])["carbon_price_usd2026"]
        .median()
        .to_dict()
    )

    proj_years = list(range(cfg.PROJECTION_START, cfg.PROJECTION_END + 1))

    # Build annual panel across (scenario, company, iso3_country, year)
    panel_rows = []
    for scen in cfg.NGFS_SCENARIOS:
        for _, row in base_em.iterrows():
            comp = row["company"]
            iso = row["iso3_country"]
            e0 = float(row["base_emissions_mt"])
            wb_floor = wb_floors.get(iso, 0.0)

            for yr in proj_years:
                dt = yr - cfg.BASE_YEAR
                em_t = e0 * ((1.0 - decarb_rate) ** dt)

                ngfs_p = ngfs_lookup.get(
                    (scen, iso, yr), ngfs_global_med.get((scen, yr), 0.0)
                )
                eff_price = max(wb_floor, float(ngfs_p)) * price_mult
                free_share = get_free_allocation_share(iso, yr, scen)

                # Unpassed carbon cost in USD millions (Mt * USD/t = USD millions)
                gross_cost_m = em_t * eff_price * (1.0 - free_share)
                net_ebitda_hit_m = gross_cost_m * (1.0 - pass_through)

                panel_rows.append(
                    {
                        "scenario": scen,
                        "company": comp,
                        "iso3_country": iso,
                        "year": yr,
                        "emissions_mt": em_t,
                        "eff_carbon_price_usd": eff_price,
                        "free_alloc_share": free_share,
                        "gross_carbon_cost_usd_m": gross_cost_m,
                        "net_carbon_cost_usd_m": net_ebitda_hit_m,
                    }
                )

    annual_ccy = pd.DataFrame(panel_rows)

    # Aggregate to (scenario, company, year)
    annual_comp = (
        annual_ccy.groupby(["scenario", "company", "year"], as_index=False)
        .agg(
            emissions_mt=("emissions_mt", "sum"),
            gross_carbon_cost_usd_m=("gross_carbon_cost_usd_m", "sum"),
            net_carbon_cost_usd_m=("net_carbon_cost_usd_m", "sum"),
        )
    )

    # Merge company financial denominators and geo/tech metadata
    meta_cols = fin_df.merge(
        geo_df[
            [
                "company",
                "geo_hhi",
                "diversification_tier",
                "ct_countries_2025",
                "ct_emissions_2025_mt",
                "eu_uk_ch_share",
                "china_share",
                "india_share",
            ]
        ],
        on="company",
        how="inner",
    ).merge(
        reality_df[
            [
                "company",
                "gem_eq_cement_cap_mtpa",
                "gem_ccs_plants",
                "old_unabated_cap_share",
            ]
        ],
        on="company",
        how="left",
    )

    annual_comp = annual_comp.merge(meta_cols, on="company", how="inner")
    annual_comp["eff_wacc"] = (annual_comp["wacc"] + wacc_shift).clip(0.04, 0.18)
    annual_comp["disc_factor"] = (1.0 + annual_comp["eff_wacc"]) ** (
        annual_comp["year"] - cfg.BASE_YEAR
    )
    annual_comp["pv_net_cost_usd_m"] = (
        annual_comp["net_carbon_cost_usd_m"] / annual_comp["disc_factor"]
    )
    annual_comp["ebitda_at_risk_pct"] = (
        100.0 * annual_comp["net_carbon_cost_usd_m"] / annual_comp["ebitda_usd_m"]
    )

    # Aggregate across 2026-2040 into Company-Scenario Climate VaR summary
    summary = (
        annual_comp.groupby(["scenario", "company"], as_index=False)
        .agg(
            pv_transition_cost_usd_m=("pv_net_cost_usd_m", "sum"),
            cum_undiscounted_cost_usd_m=("net_carbon_cost_usd_m", "sum"),
            peak_annual_cost_usd_m=("net_carbon_cost_usd_m", "max"),
            peak_ebitda_at_risk_pct=("ebitda_at_risk_pct", "max"),
            mean_ebitda_at_risk_pct=("ebitda_at_risk_pct", "mean"),
        )
    )

    # Extract 2030 milestone EBITDA-at-Risk
    y2030 = annual_comp[annual_comp["year"] == 2030][
        ["scenario", "company", "net_carbon_cost_usd_m", "ebitda_at_risk_pct"]
    ].rename(
        columns={
            "net_carbon_cost_usd_m": "cost_2030_usd_m",
            "ebitda_at_risk_pct": "ebitda_at_risk_2030_pct",
        }
    )

    summary = summary.merge(y2030, on=["scenario", "company"], how="left")
    summary = summary.merge(meta_cols, on="company", how="inner")

    # Core Climate VaR metrics
    # Uncapped ratio kept for diagnostics; headline metric is capped at 100% of EV
    summary["climate_var_ev_pct_uncapped"] = (
        100.0 * summary["pv_transition_cost_usd_m"] / summary["enterprise_value_usd_m"]
    )
    summary["stranded"] = summary["climate_var_ev_pct_uncapped"] >= cfg.CVAR_CAP_PCT
    summary["climate_var_ev_pct"] = summary["climate_var_ev_pct_uncapped"].clip(
        upper=cfg.CVAR_CAP_PCT
    )
    summary["pv_transition_cost_capped_usd_m"] = np.minimum(
        summary["pv_transition_cost_usd_m"], summary["enterprise_value_usd_m"]
    )
    summary["pv_cost_per_t_cap_usd"] = (
        summary["pv_transition_cost_usd_m"] / summary["gem_eq_cement_cap_mtpa"]
    )
    summary["emissions_per_ebitda_t_per_kusd"] = (
        summary["ct_emissions_2025_mt"] * 1e3 / summary["ebitda_usd_m"]
    )

    return summary, annual_comp


# ---------------------------------------------------------------------------
# 4. MULTINATIONAL VS REGIONAL & PORTFOLIO RISK COMPARISON
# ---------------------------------------------------------------------------
def analyze_tier_and_portfolio_risk(
    summary: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare Diversified Multinationals vs Regionally Concentrated producers
    and compute portfolio-level Climate VaR for equal-weighted baskets.
    """
    tier_rows = []
    for scen in cfg.NGFS_SCENARIOS:
        sub = summary[summary["scenario"] == scen]
        r_pearson, p_val = stats.pearsonr(sub["geo_hhi"], sub["climate_var_ev_pct"])
        rho_spearman, sp_pval = stats.spearmanr(sub["geo_hhi"], sub["climate_var_ev_pct"])

        for tier, grp in sub.groupby("diversification_tier"):
            pure_grp = grp[grp["business_mix"] == "pure_play"]
            tier_rows.append(
                {
                    "scenario": scen,
                    "diversification_tier": tier,
                    "n_companies": len(grp),
                    "n_stranded": int(grp["stranded"].sum()),
                    "mean_geo_hhi": grp["geo_hhi"].mean(),
                    "mean_cvar_ev_pct": grp["climate_var_ev_pct"].mean(),
                    "median_cvar_ev_pct": grp["climate_var_ev_pct"].median(),
                    "std_cvar_ev_pct": grp["climate_var_ev_pct"].std(),
                    "pure_play_median_cvar_ev_pct": pure_grp["climate_var_ev_pct"].median(),
                    "median_ebitda_risk_2030_pct": grp["ebitda_at_risk_2030_pct"].median(),
                    "median_peak_ebitda_risk_pct": grp["peak_ebitda_at_risk_pct"].median(),
                    "hhi_cvar_pearson_r": r_pearson,
                    "hhi_cvar_spearman_rho": rho_spearman,
                    "hhi_cvar_spearman_pval": sp_pval,
                }
            )

    tier_comp = pd.DataFrame(tier_rows)

    # Equal-weighted portfolio comparison (Diversified Basket vs Regional Basket vs 50/50 Blended)
    port_rows = []
    for scen in cfg.NGFS_SCENARIOS:
        sub = summary[summary["scenario"] == scen]
        multi = sub[sub["diversification_tier"] == "Diversified Multinational"]
        reg = sub[sub["diversification_tier"] == "Regionally Concentrated"]

        baskets = {
            "Portfolio A: Diversified Multinationals (9 firms)": multi,
            "Portfolio B: Regionally Concentrated (9 firms)": reg,
            "Portfolio C: 50/50 Blended Global Cement (18 firms)": sub,
        }
        for bname, bdf in baskets.items():
            # Equal-weighted portfolio return shock = mean of constituent Climate VaRs
            ew_cvar = float(bdf["climate_var_ev_pct"].mean())
            # Value-weighted (by Enterprise Value) portfolio Climate VaR
            vw_cvar = float(
                bdf["pv_transition_cost_capped_usd_m"].sum()
                / bdf["enterprise_value_usd_m"].sum()
                * 100.0
            )
            # Rank on the uncapped ratio so ties at the 100% cap still pick the worst firm
            worst_firm = bdf.loc[bdf["climate_var_ev_pct_uncapped"].idxmax()]
            port_rows.append(
                {
                    "scenario": scen,
                    "portfolio": bname,
                    "equal_weighted_cvar_ev_pct": ew_cvar,
                    "ev_weighted_cvar_ev_pct": vw_cvar,
                    "cross_firm_cvar_std_pct": float(bdf["climate_var_ev_pct"].std()),
                    "max_single_firm_cvar_pct": float(worst_firm["climate_var_ev_pct"]),
                    "n_stranded": int(bdf["stranded"].sum()),
                    "most_exposed_firm": worst_firm["company"],
                }
            )

    portfolio_comp = pd.DataFrame(port_rows)
    return tier_comp, portfolio_comp


# ---------------------------------------------------------------------------
# 5. CONTROLLING FOR EU CARBON-PRICE EXPOSURE
# ---------------------------------------------------------------------------
def _ols(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Plain OLS with classical standard errors. X must include the intercept column."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    pvals = 2.0 * stats.t.sf(np.abs(beta / se), df=dof)
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return beta, se, pvals, r2


def hhi_eu_control_regression(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether geographic concentration (HHI) explains Climate VaR once EU/UK/CH
    ETS exposure is held fixed. In this sample every diversified firm except Taiheiyo
    has EU exposure and no concentrated firm does, so the raw tier comparison mixes
    "diversification" with "exposure to the world's strictest carbon price".

    Per scenario:
      Model 1 (raw):        CVaR_i = a + b1 * HHI_i + e_i
      Model 2 (controlled): CVaR_i = a + b1 * HHI_i + b2 * EU_share_i + e_i
    Also reports the partial Spearman rho of HHI on CVaR after ranking out EU share.

    Caveats (state in the paper): n = 18, CVaR is capped at 100% (censored),
    and HHI and EU share are strongly correlated, so standard errors are wide.
    """
    rows = []
    for scen in cfg.NGFS_SCENARIOS:
        sub = summary[summary["scenario"] == scen].dropna(
            subset=["climate_var_ev_pct", "geo_hhi", "eu_uk_ch_share"]
        )
        y = sub["climate_var_ev_pct"].to_numpy(float)
        hhi = sub["geo_hhi"].to_numpy(float)
        eu = sub["eu_uk_ch_share"].to_numpy(float)
        ones = np.ones_like(y)

        b1, se1, p1, r2_1 = _ols(y, np.column_stack([ones, hhi]))
        b2, se2, p2, r2_2 = _ols(y, np.column_stack([ones, hhi, eu]))

        # Partial Spearman: correlate rank residuals of CVaR and HHI after regressing on rank(EU)
        ry, rh, re_ = (stats.rankdata(v) for v in (y, hhi, eu))
        Z = np.column_stack([ones, re_])
        res_y = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
        res_h = rh - Z @ np.linalg.lstsq(Z, rh, rcond=None)[0]
        partial_rho = float(np.corrcoef(res_y, res_h)[0, 1])
        n = len(y)
        t_stat = partial_rho * np.sqrt((n - 3) / max(1e-12, 1.0 - partial_rho ** 2))
        partial_p = float(2.0 * stats.t.sf(abs(t_stat), df=n - 3))

        rows.append(
            {
                "scenario": scen,
                "n": n,
                "corr_hhi_eu_share": float(np.corrcoef(hhi, eu)[0, 1]),
                "m1_hhi_coef": b1[1],
                "m1_hhi_pval": p1[1],
                "m1_r2": r2_1,
                "m2_hhi_coef": b2[1],
                "m2_hhi_pval": p2[1],
                "m2_eu_share_coef": b2[2],
                "m2_eu_share_pval": p2[2],
                "m2_r2": r2_2,
                "partial_spearman_hhi_given_eu": partial_rho,
                "partial_spearman_pval": partial_p,
            }
        )
    return pd.DataFrame(rows)


def run_cvar_model() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, annual_comp = compute_climate_var()
    tier_comp, portfolio_comp = analyze_tier_and_portfolio_risk(summary)

    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    summary.to_csv(cfg.DATA_PROCESSED / "company_cvar_summary.csv", index=False)
    annual_comp.to_parquet(cfg.DATA_PROCESSED / "company_cvar_annual.parquet", index=False)
    tier_comp.to_csv(cfg.DATA_PROCESSED / "tier_cvar_comparison.csv", index=False)
    portfolio_comp.to_csv(cfg.DATA_PROCESSED / "portfolio_cvar_comparison.csv", index=False)
    hhi_eu_control_regression(summary).to_csv(
        cfg.DATA_PROCESSED / "hhi_eu_control_regression.csv", index=False
    )

    return summary, annual_comp, tier_comp, portfolio_comp


if __name__ == "__main__":
    summary, annual, tier_df, port_df = run_cvar_model()

    print("=== 1. CLIMATE VaR (% OF ENTERPRISE VALUE) ACROSS ALL 4 NGFS SCENARIOS ===")
    piv_cvar = summary.pivot(
        index=["diversification_tier", "geo_hhi", "company"],
        columns="scenario",
        values="climate_var_ev_pct",
    ).reset_index()
    piv_cvar = piv_cvar.sort_values(["diversification_tier", "geo_hhi"])
    col_order = [
        "diversification_tier",
        "geo_hhi",
        "company",
        "Current Policies",
        "Fragmented World",
        "Delayed transition",
        "Net Zero 2050",
    ]
    print(piv_cvar[col_order].round(2).to_string(index=False))

    print("\n=== 2. PEAK ANNUAL EBITDA-AT-RISK (%) — DELAYED TRANSITION vs NET ZERO 2050 ===")
    sub_eb = summary[
        summary["scenario"].isin(["Delayed transition", "Net Zero 2050"])
    ].pivot(
        index=["diversification_tier", "business_mix", "company"],
        columns="scenario",
        values="peak_ebitda_at_risk_pct",
    ).reset_index()
    print(sub_eb.sort_values("Delayed transition", ascending=False).round(1).to_string(index=False))

    print("\n=== 3. MULTINATIONAL VS REGIONALLY CONCENTRATED GROUP SUMMARY ===")
    t_cols = [
        "scenario",
        "diversification_tier",
        "median_cvar_ev_pct",
        "pure_play_median_cvar_ev_pct",
        "median_ebitda_risk_2030_pct",
        "median_peak_ebitda_risk_pct",
        "hhi_cvar_spearman_rho",
    ]
    print(tier_df[t_cols].round(2).to_string(index=False))

    print("\n=== 4. PORTFOLIO-LEVEL CLIMATE VaR COMPARISON ===")
    print(port_df.round(2).to_string(index=False))

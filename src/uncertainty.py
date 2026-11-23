"""
Uncertainty Quantification, Back-Testing, Monte Carlo Simulation,
One-at-a-Time Sensitivity, and IPCC Calibrated Likelihoods.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src import config as cfg
from src.cvar_model import after_tax_factor, effective_discount_rate


# ---------------------------------------------------------------------------
# 1. IPCC CALIBRATED LIKELIHOOD SCALE
# ---------------------------------------------------------------------------
def ipcc_likelihood_label(prob: float) -> str:
    """Map a probability in [0, 1] to official IPCC calibrated language."""
    if prob >= 0.99:
        return "virtually certain (99-100%)"
    if prob >= 0.90:
        return "very likely (90-100%)"
    if prob >= 0.66:
        return "likely (66-100%)"
    if prob >= 0.33:
        return "about as likely as not (33-66%)"
    if prob >= 0.10:
        return "unlikely (10-33%)"
    if prob >= 0.01:
        return "very unlikely (1-10%)"
    return "exceptionally unlikely (0-1%)"


# ---------------------------------------------------------------------------
# 2. HISTORICAL BACK-TEST WITH STUDENT-t PREDICTION INTERVALS
# ---------------------------------------------------------------------------
def fit_predict(
    train: pd.DataFrame, horizon: int, model: str = "linear", level: float = 0.90
) -> pd.DataFrame:
    """
    Fit OLS ('linear' or 'loglinear') on `train` (columns: year, value) and return
    a DataFrame for the next `horizon` years with columns: year, mean, lo, hi.
    """
    df_tr = train.sort_values("year").reset_index(drop=True)
    y_raw = df_tr["value"].to_numpy(dtype=float)
    yrs = df_tr["year"].to_numpy(dtype=float)
    n = len(df_tr)
    if n < 3:
        raise ValueError("Need at least 3 observations for Student-t prediction interval.")

    if model == "loglinear":
        if np.any(y_raw <= 0):
            raise ValueError("loglinear requires strictly positive values.")
        y = np.log(y_raw)
    elif model == "linear":
        y = y_raw
    else:
        raise ValueError(f"Unknown model: {model}")

    t_mean = yrs.mean()
    t = yrs - t_mean
    X = np.column_stack([np.ones(n), t])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    dof = n - 2
    s_err = float(np.sqrt(np.sum(resid ** 2) / dof))
    xtx_inv = np.linalg.inv(X.T @ X)

    alpha = 1.0 - level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=dof))

    last_yr = int(yrs[-1])
    fut_yrs = np.arange(last_yr + 1, last_yr + horizon + 1, dtype=float)
    t_fut = fut_yrs - t_mean
    X0 = np.column_stack([np.ones(horizon), t_fut])
    y_hat = X0 @ beta

    leverage = np.einsum("ij,jk,ik->i", X0, xtx_inv, X0)
    pred_se = s_err * np.sqrt(1.0 + leverage)
    lo = y_hat - t_crit * pred_se
    hi = y_hat + t_crit * pred_se

    if model == "loglinear":
        mean_out = np.exp(y_hat)
        lo_out = np.exp(lo)
        hi_out = np.exp(hi)
    else:
        mean_out, lo_out, hi_out = y_hat, lo, hi

    return pd.DataFrame(
        {
            "year": fut_yrs.astype(int),
            "mean": mean_out,
            "lo": lo_out,
            "hi": hi_out,
        }
    )


def run_historical_backtest(level: float = 0.90) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Back-test 90% prediction intervals on complete Climate TRACE years (2021-2025):
      - Aggregate 18-company total series (train: 2021-2023, holdout: 2024-2025)
      - Panel-level across all 18 companies (18 firms x 2 holdout years = 36 tests)
    """
    ct_ccy = pd.read_csv(cfg.DATA_PROCESSED / "ct_company_country_year.csv")
    ct_full = ct_ccy[ct_ccy["year"].between(2021, cfg.BASE_YEAR)].copy()

    # 1) Aggregate sample series
    agg_series = (
        ct_full.groupby("year", as_index=False)["eq_emissions_mt"]
        .sum()
        .rename(columns={"eq_emissions_mt": "value"})
    )
    train_agg = agg_series[agg_series["year"] <= 2023]
    test_agg = agg_series[agg_series["year"] >= 2024]

    agg_rows = []
    for mod in ["linear", "loglinear"]:
        pred = fit_predict(train_agg, horizon=2, model=mod, level=level).merge(
            test_agg, on="year"
        )
        pred["model"] = mod
        pred["inside_90"] = (pred["value"] >= pred["lo"]) & (pred["value"] <= pred["hi"])
        pred["abs_pct_err"] = 100.0 * (pred["mean"] - pred["value"]).abs() / pred["value"]
        agg_rows.append(pred)
    agg_bt = pd.concat(agg_rows, ignore_index=True)

    # 2) Company-level panel back-test (36 company-year holdouts)
    comp_yr = (
        ct_full.groupby(["company", "year"], as_index=False)["eq_emissions_mt"]
        .sum()
        .rename(columns={"eq_emissions_mt": "value"})
    )
    panel_records = []
    for mod in ["linear", "loglinear"]:
        hits = 0
        total = 0
        for comp, grp in comp_yr.groupby("company"):
            tr = grp[grp["year"] <= 2023]
            te = grp[grp["year"] >= 2024]
            if len(tr) < 3 or len(te) == 0:
                continue
            pr = fit_predict(tr, horizon=2, model=mod, level=level).merge(te, on="year")
            inside = (pr["value"] >= pr["lo"]) & (pr["value"] <= pr["hi"])
            hits += int(inside.sum())
            total += int(len(pr))
        misses = total - hits
        # Binomial tail P(X >= misses) under nominal 10% miss rate
        binom_p = float(stats.binom.sf(misses - 1, total, 1.0 - level)) if misses > 0 else 1.0
        panel_records.append(
            {
                "model": mod,
                "n_holdout_points": total,
                "hits_inside_90": hits,
                "misses": misses,
                "empirical_coverage_pct": 100.0 * hits / total if total > 0 else np.nan,
                "binomial_tail_p_value": binom_p,
            }
        )
    panel_bt = pd.DataFrame(panel_records)

    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    agg_bt.to_csv(cfg.DATA_PROCESSED / "backtest_aggregate.csv", index=False)
    panel_bt.to_csv(cfg.DATA_PROCESSED / "backtest_panel.csv", index=False)
    return agg_bt, panel_bt


# ---------------------------------------------------------------------------
# 3. VECTORIZED MONTE CARLO SIMULATION
# ---------------------------------------------------------------------------
def _triangular_ppf(u: np.ndarray, low: float, mode: float, high: float) -> np.ndarray:
    """Inverse CDF of the triangular distribution Tri(low, mode, high)."""
    fc = (mode - low) / (high - low)
    left = low + np.sqrt(u * (high - low) * (mode - low))
    right = high - np.sqrt((1.0 - u) * (high - low) * (high - mode))
    return np.where(u < fc, left, right)


def _correlated_triangular(
    rng: np.random.Generator, n_draws: int, n_units: int, params: Tuple[float, float, float], rho: float
) -> np.ndarray:
    """
    Draw an (n_draws, n_units) matrix of triangular variates with pairwise correlation ~rho
    across units, via a one-factor Gaussian copula: z = sqrt(rho)*common + sqrt(1-rho)*idiosyncratic.
    rho = 1 reproduces a single shared draw; rho = 0 gives independent draws.
    """
    common = rng.standard_normal((n_draws, 1))
    idio = rng.standard_normal((n_draws, n_units))
    z = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
    return _triangular_ppf(stats.norm.cdf(z), *params)


def run_monte_carlo_cvar(
    n_draws: int = cfg.MC_DRAWS,
    seed: int = cfg.RANDOM_SEED,
    vary_pass_through: bool = True,
    vary_emissions: bool = True,
    vary_price: bool = True,
    vary_wacc: bool = True,
) -> Dict[str, object]:
    """
    Vectorized Monte Carlo simulation across N draws, 4 NGFS scenarios,
    18 companies, 4 carbon-pricing blocs and 15 projection years (2026-2040).

    Dependence structure (see config):
      - pass-through and emissions multipliers: one draw PER FIRM (correlation MC_FIRM_CORR)
      - carbon-price multiplier: one draw PER PRICING BLOC (correlation MC_PRICE_BLOC_CORR),
        applied to each firm's costs in that bloc
      - WACC shock: systematic, common to all firms (market-wide rate risk)
    Firm-specific draws are what make cross-firm comparisons such as
    P(regional > multinational) informative; with a single shared draw they are fixed by construction.
    """
    annual = pd.read_parquet(cfg.DATA_PROCESSED / "company_cvar_annual.parquet")
    summary_det = pd.read_csv(cfg.DATA_PROCESSED / "company_cvar_summary.csv")

    scenarios = cfg.NGFS_SCENARIOS
    companies = summary_det[summary_det["scenario"] == scenarios[0]]["company"].tolist()
    years = sorted(annual["year"].unique().tolist())
    blocs = cfg.PRICE_BLOCS
    bloc_cols = [f"gross_cost_{b}_usd_m" for b in blocs]

    S, C, T, B = len(scenarios), len(companies), len(years), len(blocs)

    # (S, C, T, B) tensor of baseline gross carbon costs (USD millions) split by pricing bloc
    gross_tensor = np.zeros((S, C, T, B), dtype=float)
    for si, scen in enumerate(scenarios):
        sub_s = annual[annual["scenario"] == scen].set_index(["company", "year"])
        for bi, col in enumerate(bloc_cols):
            piv = sub_s[col].unstack("year").reindex(index=companies, columns=years).fillna(0.0)
            gross_tensor[si, :, :, bi] = piv.to_numpy(dtype=float)

    meta_c = (
        summary_det[summary_det["scenario"] == scenarios[0]]
        .set_index("company")
        .reindex(companies)
    )
    ev_c = meta_c["enterprise_value_usd_m"].to_numpy(dtype=float)
    ebitda_c = meta_c["ebitda_usd_m"].to_numpy(dtype=float)
    wacc_c = meta_c["wacc"].to_numpy(dtype=float)
    tier_c = meta_c["diversification_tier"].to_numpy(dtype=str)
    dt_t = np.array([yr - cfg.BASE_YEAR for yr in years], dtype=float)

    rng = np.random.default_rng(seed)

    if vary_pass_through:
        pt = _correlated_triangular(rng, n_draws, C, cfg.MC_PASS_THROUGH_TRIANGULAR, cfg.MC_FIRM_CORR)
    else:
        pt = np.full((n_draws, C), cfg.PASS_THROUGH_DEFAULT)

    if vary_emissions:
        em_mult = _correlated_triangular(rng, n_draws, C, cfg.MC_EMISSIONS_MULT_TRIANGULAR, cfg.MC_FIRM_CORR)
    else:
        em_mult = np.ones((n_draws, C))

    if vary_price:
        pr_mult = _correlated_triangular(rng, n_draws, B, cfg.MC_CARBON_PRICE_MULT_TRIANGULAR, cfg.MC_PRICE_BLOC_CORR)
    else:
        pr_mult = np.ones((n_draws, B))

    if vary_wacc:
        w_shift = rng.normal(0.0, cfg.MC_WACC_STD, size=n_draws)
    else:
        w_shift = np.zeros(n_draws)

    # Effective (real, clipped) discount factors: shape (N, C, T) -- same convention as cvar_model
    eff_wacc = effective_discount_rate(wacc_c[None, :], w_shift[:, None])
    inv_disc = 1.0 / (1.0 + eff_wacc[:, :, None]) ** dt_t[None, None, :]

    # Firm-level multiplier on unpassed cost (pre-tax): shape (N, C)
    firm_mult = em_mult * (1.0 - pt)

    # PV of after-tax unpassed cost: shape (N, S, C)
    pv_cost = (
        np.einsum("sctb,nct,nb->nsc", gross_tensor, inv_disc, pr_mult, optimize=True)
        * firm_mult[:, None, :]
        * after_tax_factor()
    )

    # Climate VaR (% of EV): capped at 100% of EV (limited liability); uncapped kept for P(stranded)
    cvar_uncapped = 100.0 * pv_cost / ev_c[None, None, :]
    cvar_draws = np.minimum(cvar_uncapped, cfg.CVAR_CAP_PCT)

    # 2030 EBITDA-at-Risk (%, pre-tax): shape (N, S, C)
    idx_2030 = years.index(2030)
    cost_2030 = np.einsum("scb,nb->nsc", gross_tensor[:, :, idx_2030, :], pr_mult) * firm_mult[:, None, :]
    ebitda_2030_draws = 100.0 * cost_2030 / ebitda_c[None, None, :]

    # Annual universe-wide unpassed (pre-tax) cost, USD millions: shape (N, S, T)
    annual_total_cost_draws = np.einsum(
        "sctb,nb,nc->nst", gross_tensor, pr_mult, firm_mult, optimize=True
    )

    return {
        "scenarios": scenarios,
        "companies": companies,
        "years": years,
        "tiers": tier_c,
        "ev_c": ev_c,
        "cvar_draws": cvar_draws,
        "cvar_uncapped": cvar_uncapped,
        "ebitda_2030_draws": ebitda_2030_draws,
        "annual_total_cost_draws": annual_total_cost_draws,
    }


# ---------------------------------------------------------------------------
# 4. SUMMARIZE MONTE CARLO, OAT SENSITIVITY, SCENARIO GAPS & IPCC METRICS
# ---------------------------------------------------------------------------
def build_uncertainty_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mc = run_monte_carlo_cvar()
    scenarios = mc["scenarios"]
    companies = mc["companies"]
    tiers = mc["tiers"]
    cvar_draws = mc["cvar_draws"]  # (N, S, C)

    # 1) Company-Scenario 90% intervals
    comp_rows = []
    for si, scen in enumerate(scenarios):
        for ci, comp in enumerate(companies):
            arr = cvar_draws[:, si, ci]
            p_stranded = float((mc["cvar_uncapped"][:, si, ci] >= cfg.CVAR_CAP_PCT).mean())
            eb_2030 = mc["ebitda_2030_draws"][:, si, ci]
            med = float(np.median(arr))
            p05, p25, p75, p95 = [float(x) for x in np.percentile(arr, [5, 25, 75, 95])]
            width = p95 - p05
            rel_hw = width / (2.0 * med) if med > 0 else np.nan
            comp_rows.append(
                {
                    "scenario": scen,
                    "company": comp,
                    "diversification_tier": tiers[ci],
                    "cvar_median_pct": med,
                    "cvar_p05_pct": p05,
                    "cvar_p25_pct": p25,
                    "cvar_p75_pct": p75,
                    "cvar_p95_pct": p95,
                    "interval_width_90_pct": width,
                    "rel_half_width": rel_hw,
                    "prob_stranded": p_stranded,
                    "ebitda_risk_2030_p05_pct": float(np.percentile(eb_2030, 5)),
                    "ebitda_risk_2030_median_pct": float(np.median(eb_2030)),
                    "ebitda_risk_2030_p95_pct": float(np.percentile(eb_2030, 95)),
                }
            )
    mc_summary = pd.DataFrame(comp_rows)

    # 2) One-at-a-Time (OAT) Sensitivity on Delayed Transition (Portfolio-level mean VaR)
    s_idx = scenarios.index("Delayed transition")
    oat_configs = [
        ("All Inputs Varying (Joint MC)", dict(vary_pass_through=True, vary_emissions=True, vary_price=True, vary_wacc=True)),
        ("Pass-Through Rate Only (θ ~ Tri[0.15, 0.40, 0.65], per firm)", dict(vary_pass_through=True, vary_emissions=False, vary_price=False, vary_wacc=False)),
        ("Carbon Price Multiplier Only (Tri[0.75, 1.00, 1.30], per bloc)", dict(vary_pass_through=False, vary_emissions=False, vary_price=True, vary_wacc=False)),
        ("Emissions / Utilization Only (Tri[0.85, 1.00, 1.20], per firm)", dict(vary_pass_through=False, vary_emissions=True, vary_price=False, vary_wacc=False)),
        ("Discount Rate / WACC Only (N(0, 1.2%), common shock)", dict(vary_pass_through=False, vary_emissions=False, vary_price=False, vary_wacc=True)),
    ]
    oat_rows = []
    for label, flags in oat_configs:
        res = run_monte_carlo_cvar(**flags)
        port_ew = res["cvar_draws"][:, s_idx, :].mean(axis=1)
        med = float(np.median(port_ew))
        p05, p95 = [float(x) for x in np.percentile(port_ew, [5, 95])]
        width = p95 - p05
        oat_rows.append(
            {
                "driver": label,
                "median_portfolio_cvar_pct": med,
                "p05_pct": p05,
                "p95_pct": p95,
                "interval_width_90_pct": width,
                "rel_half_width": width / (2.0 * med),
            }
        )
    oat_df = pd.DataFrame(oat_rows)

    # 3) Scenario Gap vs Within-Scenario Interval Width + IPCC Exceedance Probabilities
    multi_mask = tiers == "Diversified Multinational"
    reg_mask = tiers == "Regionally Concentrated"

    scen_rows = []
    for si, scen in enumerate(scenarios):
        ew_all = cvar_draws[:, si, :].mean(axis=1)
        ew_multi = cvar_draws[:, si, multi_mask].mean(axis=1)
        ew_reg = cvar_draws[:, si, reg_mask].mean(axis=1)

        med_all = float(np.median(ew_all))
        p05_all, p95_all = [float(x) for x in np.percentile(ew_all, [5, 95])]

        p_reg_gt_multi = float((ew_reg > ew_multi).mean())
        p_all_gt_25 = float((ew_all > 25.0).mean())

        scen_rows.append(
            {
                "scenario": scen,
                "global_ew_cvar_median_pct": med_all,
                "global_ew_p05_pct": p05_all,
                "global_ew_p95_pct": p95_all,
                "within_scen_90_width_pct": p95_all - p05_all,
                "multinational_ew_median_pct": float(np.median(ew_multi)),
                "regional_ew_median_pct": float(np.median(ew_reg)),
                "prob_regional_gt_multinational": p_reg_gt_multi,
                "ipcc_regional_gt_multinational": ipcc_likelihood_label(p_reg_gt_multi),
                "prob_global_cvar_gt_25pct": p_all_gt_25,
                "ipcc_global_cvar_gt_25pct": ipcc_likelihood_label(p_all_gt_25),
            }
        )
    scen_comp = pd.DataFrame(scen_rows)

    # 4) Annual Fan-Chart Time Series (2021-2025 history + 2026-2040 MC bands per scenario)
    ct_ccy = pd.read_csv(cfg.DATA_PROCESSED / "ct_company_country_year.csv")
    hist = (
        ct_ccy[ct_ccy["year"].between(2021, cfg.BASE_YEAR)]
        .groupby("year", as_index=False)["eq_emissions_mt"]
        .sum()
    )
    fan_rows = []
    for _, r in hist.iterrows():
        fan_rows.append(
            {
                "scenario": "Historical (Climate TRACE)",
                "year": int(r["year"]),
                "sample_emissions_mt": float(r["eq_emissions_mt"]),
                "cost_median_usd_m": 0.0,
                "cost_p05_usd_m": 0.0,
                "cost_p25_usd_m": 0.0,
                "cost_p75_usd_m": 0.0,
                "cost_p95_usd_m": 0.0,
            }
        )

    ann_cost_draws = mc["annual_total_cost_draws"]  # (N, S, T)
    for si, scen in enumerate(scenarios):
        for ti, yr in enumerate(mc["years"]):
            arr = ann_cost_draws[:, si, ti]
            p05, p25, med, p75, p95 = [
                float(x) for x in np.percentile(arr, [5, 25, 50, 75, 95])
            ]
            fan_rows.append(
                {
                    "scenario": scen,
                    "year": int(yr),
                    "sample_emissions_mt": np.nan,
                    "cost_median_usd_m": med,
                    "cost_p05_usd_m": p05,
                    "cost_p25_usd_m": p25,
                    "cost_p75_usd_m": p75,
                    "cost_p95_usd_m": p95,
                }
            )
    fan_df = pd.DataFrame(fan_rows)

    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    mc_summary.to_csv(cfg.DATA_PROCESSED / "mc_company_cvar_intervals.csv", index=False)
    oat_df.to_csv(cfg.DATA_PROCESSED / "oat_sensitivity.csv", index=False)
    scen_comp.to_csv(cfg.DATA_PROCESSED / "scenario_uncertainty_summary.csv", index=False)
    fan_df.to_csv(cfg.DATA_PROCESSED / "fan_chart_data.csv", index=False)

    return mc_summary, oat_df, scen_comp, fan_df


if __name__ == "__main__":
    agg_bt, panel_bt = run_historical_backtest()
    mc_sum, oat_df, scen_df, fan_df = build_uncertainty_tables()

    print("=== 1. HISTORICAL BACK-TEST: AGGREGATE SAMPLE EMISSIONS (2024-2025 HOLDOUT) ===")
    print(agg_bt[["model", "year", "value", "mean", "lo", "hi", "inside_90", "abs_pct_err"]].round(2).to_string(index=False))

    print("\n=== 2. HISTORICAL BACK-TEST: 18-COMPANY PANEL COVERAGE (36 HOLDOUT POINTS) ===")
    print(panel_bt.round(3).to_string(index=False))

    print("\n=== 3. MONTE CARLO 90% INTERVALS FOR FLAGSHIP COMPANIES (DELAYED TRANSITION) ===")
    flagships = ["Holcim", "Heidelberg Materials", "CRH", "Anhui Conch", "UltraTech Cement", "Semen Indonesia"]
    sub_mc = mc_sum[
        (mc_sum["scenario"] == "Delayed transition") & (mc_sum["company"].isin(flagships))
    ]
    print(sub_mc[["company", "diversification_tier", "cvar_p05_pct", "cvar_median_pct", "cvar_p95_pct", "rel_half_width"]].round(2).to_string(index=False))

    print("\n=== 4. ONE-AT-A-TIME (OAT) SENSITIVITY (DELAYED TRANSITION PORTFOLIO VaR) ===")
    print(oat_df.round(2).to_string(index=False))

    print("\n=== 5. SCENARIO GAPS vs WITHIN-SCENARIO WIDTH & IPCC PROBABILITIES ===")
    cols_s = [
        "scenario",
        "global_ew_cvar_median_pct",
        "within_scen_90_width_pct",
        "multinational_ew_median_pct",
        "regional_ew_median_pct",
        "prob_regional_gt_multinational",
        "ipcc_regional_gt_multinational",
    ]
    print(scen_df[cols_s].round(2).to_string(index=False))

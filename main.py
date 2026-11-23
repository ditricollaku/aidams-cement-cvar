"""
Single executable entry point reproducing the entire Cement Climate VaR pipeline:
  1. Raw schema validation, GEM cleaning, NGFS extraction, and cached financials
  2. Equity-weighted Climate TRACE emissions panel & GEM bottom-up reconciliation
  3. Climate VaR (% EV), EBITDA-at-Risk, HHI diversification & portfolio risk
  4. Flat carbon-price sweep (US$50-150/t), robustness variants and valuation-multiple sensitivity
  5. Historical back-test, Monte Carlo 90% intervals, OAT sensitivity & scenarios
  6. Export of summary LaTeX/CSV tables and 300-DPI figures
"""
from __future__ import annotations

import argparse
import sys
import pandas as pd

from src import config as cfg
from src.data_processing import run_data_processing
from src.emissions_model import run_emissions_model
from src.cvar_model import (
    run_cvar_model,
    run_price_sweep,
    run_robustness_variants,
    run_valuation_multiple_sensitivity,
)
from src.uncertainty import run_historical_backtest, build_uncertainty_tables
from src.visualization import generate_all_figures


RAW_DIR = cfg.DATA_PROCESSED.parent / "raw"

REQUIRED_RAW_FILES = [
    "gem_cement_july2026.xlsx",
    "emissions_co2_v5_12_0.csv",
    "ownership_v5_12_0.csv",
    "wb_carbon_pricing.csv",
]
NGFS_RAW_FILE = "ngfs_downscaled_remind.xlsx"


def check_required_files(refresh_ngfs: bool, refresh_financials: bool) -> None:
    """Verify required raw files exist before running; exit with guidance if missing."""
    required = list(REQUIRED_RAW_FILES)
    if refresh_ngfs or not (cfg.DATA_PROCESSED / "ngfs_carbon_prices.csv").exists():
        required.append(NGFS_RAW_FILE)
    if not refresh_financials:
        required.append("financials_cache.csv")

    missing = [RAW_DIR / fname for fname in required if not (RAW_DIR / fname).exists()]
    if missing:
        print("ERROR: Missing required raw data file(s) in data/raw/:")
        for path in missing:
            print(f"  - {path}")
        print(
            "\nDownload the missing file(s) listed in README.md and place them in data/raw/, then re-run."
        )
        sys.exit(1)


def _save(df: pd.DataFrame, name: str, decimals: int = 2) -> None:
    """Write one summary table as CSV and LaTeX."""
    df = df.round(decimals)
    df.to_csv(cfg.OUTPUT_TABLES / f"{name}.csv", index=False)
    df.to_latex(cfg.OUTPUT_TABLES / f"{name}.tex", index=False, float_format=f"%.{decimals}f")


def export_summary_tables(
    summary: pd.DataFrame,
    tier_df: pd.DataFrame,
    port_df: pd.DataFrame,
    mc_sum: pd.DataFrame,
    oat_df: pd.DataFrame,
    scen_df: pd.DataFrame,
    agg_bt: pd.DataFrame,
    panel_bt: pd.DataFrame,
    sweep_df: pd.DataFrame,
    robust_df: pd.DataFrame,
    valuation_df: pd.DataFrame,
) -> None:
    """Save clean CSV and LaTeX (.tex) summary tables."""
    cfg.OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    piv = summary.pivot(
        index=["diversification_tier", "geo_hhi", "business_mix", "old_unabated_cap_share", "company"],
        columns="scenario",
        values="climate_var_ev_pct",
    ).reset_index()

    dt_ci = mc_sum[mc_sum["scenario"] == "Delayed transition"][
        ["company", "cvar_p05_pct", "cvar_p95_pct", "prob_stranded"]
    ].rename(columns={"prob_stranded": "prob_stranded_dt"})
    nz_strand = summary[summary["scenario"] == "Net Zero 2050"][["company", "stranded"]].rename(
        columns={"stranded": "stranded_nz2050"}
    )
    t1 = (
        piv.merge(dt_ci, on="company")
        .merge(nz_strand, on="company")
        .sort_values(["diversification_tier", "geo_hhi"])
    )
    t1 = t1[
        [
            "company",
            "diversification_tier",
            "geo_hhi",
            "Current Policies",
            "Fragmented World",
            "Delayed transition",
            "cvar_p05_pct",
            "cvar_p95_pct",
            "prob_stranded_dt",
            "Net Zero 2050",
            "stranded_nz2050",
            "old_unabated_cap_share",
        ]
    ].round(2)
    t1.to_csv(cfg.OUTPUT_TABLES / "table1_company_cvar.csv", index=False)
    t1.to_latex(cfg.OUTPUT_TABLES / "table1_company_cvar.tex", index=False, float_format="%.2f")

    t2 = tier_df[
        [
            "scenario",
            "diversification_tier",
            "n_companies",
            "n_stranded",
            "median_cvar_ev_pct",
            "pure_play_median_cvar_ev_pct",
            "median_ebitda_risk_2030_pct",
            "median_peak_ebitda_risk_pct",
            "hhi_cvar_spearman_rho",
            "hhi_cvar_spearman_pval",
        ]
    ].round(2)
    t2.to_csv(cfg.OUTPUT_TABLES / "table2_tier_comparison.csv", index=False)
    t2.to_latex(cfg.OUTPUT_TABLES / "table2_tier_comparison.tex", index=False, float_format="%.2f")

    t3 = port_df.round(2)
    t3.to_csv(cfg.OUTPUT_TABLES / "table3_portfolio_risk.csv", index=False)
    t3.to_latex(cfg.OUTPUT_TABLES / "table3_portfolio_risk.tex", index=False, float_format="%.2f")

    oat_df.round(2).to_csv(cfg.OUTPUT_TABLES / "table4_oat_sensitivity.csv", index=False)
    oat_df.round(2).to_latex(cfg.OUTPUT_TABLES / "table4_oat_sensitivity.tex", index=False, float_format="%.2f")

    scen_df.round(2).to_csv(cfg.OUTPUT_TABLES / "table5_scenario_ipcc.csv", index=False)
    scen_df.round(2).to_latex(cfg.OUTPUT_TABLES / "table5_scenario_ipcc.tex", index=False, float_format="%.2f")

    # Table 6: does HHI still matter once EU/UK/CH ETS exposure is controlled for?
    t6 = pd.read_csv(cfg.DATA_PROCESSED / "hhi_eu_control_regression.csv").round(3)
    t6.to_csv(cfg.OUTPUT_TABLES / "table6_hhi_eu_control.csv", index=False)
    t6.to_latex(cfg.OUTPUT_TABLES / "table6_hhi_eu_control.tex", index=False, float_format="%.3f")

    # Table 7: out-of-sample back-test of the emissions trend model (train 2021-2023, test 2024-2025)
    t7_panel = panel_bt.assign(series="18-company panel (company-year holdouts)")
    t7_agg = (
        agg_bt.groupby("model")
        .agg(
            n_holdout_points=("inside_90", "size"),
            hits_inside_90=("inside_90", "sum"),
            mean_abs_pct_err=("abs_pct_err", "mean"),
        )
        .reset_index()
        .assign(series="Aggregate sample emissions")
    )
    t7_agg["misses"] = t7_agg["n_holdout_points"] - t7_agg["hits_inside_90"]
    t7_agg["empirical_coverage_pct"] = 100.0 * t7_agg["hits_inside_90"] / t7_agg["n_holdout_points"]
    t7 = pd.concat([t7_agg, t7_panel], ignore_index=True)[
        [
            "series",
            "model",
            "n_holdout_points",
            "hits_inside_90",
            "misses",
            "empirical_coverage_pct",
            "binomial_tail_p_value",
            "mean_abs_pct_err",
        ]
    ]
    _save(t7, "table7_backtest", decimals=3)

    # Table 8: flat carbon-price sweep (US$50-150/t)
    _save(sweep_df, "table8_price_sweep")

    # Table 9: robustness of group results to key assumptions
    _save(robust_df, "table9_robustness")

    # Table 10: Climate VaR with a common EV/EBITDA multiple (separates valuation from carbon-cost effects)
    _save(valuation_df, "table10_valuation_multiple")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cement Climate VaR Pipeline")
    parser.add_argument("--refresh-ngfs", action="store_true", help="Re-parse raw NGFS Excel workbook")
    parser.add_argument("--refresh-financials", action="store_true", help="Re-fetch Yahoo Finance data")
    args = parser.parse_args()

    check_required_files(refresh_ngfs=args.refresh_ngfs, refresh_financials=args.refresh_financials)

    print("[1/6] Running data validation, cleaning, and ingestion...")
    run_data_processing(
        refresh_ngfs=args.refresh_ngfs,
        refresh_financials=args.refresh_financials,
    )

    print("[2/6] Running emissions model & cross-source reconciliation...")
    run_emissions_model()

    print("[3/6] Computing company & portfolio Climate VaR across NGFS scenarios...")
    summary, annual, tier_df, port_df = run_cvar_model()

    print("[4/6] Running US$50-150/t carbon-price sweep, robustness & valuation-multiple sensitivity...")
    sweep_df = run_price_sweep()
    robust_df = run_robustness_variants(summary)
    valuation_df = run_valuation_multiple_sensitivity(summary)

    print("[5/6] Running historical back-test, Monte Carlo simulation & sensitivity...")
    agg_bt, panel_bt = run_historical_backtest()
    mc_sum, oat_df, scen_df, fan_df = build_uncertainty_tables()

    print("[6/6] Exporting summary tables and 300-DPI figures...")
    export_summary_tables(
        summary, tier_df, port_df, mc_sum, oat_df, scen_df, agg_bt, panel_bt, sweep_df, robust_df, valuation_df
    )
    generate_all_figures()

    print("\nPipeline completed successfully!")
    print(f"  - Processed datasets : {cfg.DATA_PROCESSED}")
    print(f"  - Output tables      : {cfg.OUTPUT_TABLES}")
    print(f"  - Output figures     : {cfg.OUTPUT_FIGURES}")


if __name__ == "__main__":
    main()
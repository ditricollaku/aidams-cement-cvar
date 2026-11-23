"""
Single executable entry point reproducing the entire Cement Climate VaR pipeline:
  1. Raw schema validation, GEM cleaning, NGFS extraction, and cached financials
  2. Equity-weighted Climate TRACE emissions panel & GEM bottom-up reconciliation
  3. Climate VaR (% EV), EBITDA-at-Risk, HHI diversification & portfolio risk
  4. Historical back-test, Monte Carlo 90% intervals, OAT sensitivity & scenarios
  5. Export of summary LaTeX/CSV tables and 300-DPI figures
"""
from __future__ import annotations

import argparse
import pandas as pd

from src import config as cfg
from src.data_processing import run_data_processing
from src.emissions_model import run_emissions_model
from src.cvar_model import run_cvar_model
from src.uncertainty import run_historical_backtest, build_uncertainty_tables
from src.visualization import generate_all_figures


def export_summary_tables(
    summary: pd.DataFrame,
    tier_df: pd.DataFrame,
    port_df: pd.DataFrame,
    mc_sum: pd.DataFrame,
    oat_df: pd.DataFrame,
    scen_df: pd.DataFrame,
) -> None:
    """Save clean CSV and LaTeX (.tex) summary tables."""
    cfg.OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    # Table 1: Company-level Climate VaR across scenarios + Delayed Transition 90% CI
    piv = summary.pivot(
        index=["diversification_tier", "geo_hhi", "business_mix", "company"],
        columns="scenario",
        values="climate_var_ev_pct",
    ).reset_index()

    dt_ci = mc_sum[mc_sum["scenario"] == "Delayed transition"][
        ["company", "cvar_p05_pct", "cvar_p95_pct"]
    ]
    t1 = piv.merge(dt_ci, on="company").sort_values(["diversification_tier", "geo_hhi"])
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
            "Net Zero 2050",
        ]
    ].round(2)
    t1.to_csv(cfg.OUTPUT_TABLES / "table1_company_cvar.csv", index=False)
    t1.to_latex(cfg.OUTPUT_TABLES / "table1_company_cvar.tex", index=False, float_format="%.2f")

    # Table 2: Multinational vs Regional Group Summary
    t2 = tier_df[
        [
            "scenario",
            "diversification_tier",
            "n_companies",
            "median_cvar_ev_pct",
            "pure_play_median_cvar_ev_pct",
            "median_ebitda_risk_2030_pct",
            "median_peak_ebitda_risk_pct",
            "hhi_cvar_spearman_rho",
        ]
    ].round(2)
    t2.to_csv(cfg.OUTPUT_TABLES / "table2_tier_comparison.csv", index=False)
    t2.to_latex(cfg.OUTPUT_TABLES / "table2_tier_comparison.tex", index=False, float_format="%.2f")

    # Table 3: Portfolio Risk Comparison
    t3 = port_df.round(2)
    t3.to_csv(cfg.OUTPUT_TABLES / "table3_portfolio_risk.csv", index=False)
    t3.to_latex(cfg.OUTPUT_TABLES / "table3_portfolio_risk.tex", index=False, float_format="%.2f")

    # Table 4: OAT Sensitivity & Scenario Uncertainty Summary
    oat_df.round(2).to_csv(cfg.OUTPUT_TABLES / "table4_oat_sensitivity.csv", index=False)
    oat_df.round(2).to_latex(cfg.OUTPUT_TABLES / "table4_oat_sensitivity.tex", index=False, float_format="%.2f")

    scen_df.round(2).to_csv(cfg.OUTPUT_TABLES / "table5_scenario_ipcc.csv", index=False)
    scen_df.round(2).to_latex(cfg.OUTPUT_TABLES / "table5_scenario_ipcc.tex", index=False, float_format="%.2f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cement Climate VaR Pipeline")
    parser.add_argument("--refresh-ngfs", action="store_true", help="Re-parse raw NGFS Excel workbook")
    parser.add_argument("--refresh-financials", action="store_true", help="Re-fetch Yahoo Finance data")
    args = parser.parse_args()

    print("[1/5] Running data validation, cleaning, and ingestion...")
    val_rep, gem_df, ngfs_df, fin_df = run_data_processing(
        refresh_ngfs=args.refresh_ngfs,
        refresh_financials=args.refresh_financials,
    )

    print("[2/5] Running emissions model & cross-source reconciliation...")
    ct_py, ct_ccy, geo_df, reality_df = run_emissions_model()

    print("[3/5] Computing company & portfolio Climate VaR across NGFS scenarios...")
    summary, annual, tier_df, port_df = run_cvar_model()

    print("[4/5] Running historical back-test, Monte Carlo simulation & sensitivity...")
    agg_bt, panel_bt = run_historical_backtest()
    mc_sum, oat_df, scen_df, fan_df = build_uncertainty_tables()

    print("[5/5] Exporting summary tables and 300-DPI figures...")
    export_summary_tables(summary, tier_df, port_df, mc_sum, oat_df, scen_df)
    generate_all_figures()

    print("\nPipeline completed successfully!")
    print(f"  - Processed datasets : {cfg.DATA_PROCESSED}")
    print(f"  - Output tables      : {cfg.OUTPUT_TABLES}")
    print(f"  - Output figures     : {cfg.OUTPUT_FIGURES}")


if __name__ == "__main__":
    main()

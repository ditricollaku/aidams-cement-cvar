"""
Publication-grade figures (300 DPI PNGs) and interactive Plotly map
for geospatial footprint, Climate VaR, sensitivity, reality check, and fan chart.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from src import config as cfg

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def plot_geospatial_footprint() -> None:
    """Geospatial plant footprint contrasting Diversified vs Regional producers."""
    gem = pd.read_parquet(cfg.DATA_PROCESSED / "gem_plants_clean.parquet")
    geo = pd.read_csv(cfg.DATA_PROCESSED / "company_geo_profile.csv")

    op = gem[gem["is_operating"] & gem["lat"].notna() & gem["lon"].notna()].copy()
    op = op.merge(geo[["company", "diversification_tier", "geo_hhi"]], on="company", how="inner")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    tiers = [
        ("Diversified Multinational", "#1f77b4", axes[0]),
        ("Regionally Concentrated", "#d62728", axes[1]),
    ]

    for tier_name, color, ax in tiers:
        sub = op[op["diversification_tier"] == tier_name]
        sizes = np.clip(sub["eq_cement_cap_mtpa"].fillna(1.0) * 18.0, 12.0, 220.0)
        ax.scatter(
            sub["lon"],
            sub["lat"],
            s=sizes,
            c=color,
            alpha=0.58,
            edgecolors="black",
            linewidths=0.35,
        )
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
        ax.set_xlim(-130, 155)
        ax.set_ylim(-45, 65)
        ax.set_title(
            f"{tier_name} Producers\n({len(sub)} equity-linked plants across {sub['Country/area'].nunique()} countries)"
        )
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.suptitle(
        "Global Cement Plant Footprint by Corporate Diversification Tier (Marker Size = Equity Capacity)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(cfg.OUTPUT_FIGURES / "fig1_geospatial_footprint.png", bbox_inches="tight")
    plt.close(fig)

    op["plot_size_mtpa"] = op["eq_cement_cap_mtpa"].fillna(1.0)
    fig_px = px.scatter_geo(
        op,
        lat="lat",
        lon="lon",
        color="diversification_tier",
        size="plot_size_mtpa",
        hover_name="Plant name (English)",
        hover_data={
            "company": True,
            "Country/area": True,
            "eq_cement_cap_mtpa": ":.2f",
            "has_ccs": True,
            "lat": False,
            "lon": False,
        },
        title="Interactive Cement Asset Map: Diversified Multinationals vs. Regional Producers",
        projection="natural earth",
    )
    fig_px.write_html(cfg.OUTPUT_FIGURES / "interactive_plant_map.html")


def plot_hhi_vs_cvar_and_portfolios() -> None:
    """Geographic HHI vs Climate VaR (90% CI) and Portfolio Tail Dispersion."""
    summary = pd.read_csv(cfg.DATA_PROCESSED / "company_cvar_summary.csv")
    mc = pd.read_csv(cfg.DATA_PROCESSED / "mc_company_cvar_intervals.csv")
    port = pd.read_csv(cfg.DATA_PROCESSED / "portfolio_cvar_comparison.csv")

    dt_sub = summary[summary["scenario"] == "Delayed transition"].merge(
        mc[mc["scenario"] == "Delayed transition"][
            ["company", "cvar_p05_pct", "cvar_median_pct", "cvar_p95_pct"]
        ],
        on="company",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    palette = {
        "Diversified Multinational": "#1f77b4",
        "Regionally Concentrated": "#d62728",
    }
    for tier, grp in dt_sub.groupby("diversification_tier"):
        y_med = grp["cvar_median_pct"].to_numpy()
        y_err_lo = y_med - grp["cvar_p05_pct"].to_numpy()
        y_err_hi = grp["cvar_p95_pct"].to_numpy() - y_med
        ax1.errorbar(
            grp["geo_hhi"],
            y_med,
            yerr=[y_err_lo, y_err_hi],
            fmt="o",
            color=palette[tier],
            ecolor=palette[tier],
            elinewidth=1.1,
            capsize=3,
            markersize=7,
            label=tier,
            alpha=0.85,
        )
        for _, r in grp.iterrows():
            if r["core_example"] or r["cvar_median_pct"] > 100:
                ax1.annotate(
                    r["company"],
                    (r["geo_hhi"], r["cvar_median_pct"]),
                    xytext=(-48, 8) if r["company"] == "UltraTech Cement" else (5, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

    ax1.axvline(
        cfg.HHI_DIVERSIFICATION_THRESHOLD,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="HHI = 0.50 Cutoff",
    )
    ax1.set_xlabel("Geographic Concentration Index (HHI)")
    ax1.set_ylabel("Climate VaR (% of Enterprise Value, 90% CI)")
    ax1.set_title("A. Geographic HHI vs. Climate VaR (Delayed Transition)")
    ax1.legend(loc="upper left", fontsize=8.5)

    scen_order = ["Current Policies", "Fragmented World", "Delayed transition", "Net Zero 2050"]
    x = np.arange(len(scen_order))
    width = 0.26

    p_configs = [
        ("Portfolio A: Diversified Multinationals (9 firms)", "Multinationals (9)", "#1f77b4", -width),
        ("Portfolio B: Regionally Concentrated (9 firms)", "Regional (9)", "#d62728", 0.0),
        ("Portfolio C: 50/50 Blended Global Cement (18 firms)", "50/50 Blended (18)", "#2ca02c", width),
    ]
    for full_name, short_name, col, offset in p_configs:
        sub = port[port["portfolio"] == full_name].set_index("scenario").reindex(scen_order)
        vals = sub["equal_weighted_cvar_ev_pct"].to_numpy()
        stds = sub["cross_firm_cvar_std_pct"].to_numpy()
        ax2.bar(
            x + offset,
            vals,
            width=width,
            color=col,
            alpha=0.85,
            label=short_name,
        )
        ax2.errorbar(
            x + offset,
            vals,
            yerr=[np.zeros_like(stds), stds],
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Current\nPolicies", "Fragmented\nWorld", "Delayed\nTransition", "Net Zero\n2050"])
    ax2.set_ylabel("Equal-Weighted Portfolio Climate VaR (% EV)")
    ax2.set_title("B. Portfolio Climate VaR & Cross-Firm Tail Dispersion (+1 SD)")
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(cfg.OUTPUT_FIGURES / "fig2_hhi_vs_cvar.png", bbox_inches="tight")
    plt.close(fig)


def plot_oat_sensitivity_tornado() -> None:
    """One-at-a-Time (OAT) Sensitivity Tornado Chart."""
    oat = pd.read_csv(cfg.DATA_PROCESSED / "oat_sensitivity.csv")
    joint_w = float(oat.iloc[0]["interval_width_90_pct"])
    sub = oat.iloc[1:].sort_values("interval_width_90_pct", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    y_pos = np.arange(len(sub))
    meds = sub["median_portfolio_cvar_pct"].to_numpy()
    lefts = sub["p05_pct"].to_numpy()
    widths = sub["interval_width_90_pct"].to_numpy()

    ax.barh(y_pos, widths, left=lefts, height=0.52, color="#2b5c8f", alpha=0.85, edgecolor="black")
    ax.axvline(meds.mean(), color="#d62728", linestyle="--", linewidth=1.2, label="Baseline Median (~71.5% EV)")

    for idx, (_, r) in enumerate(sub.iterrows()):
        ax.text(
            r["p95_pct"] + 1.0,
            idx,
            f"[{r['p05_pct']:.1f}%, {r['p95_pct']:.1f}%] (width: {r['interval_width_90_pct']:.1f} pp)",
            va="center",
            fontsize=8.5,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["driver"])
    ax.set_xlim(40, 118)
    ax.set_xlabel("Portfolio Climate VaR (% of Enterprise Value, Delayed Transition)")
    ax.set_title(
        f"One-at-a-Time Sensitivity of 90% Climate VaR Band (Joint MC Width = {joint_w:.1f} pp)"
    )
    ax.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(cfg.OUTPUT_FIGURES / "fig3_tornado_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def plot_reality_check() -> None:
    """Cross-source reconciliation: Climate TRACE 2025 vs GEM bottom-up."""
    check = pd.read_csv(cfg.DATA_PROCESSED / "emissions_reality_check.csv")
    check = check.sort_values("ct_emissions_2025_mt", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(check))
    h = 0.38

    ax.barh(y - h / 2, check["ct_emissions_2025_mt"], height=h, label="Climate TRACE (2025 Satellite/ML)", color="#1f77b4")
    ax.barh(y + h / 2, check["gem_est_emissions_2025_mt"], height=h, label="GEM Bottom-Up (2025 Engineering)", color="#ff7f0e", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(check["company"])
    ax.set_xlabel("Equity-Weighted Scope 1 Emissions (MtCO2 / yr)")
    ax.set_title("Cross-Source Reality Check: Climate TRACE vs. GEM Bottom-Up Estimates (2025)")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(cfg.OUTPUT_FIGURES / "fig4_reality_check.png", bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_fan_chart() -> None:
    """Historical 2021-2025 back-test and 2026-2040 scenario transition cost fan chart."""
    from src.uncertainty import fit_predict

    fan = pd.read_csv(cfg.DATA_PROCESSED / "fan_chart_data.csv")
    hist = fan[fan["scenario"] == "Historical (Climate TRACE)"].sort_values("year")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax1.plot(
        hist["year"],
        hist["sample_emissions_mt"],
        marker="o",
        color="black",
        linewidth=2.0,
        label="Observed Emissions (2021–2025)",
    )
    tr = hist[hist["year"] <= 2023][["year", "sample_emissions_mt"]].rename(
        columns={"sample_emissions_mt": "value"}
    )
    bt = fit_predict(tr, horizon=2, model="linear", level=0.90)
    ax1.plot(bt["year"], bt["mean"], linestyle="--", color="#1f77b4", label="OLS Back-Test Mean")
    ax1.fill_between(
        bt["year"],
        bt["lo"],
        bt["hi"],
        color="#1f77b4",
        alpha=0.22,
        label="90% Student-t Prediction Band",
    )
    ax1.set_xticks([2021, 2022, 2023, 2024, 2025])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total Sample Scope 1 Emissions (MtCO2)")
    ax1.set_title("A. Historical Sample Emissions & Hold-Out Back-Test (2024–2025)")
    ax1.legend(loc="lower left", fontsize=8.5)

    scen_styles = [
        ("Current Policies", "#2ca02c"),
        ("Delayed transition", "#ff7f0e"),
        ("Net Zero 2050", "#d62728"),
    ]
    for scen, col in scen_styles:
        sub = fan[fan["scenario"] == scen].sort_values("year")
        yrs = sub["year"].to_numpy()
        med = sub["cost_median_usd_m"].to_numpy() / 1e3
        p05 = sub["cost_p05_usd_m"].to_numpy() / 1e3
        p25 = sub["cost_p25_usd_m"].to_numpy() / 1e3
        p75 = sub["cost_p75_usd_m"].to_numpy() / 1e3
        p95 = sub["cost_p95_usd_m"].to_numpy() / 1e3

        ax2.fill_between(yrs, p05, p95, color=col, alpha=0.13)
        ax2.fill_between(yrs, p25, p75, color=col, alpha=0.28)
        ax2.plot(yrs, med, color=col, linewidth=2.0, label=f"{scen} (Median, 50% & 90% CI)")

    ax2.set_xlabel("Projection Year")
    ax2.set_ylabel("Annual Unpassed Carbon Cost (USD Billions, 2026$)")
    ax2.set_title("B. Projected 18-Firm Transition Cost Fan Chart (2026–2040)")
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(cfg.OUTPUT_FIGURES / "fig_uncertainty.png", bbox_inches="tight")
    plt.close(fig)


def generate_all_figures() -> None:
    cfg.OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    plot_geospatial_footprint()
    plot_hhi_vs_cvar_and_portfolios()
    plot_oat_sensitivity_tornado()
    plot_reality_check()
    plot_uncertainty_fan_chart()


if __name__ == "__main__":
    generate_all_figures()
    print("Saved all 5 figures + interactive HTML map to:", cfg.OUTPUT_FIGURES)
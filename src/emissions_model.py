"""
Emissions modeling module:
1. Equity-weighted Climate TRACE historical emissions panel (2021-2026)
2. Geographic HHI diversification index & jurisdiction breakdown
3. Bottom-up GEM Scope 1 emissions model & cross-source reality check
"""
from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pandas as pd

from src import config as cfg


# ---------------------------------------------------------------------------
# 1. CLIMATE TRACE EQUITY-WEIGHTED PANEL (2021-2026)
# ---------------------------------------------------------------------------
_PATH_PCT_RE = re.compile(r"\[(\d+(?:\.\d+)?)%\]")


def _known_path_share_pct(path: str) -> float:
    """
    Effective share (%) implied by the KNOWN link percentages of an ownership path, e.g.
      'HM AG -> Akcansa [39.72%] -> Cimsa [8.98%] -> Cimsa Espana [unknown %] -> Plant [100.0%]'
    -> 39.72% * 8.98% * 100% = 3.57%. Unknown links are treated as 100% (upper bound).
    Climate TRACE leaves overall_share_percent empty whenever ANY link is unknown, which
    previously defaulted the whole chain to 100% and discarded the known minority links.
    """
    pcts = _PATH_PCT_RE.findall(str(path))
    if not pcts:
        return np.nan
    return float(np.prod([float(p) / 100.0 for p in pcts]) * 100.0)


def build_ct_company_panel() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build deduplicated, equity-weighted Climate TRACE cement emissions panel:
      - ct_plant_year: (company, source_id, iso3_country, year)
      - ct_company_country_year: aggregated by (company, iso3_country, year)
    """
    em = pd.read_csv(cfg.CT_EMISSIONS_FILE, low_memory=False)
    own = pd.read_csv(cfg.CT_OWNERSHIP_FILE, low_memory=False)

    em["year"] = pd.to_datetime(em["start_time"]).dt.year

    # Match each target company against ownership table and deduplicate multi-layer parents
    # Taking min(overall_share_percent) within the same ownership chain preserves the
    # ultimate parent's effective equity stake (e.g., 83.1% instead of local sub's 100%).
    matched_links = []
    for comp_name, meta in cfg.COMPANY_UNIVERSE.items():
        mask = own["parent_name"].fillna("").str.contains(
            meta["ct_regex"], case=False, regex=True
        )
        if meta["exclude_name"]:
            mask &= ~own["source_name"].fillna("").str.contains(
                meta["exclude_name"], case=False, regex=True
            )
            mask &= ~own["parent_name"].fillna("").str.contains(
                meta["exclude_name"], case=False, regex=True
            )
        if meta.get("exclude_iso3"):
            mask &= ~own["iso3_country"].isin(meta["exclude_iso3"])

        sub = own.loc[
            mask, ["source_id", "source_name", "iso3_country", "overall_share_percent", "ownership_path"]
        ].copy()
        if sub.empty:
            continue
        # Recover partial shares for chains containing an unknown link (see _known_path_share_pct)
        sub["overall_share_percent"] = sub["overall_share_percent"].fillna(
            sub["ownership_path"].map(_known_path_share_pct)
        )

        # Deduplicate to 1 row per (company, source_id)
        dedup = (
            sub.groupby("source_id", as_index=False)
            .agg(
                source_name=("source_name", "first"),
                iso3_country=("iso3_country", "first"),
                share_pct=("overall_share_percent", "min"),
            )
        )
        # Share still missing only if the path carries no percentage at all -> default to 100%
        dedup["equity_share"] = (dedup["share_pct"].fillna(100.0) / 100.0).clip(0.0, 1.0)
        dedup["company"] = comp_name
        matched_links.append(dedup[["company", "source_id", "equity_share"]])

    links = pd.concat(matched_links, ignore_index=True)

    # Guardrail: if multiple target companies share a JV plant and sum(equity_share) > 1.0, normalize
    jv_sums = links.groupby("source_id")["equity_share"].transform("sum")
    links["equity_share"] = np.where(
        jv_sums > 1.0, links["equity_share"] / jv_sums, links["equity_share"]
    )

    # Merge with Climate TRACE annual emissions (2021-2026)
    ct_plant_year = links.merge(
        em[
            [
                "source_id",
                "source_name",
                "source_type",
                "iso3_country",
                "lat",
                "lon",
                "year",
                "emissions_quantity",
                "activity",
                "emissions_factor",
                "capacity",
                "capacity_factor",
            ]
        ],
        on="source_id",
        how="inner",
    )

    # Convert tonnes to Mt and apply equity weights
    ct_plant_year["eq_emissions_mt"] = (
        ct_plant_year["emissions_quantity"] * ct_plant_year["equity_share"] / 1e6
    )
    ct_plant_year["eq_activity_mt"] = (
        ct_plant_year["activity"] * ct_plant_year["equity_share"] / 1e6
    )
    ct_plant_year["eq_capacity_mt"] = (
        ct_plant_year["capacity"] * ct_plant_year["equity_share"] / 1e6
    )

    ct_company_country_year = (
        ct_plant_year.groupby(["company", "iso3_country", "year"], as_index=False)
        .agg(
            n_plants=("source_id", "nunique"),
            eq_emissions_mt=("eq_emissions_mt", "sum"),
            eq_activity_mt=("eq_activity_mt", "sum"),
            eq_capacity_mt=("eq_capacity_mt", "sum"),
            mean_ef=("emissions_factor", "mean"),
        )
    )
    ct_company_country_year["implied_util"] = np.where(
        ct_company_country_year["eq_capacity_mt"] > 0,
        ct_company_country_year["eq_activity_mt"] / ct_company_country_year["eq_capacity_mt"],
        np.nan,
    )

    ct_plant_year.to_parquet(cfg.DATA_PROCESSED / "ct_plant_year.parquet", index=False)
    ct_company_country_year.to_csv(
        cfg.DATA_PROCESSED / "ct_company_country_year.csv", index=False
    )
    return ct_plant_year, ct_company_country_year


# ---------------------------------------------------------------------------
# 2. GEOGRAPHIC DIVERSIFICATION (HHI) & JURISDICTION SHARES
# ---------------------------------------------------------------------------
def compute_geographic_profile(
    ct_company_country_year: pd.DataFrame, base_year: int = cfg.BASE_YEAR
) -> pd.DataFrame:
    """
    Compute each company's Geographic HHI, diversification tier, and key
    carbon-pricing jurisdiction shares using base_year (2025) emissions.
    """
    sub = ct_company_country_year[ct_company_country_year["year"] == base_year].copy()

    rows = []
    for comp, grp in sub.groupby("company"):
        tot = grp["eq_emissions_mt"].sum()
        if tot <= 0:
            continue
        shares = grp["eq_emissions_mt"] / tot
        hhi = float((shares ** 2).sum())
        top_row = grp.loc[grp["eq_emissions_mt"].idxmax()]

        eu_share = grp.loc[
            grp["iso3_country"].isin(cfg.EU_UK_CH_ETS_ISO3), "eq_emissions_mt"
        ].sum() / tot
        chn_share = grp.loc[grp["iso3_country"] == "CHN", "eq_emissions_mt"].sum() / tot
        ind_share = grp.loc[grp["iso3_country"] == "IND", "eq_emissions_mt"].sum() / tot
        usca_share = grp.loc[
            grp["iso3_country"].isin(["USA", "CAN"]), "eq_emissions_mt"
        ].sum() / tot

        tier = (
            "Diversified Multinational"
            if hhi < cfg.HHI_DIVERSIFICATION_THRESHOLD
            else "Regionally Concentrated"
        )

        rows.append(
            {
                "company": comp,
                "ct_plants_2025": int(grp["n_plants"].sum()),
                "ct_countries_2025": int(grp["iso3_country"].nunique()),
                "ct_emissions_2025_mt": tot,
                "ct_activity_2025_mt": grp["eq_activity_mt"].sum(),
                "ct_capacity_2025_mt": grp["eq_capacity_mt"].sum(),
                "ct_implied_util_2025": (
                    grp["eq_activity_mt"].sum() / grp["eq_capacity_mt"].sum()
                    if grp["eq_capacity_mt"].sum() > 0
                    else np.nan
                ),
                "geo_hhi": hhi,
                "diversification_tier": tier,
                "top_country": top_row["iso3_country"],
                "top_country_share": float(top_row["eq_emissions_mt"] / tot),
                "eu_uk_ch_share": float(eu_share),
                "china_share": float(chn_share),
                "india_share": float(ind_share),
                "us_ca_share": float(usca_share),
            }
        )

    geo_df = pd.DataFrame(rows).sort_values("geo_hhi")
    geo_df.to_csv(cfg.DATA_PROCESSED / "company_geo_profile.csv", index=False)
    return geo_df


# ---------------------------------------------------------------------------
# 3. GEM BOTTOM-UP EMISSIONS MODEL & CROSS-SOURCE REALITY CHECK
# ---------------------------------------------------------------------------
def build_gem_bottom_up_and_reality_check(
    geo_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate 2025 bottom-up Scope 1 CO2 from GEM plant-level clinker capacity,
    kiln technology, clinker substitution, alt-fuels, and CCS, then compare
    against Climate TRACE 2025 totals (Lab 3 reality check).
    """
    gem = pd.read_parquet(cfg.DATA_PROCESSED / "gem_plants_clean.parquet")
    op = gem[gem["is_operating"]].copy()

    # Map EN-standard clinker substitution rate -> implied clinker-to-cement ratio
    sub_ratio_map = {
        "0–5%": 0.95,
        "0-5%": 0.95,
        "a": 0.85,
        "b": 0.72,
        "c": 0.57,
        ">50%": 0.45,
    }
    sub_key = op["Clinker substitution rate"].astype(str).str.strip().str.lower()
    op["clinker_ratio"] = sub_key.map(sub_ratio_map).fillna(0.72)

    # Grinding plants have no kilns (0 direct clinker process emissions)
    is_kiln_plant = op["Plant type"].isin(["integrated", "clinker only", "unknown"])
    op["eff_eq_clinker_cap_mtpa"] = np.where(
        ~is_kiln_plant,
        0.0,
        op["eq_clinker_cap_mtpa"].fillna(op["eq_cement_cap_mtpa"] * op["clinker_ratio"]),
    )

    # Technology-adjusted emission factor (tCO2 / t clinker, GCCA/IPCC benchmarks)
    kiln_ef_map = {
        "dry": 0.84,
        "semidry": 0.88,
        "mixed": 0.89,
        "wet": 0.96,
    }
    base_ef = op["Clinker production method"].map(kiln_ef_map).fillna(0.86)
    # Alternative fuels lower fossil thermal emissions (~0.06 tCO2/t clinker average saving)
    alt_fuel_adj = np.where(op["has_alt_fuel"], -0.06, 0.0)
    # Early-stage CCS/CCUS flagship adjustment (15% site-wide average capture proxy)
    ccs_mult = np.where(op["has_ccs"], 0.85, 1.0)
    op["ef_tco2_per_t_clinker"] = (base_ef + alt_fuel_adj) * ccs_mult

    # Attach company-level empirical utilization from Climate TRACE (fallback = global median)
    util_map = geo_df.set_index("company")["ct_implied_util_2025"].to_dict()
    global_med_util = float(geo_df["ct_implied_util_2025"].median())
    op["util_rate"] = op["company"].map(util_map).fillna(global_med_util).clip(0.35, 0.95)

    op["gem_est_emissions_mt"] = (
        op["eff_eq_clinker_cap_mtpa"] * op["util_rate"] * op["ef_tco2_per_t_clinker"]
    )

    # Stranded-asset / high-tech-risk capacity share: unabated integrated kilns >= 30 yrs old
    op["old_unabated_clink_cap"] = np.where(
        is_kiln_plant & (~op["has_ccs"]) & (op["plant_age_yrs"].fillna(0) >= 30),
        op["eff_eq_clinker_cap_mtpa"].fillna(0.0),
        0.0,
    )

    gem_agg = (
        op.groupby("company", as_index=False)
        .agg(
            gem_op_plants=("GEM plant ID", "count"),
            gem_kiln_plants=("Plant type", lambda s: s.isin(["integrated", "clinker only"]).sum()),
            gem_eq_cement_cap_mtpa=("eq_cement_cap_mtpa", "sum"),
            gem_eff_clinker_cap_mtpa=("eff_eq_clinker_cap_mtpa", "sum"),
            gem_est_emissions_2025_mt=("gem_est_emissions_mt", "sum"),
            gem_ccs_plants=("has_ccs", "sum"),
            gem_alt_fuel_share=("has_alt_fuel", "mean"),
            gem_mean_plant_age=("plant_age_yrs", "mean"),
            old_unabated_clink_cap=("old_unabated_clink_cap", "sum"),
        )
    )
    gem_agg["old_unabated_cap_share"] = np.where(
        gem_agg["gem_eff_clinker_cap_mtpa"] > 0,
        gem_agg["old_unabated_clink_cap"] / gem_agg["gem_eff_clinker_cap_mtpa"],
        0.0,
    )

    # Reality-check merge: Climate TRACE vs GEM bottom-up
    check = geo_df.merge(gem_agg, on="company", how="outer")
    check["ct_to_gem_ratio"] = check["ct_emissions_2025_mt"] / check["gem_est_emissions_2025_mt"]
    check.to_csv(cfg.DATA_PROCESSED / "emissions_reality_check.csv", index=False)
    return check


def run_emissions_model() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ct_plant_year, ct_ccy = build_ct_company_panel()
    geo_df = compute_geographic_profile(ct_ccy)
    reality_check = build_gem_bottom_up_and_reality_check(geo_df)
    return ct_plant_year, ct_ccy, geo_df, reality_check


if __name__ == "__main__":
    ct_py, ct_ccy, geo, check = run_emissions_model()

    print("=== 1. GEOGRAPHIC DIVERSIFICATION (HHI) & JURISDICTION SHARES (2025) ===")
    cols1 = [
        "company",
        "diversification_tier",
        "geo_hhi",
        "ct_countries_2025",
        "top_country",
        "top_country_share",
        "eu_uk_ch_share",
        "china_share",
        "india_share",
    ]
    print(geo[cols1].round(3).to_string(index=False))

    print("\n=== 2. LAB 3 REALITY CHECK: CLIMATE TRACE 2025 vs GEM BOTTOM-UP (MtCO2) ===")
    cols2 = [
        "company",
        "ct_plants_2025",
        "gem_kiln_plants",
        "ct_implied_util_2025",
        "ct_emissions_2025_mt",
        "gem_est_emissions_2025_mt",
        "ct_to_gem_ratio",
        "gem_ccs_plants",
        "old_unabated_cap_share",
    ]
    print(
        check[cols2]
        .sort_values("ct_emissions_2025_mt", ascending=False)
        .round(2)
        .to_string(index=False)
    )

    print("\n=== 3. HISTORICAL EMISSIONS TREND (2021-2026, TOTAL SAMPLE MtCO2) ===")
    yr_tot = ct_ccy.groupby("year")["eq_emissions_mt"].sum()
    print(yr_tot.round(2).to_string())

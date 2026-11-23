"""
Central configuration for the Cement Climate VaR project.
Single source of truth for file paths, company universe, NGFS scenarios,
policy parameters, and Monte Carlo settings.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. DIRECTORY & FILE PATHS
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
OUTPUT_FIGURES = ROOT_DIR / "outputs" / "figures"
OUTPUT_TABLES = ROOT_DIR / "outputs" / "tables"

GEM_FILE = DATA_RAW / "gem_cement_july2026.xlsx"
CT_EMISSIONS_FILE = DATA_RAW / "emissions_co2_v5_12_0.csv"
CT_OWNERSHIP_FILE = DATA_RAW / "ownership_v5_12_0.csv"
WB_CARBON_FILE = DATA_RAW / "wb_carbon_pricing.csv"
NGFS_DOWNSCALED_FILE = DATA_RAW / "ngfs_downscaled_remind.xlsx"
FINANCIALS_CACHE_FILE = DATA_RAW / "financials_cache.csv"

# ---------------------------------------------------------------------------
# 2. COMPANY UNIVERSE (18 Major Listed Global & Regional Cement Producers)
# ---------------------------------------------------------------------------
# Fields:
#   ticker:        Yahoo Finance symbol
#   gem_regex:     Case-insensitive regex matched against GEM 'Parent'
#   ct_regex:      Case-insensitive regex matched against Climate TRACE 'parent_name'
#   exclude_name:  Optional regex excluded from plant/source names (perimeter alignment)
#   hq_region:     Headquarters region
#   business_mix:  'pure_play' vs 'diversified_materials' (controls for non-cement EBITDA)
#   core_example:  True for the 5 flagship companies named in the research question
COMPANY_UNIVERSE = {
    # --- Flagship & Global Multinationals ---
    "Holcim": {
        "ticker": "HOLN.SW",
        "gem_regex": r"holcim|lafarge",
        "ct_regex": r"holcim|lafarge",
        "exclude_name": r"amrize",  # Post-2025 North America spin-off perimeter
        "hq_region": "Europe",
        "business_mix": "pure_play",
        "core_example": True,
    },
    "Heidelberg Materials": {
        "ticker": "HEI.DE",
        "gem_regex": r"heidelberg|indocement",
        "ct_regex": r"heidelberg|indocement",
        "exclude_name": None,
        "hq_region": "Europe",
        "business_mix": "pure_play",
        "core_example": True,
    },
    "CRH": {
        "ticker": "CRH",
        "gem_regex": r"\bcrh\b",
        "ct_regex": r"\bcrh\b",
        "exclude_name": None,
        "hq_region": "Europe/North America",
        "business_mix": "diversified_materials",
        "core_example": True,
    },
    "CEMEX": {
        "ticker": "CX",
        "gem_regex": r"\bcemex\b",
        "ct_regex": r"\bcemex\b",
        "exclude_name": None,
        "hq_region": "Latin America",
        "business_mix": "pure_play",
        "core_example": False,
    },
    "Buzzi": {
        "ticker": "BZU.MI",
        "gem_regex": r"\bbuzzi\b",
        "ct_regex": r"\bbuzzi\b",
        "exclude_name": None,
        "hq_region": "Europe",
        "business_mix": "pure_play",
        "core_example": False,
    },
    "Vicat": {
        "ticker": "VCT.PA",
        "gem_regex": r"\bvicat\b",
        "ct_regex": r"\bvicat\b",
        "exclude_name": None,
        "hq_region": "Europe",
        "business_mix": "pure_play",
        "core_example": False,
    },
    "Titan Cement": {
        "ticker": "TITC.BR",
        "gem_regex": r"titan cement|titan group|titan america",
        "ct_regex": r"titan cement|titan group|titan america",
        "exclude_name": None,
        "hq_region": "Europe",
        "business_mix": "pure_play",
        "core_example": False,
    },
    "Cementir": {
        "ticker": "CEM.MI",
        "gem_regex": r"cementir",
        "ct_regex": r"cementir",
        "exclude_name": None,
        "hq_region": "Europe",
        "business_mix": "pure_play",
        "core_example": False,
    },
    "Taiheiyo Cement": {
        "ticker": "5233.T",
        "gem_regex": r"taiheiyo",
        "ct_regex": r"taiheiyo",
        "exclude_name": None,
        "hq_region": "Japan",
        "business_mix": "pure_play",
        "core_example": False,
    },
    # --- Major Regional Producers (China, India, ASEAN) ---
    "Anhui Conch": {
        "ticker": "0914.HK",
        "gem_regex": r"conch",
        "ct_regex": r"conch",
        "exclude_name": None,
        "hq_region": "China",
        "business_mix": "pure_play",
        "core_example": True,
    },
    "CNBM": {
        "ticker": "3323.HK",
        "gem_regex": r"china national building material|\bcnbm\b|sinoma|south cement|north cement|southwest cement|zhonglian",
        "ct_regex": r"china national building material|\bcnbm\b|sinoma|south cement|north cement|southwest cement|zhonglian",
        "exclude_name": None,
        "hq_region": "China",
        "business_mix": "diversified_materials",
        "core_example": False,
    },
    "Huaxin Cement": {
        "ticker": "6655.HK",
        "gem_regex": r"huaxin",
        "ct_regex": r"huaxin",
        "exclude_name": None,
        "hq_region": "China",
        "business_mix": "pure_play",
        "core_example": False,
    },
    "UltraTech Cement": {
        "ticker": "ULTRACEMCO.NS",
        "gem_regex": r"ultratech|ultra tech",
        "ct_regex": r"ultratech|ultra tech",
        "exclude_name": None,
        "hq_region": "India",
        "business_mix": "pure_play",
        "core_example": True,
    },
    "Ambuja Cements": {
        "ticker": "AMBUJACEM.NS",
        "gem_regex": r"ambuja|\bacc ltd\b",
        "ct_regex": r"ambuja|\bacc ltd\b",
        "exclude_name": None,
        "hq_region": "India",
        "business_mix": "pure_play",
        "core_example": False,
    },
    "Shree Cement": {
        "ticker": "SHREECEM.NS",
        "gem_regex": r"shree cement",
        "ct_regex": r"shree cement",
        "exclude_name": None,
        "hq_region": "India",
        "business_mix": "pure_play",
        "core_example": False,
    },
    "Dalmia Bharat": {
        "ticker": "DALBHARAT.NS",
        "gem_regex": r"dalmia",
        "ct_regex": r"dalmia",
        "exclude_name": None,
        "hq_region": "India",
        "business_mix": "pure_play",
        "core_example": False,
    },
    "Semen Indonesia": {
        "ticker": "SMGR.JK",
        "gem_regex": r"semen indonesia",
        "ct_regex": r"semen indonesia",
        "exclude_name": None,
        "hq_region": "Southeast Asia",
        "business_mix": "pure_play",
        "core_example": False,
    },
    "Siam Cement": {
        "ticker": "SCC.BK",
        "gem_regex": r"siam cement|\bscg\b",
        "ct_regex": r"siam cement|\bscg\b",
        "exclude_name": None,
        "hq_region": "Southeast Asia",
        "business_mix": "diversified_materials",
        "core_example": False,
    },
}

# Objective classification threshold on Geographic HHI (sum of squared country emission shares)
# HHI < 0.50 -> "Diversified Multinational" | HHI >= 0.50 -> "Regionally Concentrated"
HHI_DIVERSIFICATION_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# 3. NGFS SCENARIO & CARBON PRICE SETTINGS
# ---------------------------------------------------------------------------
NGFS_MODEL = "Downscaling[REMIND-MAgPIE 3.3-4.8]"
NGFS_CARBON_VAR_PRIMARY = "Price|Carbon|Demand|Industry"
NGFS_CARBON_VAR_FALLBACK = "Price|Carbon"

NGFS_SCENARIOS = [
    "Current Policies",
    "Delayed transition",
    "Net Zero 2050",
    "Fragmented World",
]

# Convert NGFS US$2010/tCO2 -> US$2026/tCO2 using BLS CPI-U (337.511 / 218.056)
CPI_2010_TO_2026 = 337.511 / 218.056  # ~1.5478

# Projection horizon for Climate VaR DCF
BASE_YEAR = 2025
PROJECTION_START = 2026
PROJECTION_END = 2040  # 15-year transition risk horizon (captures EU CBAM + post-2030 Delayed shock)

# ---------------------------------------------------------------------------
# 4. POLICY & FINANCIAL MODEL PARAMETERS
# ---------------------------------------------------------------------------
# EU ETS + UK ETS + Swiss ETS countries (ISO3) subject to CBAM free-allocation phase-out
EU_UK_CH_ETS_ISO3 = {
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "NOR", "ISL", "LIE",
    "GBR", "CHE",
}

# EU CBAM / ETS free-allocation phase-out schedule for cement (share of emissions FREE)
# Official EU CBAM factor: 97.5% free in 2026 -> 0% free by 2034
EU_FREE_ALLOCATION_SCHEDULE = {
    2026: 0.975,
    2027: 0.950,
    2028: 0.900,
    2029: 0.775,
    2030: 0.515,
    2031: 0.390,
    2032: 0.265,
    2033: 0.140,
    2034: 0.000,
}

# Baseline compliance/free-allocation rules for non-EU markets:
#   China ETS (cement intensity-based Phase 1 -> tightening Phase 2 from 2027+)
#   India CCTS (intensity-based from 2026-2027, tightening post-2030)
#   Other markets: effective priced-emissions fraction ramps up with NGFS shadow price
CHINA_FREE_ALLOCATION_BASE = 0.85   # intensity benchmark covers ~85% initially, tightens to 30% by 2040
INDIA_FREE_ALLOCATION_BASE = 0.90   # intensity benchmark covers ~90% initially, tightens to 40% by 2040
OTHER_FREE_ALLOCATION_BASE = 0.50   # conservative benchmark for other carbon pricing regimes

# Cost pass-through rate (share of carbon cost passed to customers; 1 - pass_through hits EBITDA)
# Cement is locally traded (high transport cost), allowing moderate pass-through (~30-60%)
PASS_THROUGH_DEFAULT = 0.40

# Baseline financial assumptions (overridden by company-specific WACC where beta/debt available)
RISK_FREE_RATE = 0.040        # 4.0% USD 10Y Treasury proxy
EQUITY_RISK_PREMIUM = 0.055   # 5.5% Damodaran global ERP
COST_OF_DEBT_PRETAX = 0.055   # 5.5% corporate bond proxy
TAX_RATE_DEFAULT = 0.25       # 25% corporate tax rate
WACC_DEFAULT = 0.080          # 8.0% fallback WACC

# Annual autonomous decarbonization rate (clinker substitution + energy efficiency)
ANNUAL_DECARB_RATE_BASE = 0.015  # 1.5%/yr baseline intensity reduction

# ---------------------------------------------------------------------------
# 5. UNCERTAINTY & MONTE CARLO PARAMETERS
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# Limited liability: equity and debt holders cannot lose more than the firm is worth.
# Climate VaR is capped at 100% of EV; firms hitting the cap are flagged "stranded"
# (the unpassed carbon bill exceeds the value of the current business model).
CVAR_CAP_PCT = 100.0
MC_DRAWS = 5_000

# Parameter uncertainty distributions for Monte Carlo & OAT sensitivity
MC_PASS_THROUGH_TRIANGULAR = (0.15, 0.40, 0.65)    # (low, mode, high)
MC_EMISSIONS_MULT_TRIANGULAR = (0.85, 1.00, 1.20)  # epistemic emissions / utilization multiplier
MC_CARBON_PRICE_MULT_TRIANGULAR = (0.75, 1.00, 1.30)
MC_WACC_STD = 0.012                                # +/- 1.2% WACC uncertainty

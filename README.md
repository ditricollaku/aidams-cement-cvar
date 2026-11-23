# Cross-Company Climate Value-at-Risk (Climate VaR) in Global Cement

**Course:** Research and Emerging Topics in Data Science: Climate Risks (AIDAMS, Fall 2026)  
**Role:** Climate Risk Analyst (Finance / Quant Track)

## Research Question
*How does transition-risk financial exposure (Climate VaR) differ between geographically diversified multinational cement producers (e.g., Holcim, Heidelberg Materials, CRH) and regionally concentrated producers (e.g., Anhui Conch, UltraTech Cement), and what does this imply for investment and portfolio risk?*

## Repository Structure
- `main.py` — Single executable entry point reproducing data validation, modeling, uncertainty analysis, summary tables, and figures.
- `src/config.py` — Central configuration (18-company universe and perimeter rules, tickers, NGFS scenarios, free-allocation schedules, discounting/tax conventions, Monte Carlo parameters and correlations, carbon-price sweep grid, and `RANDOM_SEED = 42`).
- `src/data_processing.py` — Raw Pandera schema validation, GEM equity-share entity resolution, NGFS carbon price extraction (`US$2010` to `US$2026`), and cached Yahoo Finance financials.
- `src/emissions_model.py` — Equity-weighted Climate TRACE historical emissions panel (2021–2026; 2026 is a partial year and is not used for modelling), Geographic HHI calculation, and GEM bottom-up reality check.
- `src/cvar_model.py` — Hybrid World Bank + NGFS carbon pricing, EU CBAM / China ETS / India CCTS free-allocation schedules, real-rate CAPM WACC discounting of after-tax carbon costs, Climate VaR (% EV), EBITDA-at-Risk, portfolio comparison, the US\$50–150/t flat carbon-price sweep, and robustness variants.
- `src/uncertainty.py` — Student-*t* historical back-test, 5,000-draw Monte Carlo simulation with firm-specific and bloc-specific draws, one-at-a-time (OAT) sensitivity, and IPCC calibrated likelihoods.
- `src/visualization.py` — Publication-ready 300-DPI figures (`outputs/figures/`) and interactive Plotly plant map.

## Dependencies
Managed via [Pixi](https://pixi.sh) (`pixi.toml` and `pixi.lock`), targeting Python 3.12 across `osx-arm64`, `osx-64`, `linux-64`, and `win-64`:
- **Data & Numerics:** `pandas`, `numpy`, `scipy`, `openpyxl`, `pyarrow`
- **Validation & Market Data:** `pandera`, `yfinance`
- **Visualization:** `matplotlib`, `plotly`

## Input Data Files
All required raw files are tracked in `data/raw/` so the pipeline runs out of the box after cloning. If you wish to re-download any dataset from its upstream source, place it at the exact path below:

| File Path | Source & Download Page | Purpose |
| :--- | :--- | :--- |
| `data/raw/gem_cement_july2026.xlsx` | [Global Energy Monitor — Global Cement and Concrete Tracker](https://globalenergymonitor.org/projects/global-cement-concrete-tracker) | Plant capacity, coordinates, ownership shares, and CCS status |
| `data/raw/emissions_co2_v5_12_0.csv` | [Climate TRACE — Data Downloads (Cement Emissions)](https://climatetrace.org/data) | Satellite/ML plant-level Scope 1 CO2 emissions (2021–2026) |
| `data/raw/ownership_v5_12_0.csv` | [Climate TRACE — Data Downloads (Cement Ownership)](https://climatetrace.org/data) | Plant-to-company equity ownership mapping |
| `data/raw/wb_carbon_pricing.csv` | [World Bank Carbon Pricing Dashboard](https://carbonpricingdashboard.worldbank.org/) | Implemented 2026 national/regional industrial carbon price floors |
| `data/raw/ngfs_downscaled_remind.xlsx` | [NGFS Phase 5 Scenario Explorer (IIASA)](https://data.ene.iiasa.ac.at/ngfs/) | Downscaled REMIND-MAgPIE country carbon price pathways (2026–2040) |
| `data/raw/financials_cache.csv` | Yahoo Finance (via `yfinance`, committed snapshot) | Cached Enterprise Value, EBITDA, and Beta for the 18 producers |

## Setup & Execution
1. Install Pixi (if not already installed) and create the locked environment:
   ```bash
   pixi install
   ```
2. Run the full end-to-end pipeline:
   ```bash
   pixi run main
   ```
   *(Equivalently: `pixi run python main.py`)*

### Command-Line Flags
- **`--refresh-ngfs`**: Forces re-parsing of the 50 MB raw workbook (`data/raw/ngfs_downscaled_remind.xlsx`) instead of reusing the extracted cache (`data/processed/ngfs_carbon_prices.csv`).
- **`--refresh-financials`**: Re-queries live corporate Enterprise Value, EBITDA, and Beta from Yahoo Finance instead of loading `data/raw/financials_cache.csv`.

## Outputs
Running `main.py` populates three directories:
- **`data/processed/`**: Cleaned plant/emissions panels (`gem_plants_clean.parquet`, `ct_plant_year.parquet`, `ct_company_country_year.csv`), `pandera_validation_report.csv`, `ngfs_carbon_prices.csv`, company/portfolio Climate VaR panels, and Monte Carlo interval datasets.
- **`outputs/tables/`**: Nine summary tables exported in both `.csv` and LaTeX (`.tex`) formats:
  - `table1_company_cvar` — Company-level Climate VaR (% EV) across all 4 NGFS scenarios + Delayed Transition 90% CI, stranding flags, and the stranded-asset indicator `old_unabated_cap_share` (share of equity clinker capacity in kilns ≥30 years old without CCS, from GEM).
  - `table2_tier_comparison` — Diversified Multinational vs. Regionally Concentrated median VaR, EBITDA-at-Risk, and HHI correlations.
  - `table3_portfolio_risk` — Equal-weighted & EV-weighted portfolio VaR and cross-firm tail dispersion.
  - `table4_oat_sensitivity` — One-at-a-time uncertainty sensitivity breakdown.
  - `table5_scenario_ipcc` — Cross-scenario intervals and IPCC calibrated likelihoods.
  - `table6_hhi_eu_control` — OLS of Climate VaR on HHI with and without EU/UK/CH ETS emissions share as a control, plus partial Spearman correlation (separates diversification from EU carbon-price exposure).
  - `table7_backtest` — Out-of-sample back-test of the emissions trend model (train 2021–2023, test 2024–2025): 90% prediction-interval coverage for the aggregate series and the 18-company panel, with a binomial test against nominal coverage.
  - `table8_price_sweep` — Flat carbon price of US\$50, 75, 100, 125 and 150/tCO₂ (constant 2026–2040, all jurisdictions) under current-policy and strict free-allocation paths: tier medians, universe Climate VaR, stranded firms, HHI–VaR correlation. Company-level results: `data/processed/price_sweep_company.csv`.
  - `table9_robustness` — Group results under alternative assumptions: (i) no NGFS shadow price charged where no carbon price is implemented (Current Policies / Fragmented World), (ii) China/India on the generic free-allocation path. Company-level results: `data/processed/robustness_company.csv`.
- **`outputs/figures/`**: Five 300-DPI PNG figures (`fig1_geospatial_footprint.png`, `fig2_hhi_vs_cvar.png`, `fig3_tornado_sensitivity.png`, `fig4_reality_check.png`, `fig_uncertainty.png`) plus `interactive_plant_map.html`.

## Key Modelling Conventions
- **Company perimeter:** plants are linked to the 18 parents through Climate TRACE ownership chains (name matching in `config.py`, including major renamed subsidiaries, e.g. CNBM's Tianshan/Nanfang units, UltraTech's Kesoram plants, Ambuja's Sanghi/Orient plants). Holcim excludes the US and Canada (Amrize spin-off, June 2025). When a chain contains a link with unknown ownership, the share is the product of the known links rather than 100%.
- **Discounting:** carbon prices are in constant US\$2026, so cash flows are discounted at a real rate: nominal CAPM WACC converted with the Fisher relation at `EXPECTED_INFLATION = 2.5%` (`DISCOUNT_AT_REAL_RATE`).
- **Taxes:** unpassed carbon costs are tax-deductible, so Climate VaR uses after-tax costs (`CARBON_COST_TAX_DEDUCTIBLE`); EBITDA-at-Risk is pre-tax.
- **Monte Carlo dependence:** pass-through and emissions multipliers are drawn per firm (pairwise correlation `MC_FIRM_CORR = 0.3`), carbon-price multipliers per pricing bloc (EU/UK/CH, China, India, Other; correlation `MC_PRICE_BLOC_CORR = 0.5`), and the WACC shock is common to all firms.

## Reproducibility Notes
- **Financial Snapshot:** `data/raw/financials_cache.csv` is a point-in-time snapshot from Yahoo Finance; its `snapshot_date` column records when it was taken (re-fetches stamp the current date). Running the default command reproduces the exact reported tables and figures; passing `--refresh-financials` pulls live market capitalizations and FX rates, which will change EV-normalized Climate VaR percentages.
- **Deterministic Simulation:** All stochastic Monte Carlo and OAT sensitivity draws in `src/uncertainty.py` use a fixed local RNG seeded with `RANDOM_SEED = 42` (`src/config.py`), ensuring tables regenerate identically across runs.

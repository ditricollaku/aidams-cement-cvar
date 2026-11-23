# Cross-Company Climate Value-at-Risk (Climate VaR) in Global Cement

**Course:** Research and Emerging Topics in Data Science: Climate Risks (AIDAMS, Fall 2026)  
**Role:** Climate Risk Analyst (Finance / Quant Track)

## Research Question
*How does transition-risk financial exposure (Climate VaR) differ between geographically diversified multinational cement producers (e.g., Holcim, Heidelberg Materials, CRH) and regionally concentrated producers (e.g., Anhui Conch, UltraTech Cement), and what does this imply for investment and portfolio risk?*

## Repository Structure
- `main.py` — Single executable entry point reproducing data validation, modeling, uncertainty analysis, summary tables, and figures.
- `src/config.py` — Central configuration (18-company universe, tickers, NGFS scenarios, free-allocation schedules, Monte Carlo parameters, and `RANDOM_SEED = 42`).
- `src/data_processing.py` — Raw Pandera schema validation, GEM equity-share entity resolution, NGFS carbon price extraction (`US$2010` to `US$2026`), and cached Yahoo Finance financials.
- `src/emissions_model.py` — Equity-weighted Climate TRACE historical emissions panel (2021–2026), Geographic HHI calculation, and GEM bottom-up reality check.
- `src/cvar_model.py` — Hybrid World Bank + NGFS carbon pricing, EU CBAM / China ETS / India CCTS free-allocation schedules, CAPM WACC discounting, Climate VaR (% EV), EBITDA-at-Risk, and portfolio comparison.
- `src/uncertainty.py` — Student-*t* historical back-test, 5,000-draw Monte Carlo simulation, one-at-a-time (OAT) sensitivity, and IPCC calibrated likelihoods.
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
- **`outputs/tables/`**: Five summary tables exported in both `.csv` and LaTeX (`.tex`) formats:
  - `table1_company_cvar` — Company-level Climate VaR (% EV) across all 4 NGFS scenarios + Delayed Transition 90% CI.
  - `table2_tier_comparison` — Diversified Multinational vs. Regionally Concentrated median VaR, EBITDA-at-Risk, and HHI correlations.
  - `table3_portfolio_risk` — Equal-weighted & EV-weighted portfolio VaR and cross-firm tail dispersion.
  - `table4_oat_sensitivity` — One-at-a-time uncertainty sensitivity breakdown.
  - `table5_scenario_ipcc` — Cross-scenario intervals and IPCC calibrated likelihoods.
- **`outputs/figures/`**: Five 300-DPI PNG figures (`fig1_geospatial_footprint.png`, `fig2_hhi_vs_cvar.png`, `fig3_tornado_sensitivity.png`, `fig4_reality_check.png`, `fig_uncertainty.png`) plus `interactive_plant_map.html`.

## Reproducibility Notes
- **Financial Snapshot:** `data/raw/financials_cache.csv` is a point-in-time snapshot from Yahoo Finance. Running the default command reproduces the exact reported tables and figures; passing `--refresh-financials` pulls live market capitalizations and FX rates, which will change EV-normalized Climate VaR percentages.
- **Deterministic Simulation:** All stochastic Monte Carlo and OAT sensitivity draws in `src/uncertainty.py` use a fixed local RNG seeded with `RANDOM_SEED = 42` (`src/config.py`), ensuring tables regenerate identically across runs.

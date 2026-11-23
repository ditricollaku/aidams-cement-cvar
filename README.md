# Cross-Company Climate Value-at-Risk (Climate VaR) in Global Cement

**Course:** Research and Emerging Topics in Data Science: Climate Risks (AIDAMS, Fall 2026)  
**Role:** Climate Risk Analyst (Finance / Quant Track)  

## Research Question
*How does transition-risk financial exposure (Climate VaR) differ between geographically diversified multinational cement producers (e.g., Holcim, Heidelberg Materials, CRH) and regionally concentrated producers (e.g., Anhui Conch, UltraTech Cement), and what does this imply for investment and portfolio risk?*

## Repository Structure
- `main.py`: Single executable entry point reproducing all processed datasets, tables, and figures.
- `src/config.py`: Central configuration (18-company universe, tickers, NGFS scenarios, World Bank floors, Monte Carlo parameters).
- `src/data_processing.py`: Raw Pandera schema validation, GEM equity-share entity resolution, NGFS carbon price extraction (`US$2010` to `US$2026`), and cached Yahoo Finance corporate financials.
- `src/emissions_model.py`: Equity-weighted Climate TRACE historical emissions panel (2021–2026), Geographic HHI calculation, and GEM bottom-up reality check.
- `src/cvar_model.py`: Hybrid World Bank + NGFS carbon pricing, EU CBAM / China ETS / India CCTS free-allocation schedules, CAPM WACC discounting, Climate VaR (% EV), EBITDA-at-Risk, and portfolio basket comparison.
- `src/uncertainty.py`: Student-*t* historical back-test, 5,000-draw Monte Carlo simulation, one-at-a-time (OAT) sensitivity, and IPCC calibrated likelihoods.
- `src/visualization.py`: Publication-ready 300-DPI figures (`outputs/figures/`) and interactive Plotly map.

## Setup & Execution
1. Install dependencies using [Pixi](https://pixi.sh):
   ```bash
   pixi install

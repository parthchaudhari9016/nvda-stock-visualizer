# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project overview
- Small Python script that downloads 6 months of NVIDIA (NVDA) daily prices via yfinance, computes a 20‑day SMA, and plots/saves a PNG chart.
- Entrypoint: src/plot_nvda.py → main(). Output written to nvda_6mo.png in the repo root and the chart is shown interactively.

Common commands
- Create and activate a virtual environment
  - PowerShell:
    - python -m venv .venv
    - .venv\Scripts\Activate.ps1
  - Bash:
    - python -m venv .venv
    - source .venv/bin/activate
- Install dependencies
  - pip install -r requirements.txt
  - (optional dev tools) pip install -r requirements-dev.txt
- Run the plotter
  - python src/plot_nvda.py
- Lint
  - ruff check .
  - black .
- Tests
  - pytest
  - Run a single test: pytest tests/test_plot_nvda.py::test_compute_sma_basic -q

Architecture and workflow
- Data acquisition: yfinance.download(tickers="NVDA", period="6mo", interval="1d", auto_adjust=True, progress=False)
  - Exits early if no data is returned (internet/ticker issues).
- Transformation: extract Close series; compute 20‑day SMA with pandas rolling(min_periods=1).
- Visualization: Matplotlib line plot for Close and SMA 20; grid/labels; tight_layout; savefig to Path(__file__).resolve().parents[1] / "nvda_6mo.png"; then plt.show().
- Configuration points: in src/plot_nvda.py: change defaults in main(symbol, period, interval) or pass -- not CLI yet, call from Python to override.

Notes
- Requires internet connectivity for data download.
- Running will both save the image and display an interactive window (may block in headless environments).

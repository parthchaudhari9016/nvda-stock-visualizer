# NVDA Stock Visualizer (6 months)

Visualize the last 6 months of NVIDIA (NVDA) daily prices using Matplotlib.

## Quickstart

1) Create a virtual environment
- PowerShell:
  - `python -m venv .venv`
  - `.venv\Scripts\Activate.ps1`
- Bash (optional):
  - `python -m venv .venv`
  - `source .venv/bin/activate`

2) Install dependencies

```
pip install -r requirements.txt
```

3) Run the plotter

```
python src\plot_nvda.py
```

This saves `nvda_6mo.png` in the project root and shows the chart.

## Notes
- Data is fetched via `yfinance` for ticker `NVDA`, period `6mo`, interval `1d`.
- The script plots the Close price and a 20-day simple moving average.

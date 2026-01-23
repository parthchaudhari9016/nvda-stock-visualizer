<p align="center">
  <h1 align="center">📈 NVDA Stock Analyzer</h1>
  <p align="center">
    <strong>Professional Quantitative Analysis Toolkit for Stock Market Visualization</strong>
  </p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#cli-usage">CLI Usage</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#technical-indicators">Indicators</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Version-2.0.0-green?style=for-the-badge" alt="Version">
</p>

---

## 🎯 Overview

NVDA Stock Analyzer is a professional-grade Python toolkit designed for quantitative analysts, traders, and data scientists. It provides comprehensive tools for fetching market data, calculating technical indicators, performing statistical analysis, and generating publication-ready visualizations and reports.

### ✨ Key Highlights

- **20+ Technical Indicators** - SMA, EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, and more
- **Advanced Risk Metrics** - VaR, CVaR, Sharpe Ratio, Sortino Ratio, Maximum Drawdown
- **Professional Charts** - Candlestick, multi-panel technical analysis, comparison charts
- **Multi-Format Reports** - Markdown, HTML, JSON, and plain text exports
- **Intelligent Caching** - Reduces API calls with automatic data caching
- **Beautiful CLI** - Rich terminal interface with colored output

---

## 🚀 Features

### 📊 Technical Indicators

| Category | Indicators |
|----------|------------|
| **Trend** | SMA, EMA, WMA, DEMA |
| **Momentum** | RSI, MACD, Stochastic, ROC, CCI |
| **Volatility** | Bollinger Bands, ATR, Keltner Channels |
| **Volume** | OBV, VWAP, MFI |

### 📈 Statistical Analysis

| Metric | Description |
|--------|-------------|
| **Performance** | Total Return, Annualized Return, Volatility |
| **Risk-Adjusted** | Sharpe Ratio, Sortino Ratio, Calmar Ratio |
| **Risk** | VaR (95%, 99%), CVaR, Maximum Drawdown |
| **Distribution** | Skewness, Kurtosis, Percentiles |

### 🎨 Visualizations

- **Price Charts** with volume overlay
- **Candlestick Charts** with SMA overlays  
- **Technical Analysis Dashboards** with MACD, RSI, Bollinger Bands
- **Performance Comparison** charts for multiple symbols
- **Dark theme** optimized for professional use

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/parthchaudhari9016/nvda-stock-visualizer.git
cd nvda-stock-visualizer

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Development Install

```bash
# Install with dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Or using pip editable install
pip install -e ".[dev]"
```

---

## ⚡ Quick Start

### Python API

```python
from src import DataFetcher, TechnicalIndicators, StatisticalAnalysis, StockVisualizer

# Fetch data
fetcher = DataFetcher()
data = fetcher.fetch("NVDA", period="1y", interval="1d")

# Add technical indicators
data_with_indicators = TechnicalIndicators.add_all_indicators(data)

# Perform statistical analysis
analyzer = StatisticalAnalysis(data)
performance = analyzer.calculate_performance_metrics()
print(f"Total Return: {performance.total_return}%")
print(f"Sharpe Ratio: {performance.sharpe_ratio}")

# Generate visualization
viz = StockVisualizer(data, symbol="NVDA")
viz.plot_technical_analysis()
viz.save("nvda_analysis.png")
viz.show()
```

### Generate Reports

```python
from src.report_generator import ReportGenerator

generator = ReportGenerator(data, symbol="NVDA", period="1y")

# Generate different formats
generator.generate_markdown_report("report.md")
generator.generate_html_report("report.html")
generator.generate_json_report("report.json")

# Get text summary
print(generator.generate_text_summary())
```

---

## 💻 CLI Usage

The toolkit includes a powerful command-line interface with rich terminal output.

### Analyze a Stock

```bash
# Full analysis with chart output
python -m src.cli analyze NVDA --period 1y --output analysis.png

# Quick analysis
python -m src.cli analyze AAPL
```

### Generate Charts

```bash
# Technical analysis chart
python -m src.cli chart NVDA --type technical --days 180

# Candlestick chart
python -m src.cli chart NVDA --type candlestick --days 60 --output candles.png

# Price chart
python -m src.cli chart MSFT --type price
```

### Get Stock Information

```bash
python -m src.cli info NVDA
```

### CLI Options

| Command | Options | Description |
|---------|---------|-------------|
| `analyze` | `--period`, `--interval`, `--output` | Full statistical analysis |
| `chart` | `--type`, `--days`, `--output` | Generate visualizations |
| `info` | - | Display stock information |

---

## 📚 API Reference

### DataFetcher

```python
from src import DataFetcher

fetcher = DataFetcher()

# Basic fetch
data = fetcher.fetch("NVDA", period="1y", interval="1d")

# Fetch multiple symbols
multi_data = fetcher.fetch_multiple(["NVDA", "AMD", "INTC"], period="6mo")

# Get stock info
info = fetcher.get_stock_info("NVDA")

# Cache management
fetcher.clear_cache()
```

### TechnicalIndicators

```python
from src import TechnicalIndicators as TI

# Individual indicators
sma_20 = TI.sma(data["Close"], period=20)
ema_12 = TI.ema(data["Close"], period=12)
rsi = TI.rsi(data["Close"], period=14)

# MACD (returns tuple)
macd_line, signal_line, histogram = TI.macd(data["Close"])

# Bollinger Bands (returns tuple)
upper, middle, lower = TI.bollinger_bands(data["Close"])

# Add all indicators to DataFrame
data_enriched = TI.add_all_indicators(data)
```

### StatisticalAnalysis

```python
from src import StatisticalAnalysis

analyzer = StatisticalAnalysis(data, price_column="Close")

# Get statistics
stats = analyzer.get_descriptive_stats()
performance = analyzer.calculate_performance_metrics()
risk = analyzer.calculate_risk_metrics()

# Returns analysis
daily_returns = analyzer.calculate_returns("daily")
cumulative = analyzer.calculate_cumulative_returns()

# Complete summary
summary = analyzer.generate_summary()
```

### StockVisualizer

```python
from src import StockVisualizer

viz = StockVisualizer(data, symbol="NVDA")

# Different chart types
viz.plot_price(show_volume=True, show_sma=[20, 50])
viz.plot_candlestick(tail_days=60)
viz.plot_technical_analysis(indicators=["macd", "rsi", "volume"])

# Save and display
viz.save("chart.png", dpi=300)
viz.show()
viz.close()
```

---

## 📐 Technical Indicators

### Trend Indicators

| Indicator | Function | Parameters |
|-----------|----------|------------|
| Simple Moving Average | `TI.sma(series, period)` | period=20 |
| Exponential Moving Average | `TI.ema(series, period)` | period=20 |
| Weighted Moving Average | `TI.wma(series, period)` | period=20 |
| Double EMA | `TI.dema(series, period)` | period=20 |

### Momentum Indicators

| Indicator | Function | Parameters |
|-----------|----------|------------|
| RSI | `TI.rsi(series, period)` | period=14 |
| MACD | `TI.macd(series, fast, slow, signal)` | 12, 26, 9 |
| Stochastic | `TI.stochastic(high, low, close, k, d)` | 14, 3 |
| Rate of Change | `TI.roc(series, period)` | period=12 |
| CCI | `TI.cci(high, low, close, period)` | period=20 |

### Volatility Indicators

| Indicator | Function | Parameters |
|-----------|----------|------------|
| Bollinger Bands | `TI.bollinger_bands(series, period, std)` | 20, 2.0 |
| ATR | `TI.atr(high, low, close, period)` | period=14 |
| Keltner Channels | `TI.keltner_channels(...)` | 20, 10, 2.0 |
| Historical Volatility | `TI.historical_volatility(series, period)` | period=20 |

### Volume Indicators

| Indicator | Function | Parameters |
|-----------|----------|------------|
| On-Balance Volume | `TI.obv(close, volume)` | - |
| VWAP | `TI.vwap(high, low, close, volume)` | - |
| Money Flow Index | `TI.mfi(high, low, close, volume, period)` | period=14 |

---

## 📁 Project Structure

```
nvda-stock-visualizer/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── cli.py                # Command-line interface
│   ├── config.py             # Configuration management
│   ├── data_fetcher.py       # Market data acquisition
│   ├── technical_indicators.py  # Technical analysis
│   ├── statistical_analysis.py  # Statistical metrics
│   ├── visualizer.py         # Chart generation
│   └── report_generator.py   # Report exports
├── tests/
│   └── test_*.py             # Unit tests
├── requirements.txt          # Dependencies
├── requirements-dev.txt      # Dev dependencies
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_indicators.py -v
```

---

## 🔧 Configuration

The toolkit uses a centralized configuration system:

```python
from src.config import get_config, Config, Theme

config = get_config()

# Customize chart settings
config.chart.theme = Theme.DARK
config.chart.figsize = (14, 10)
config.chart.dpi = 150

# Customize analysis parameters
config.analysis.sma_periods = [10, 20, 50]
config.analysis.rsi_period = 14
config.analysis.risk_free_rate = 0.05
```

---

## 📊 Sample Output

### Performance Metrics
```
====================================================
  NVDA STOCK ANALYSIS SUMMARY
====================================================

Period: 2024-01-01 to 2025-01-24

PRICE:
  Start: $481.52  →  End: $142.89
  High: $152.89   |  Low: $75.61

RETURNS:
  Total Return: 85.42%
  Annualized: 78.23%
  Volatility: 45.12%

RISK-ADJUSTED:
  Sharpe Ratio: 1.523
  Sortino Ratio: 2.145
  Max Drawdown: -15.23%

RISK:
  VaR (95%): -3.21%
  CVaR (95%): -4.87%
====================================================
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Make your changes with tests
4. Run linting (`ruff check . && black .`)
5. Commit (`git commit -m 'Add feature'`)
6. Push (`git push origin feature/enhancement`)
7. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⚠️ Disclaimer

This software is for educational and informational purposes only. It is not intended to provide investment advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

---

<p align="center">
  <strong>Built with ❤️ for quantitative analysis</strong>
  <br>
  <sub>Made by Parth Chaudhari</sub>
</p>

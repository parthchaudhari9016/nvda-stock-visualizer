from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


def fetch_price_data(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Download price data for a symbol using yfinance.

    Returns a DataFrame with at least a 'Close' column.
    """
    print(f"Downloading {symbol} data for last {period} at {interval} interval...")
    data = yf.download(
        tickers=symbol, period=period, interval=interval, auto_adjust=True, progress=False
    )
    if data.empty:
        raise SystemExit("No data returned. Check internet connection or ticker symbol.")
    return data


def compute_sma(close: pd.Series, window: int = 20) -> pd.Series:
    """Compute simple moving average over the provided close series."""
    return close.rolling(window=window, min_periods=1).mean()


def make_plot(
    close: pd.Series,
    sma: pd.Series,
    title: str,
    out_path: Path,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.plot(close.index, close.values, label="Close", color="#2c7fb8")
    ax.plot(sma.index, sma.values, label="SMA 20", color="#f03b20")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

    if show:
        plt.show()


def main(
    symbol: str = "NVDA",
    period: str = "6mo",
    interval: str = "1d",
    out_file: Optional[str] = None,
) -> Path:
    data = fetch_price_data(symbol=symbol, period=period, interval=interval)
    close = data["Close"].copy()
    sma20 = compute_sma(close, window=20)

    out_path = Path(out_file) if out_file else Path(__file__).resolve().parents[1] / "nvda_6mo.png"
    make_plot(close, sma20, f"{symbol} Closing Price - Last 6 Months", out_path, show=True)
    return out_path


if __name__ == "__main__":
    main()

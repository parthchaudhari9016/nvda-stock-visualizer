import matplotlib
import pandas as pd

# Use non-interactive backend for tests
matplotlib.use("Agg", force=True)

import src.plot_nvda as plot_nvda


def test_compute_sma_basic():
    close = pd.Series([1.0, 3.0, 5.0])
    sma2 = plot_nvda.compute_sma(close, window=2)
    # Expected: [1.0, (1+3)/2, (3+5)/2] = [1.0, 2.0, 4.0]
    assert sma2.tolist() == [1.0, 2.0, 4.0]


def test_make_plot_saves_file(tmp_path):
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    close = pd.Series([10.0, 11.0, 12.0], index=idx)
    sma = plot_nvda.compute_sma(close, window=2)
    out = tmp_path / "figure.png"

    plot_nvda.make_plot(close, sma, title="Test", out_path=out, show=False)

    assert out.exists()
    assert out.stat().st_size > 0


def test_fetch_price_data_monkeypatched(monkeypatch):
    # Stub yfinance.download to avoid network
    def fake_download(tickers, period, interval, auto_adjust, progress):
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        return pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)

    monkeypatch.setattr(plot_nvda.yf, "download", fake_download)

    df = plot_nvda.fetch_price_data("NVDA", period="6mo", interval="1d")
    assert not df.empty
    assert "Close" in df.columns

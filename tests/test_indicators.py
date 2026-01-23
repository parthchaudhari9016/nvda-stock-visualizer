"""
Tests for Technical Indicators module.
"""

import pandas as pd
import numpy as np
import pytest

from src.technical_indicators import TechnicalIndicators as TI


class TestMovingAverages:
    """Tests for moving average indicators."""
    
    def test_sma_basic(self):
        """Test basic SMA calculation."""
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = TI.sma(close, period=3)
        
        # First value uses min_periods=1
        assert sma.iloc[0] == 1.0
        assert sma.iloc[1] == 1.5
        assert sma.iloc[2] == 2.0  # (1+2+3)/3
        assert sma.iloc[3] == 3.0  # (2+3+4)/3
        assert sma.iloc[4] == 4.0  # (3+4+5)/3
    
    def test_ema_basic(self):
        """Test basic EMA calculation."""
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ema = TI.ema(close, period=3)
        
        assert len(ema) == 5
        # EMA should be between min and max of series
        assert ema.iloc[-1] > close.mean()
    
    def test_wma_basic(self):
        """Test basic WMA calculation."""
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        wma = TI.wma(close, period=3)
        
        # WMA at index 2: (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6 = 2.33
        assert abs(wma.iloc[2] - 2.333) < 0.01


class TestMomentumIndicators:
    """Tests for momentum indicators."""
    
    def test_rsi_bounds(self):
        """Test RSI is between 0 and 100."""
        close = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                          110, 108, 106, 104, 102, 100, 98, 96, 95, 97])
        rsi = TI.rsi(close, period=14)
        
        assert rsi.min() >= 0
        assert rsi.max() <= 100
    
    def test_macd_components(self):
        """Test MACD returns correct components."""
        close = pd.Series(np.random.randn(50).cumsum() + 100)
        macd_line, signal_line, histogram = TI.macd(close)
        
        assert len(macd_line) == 50
        assert len(signal_line) == 50
        assert len(histogram) == 50
        
        # Histogram should be MACD - Signal
        assert abs((macd_line - signal_line - histogram).sum()) < 0.001
    
    def test_stochastic_bounds(self):
        """Test Stochastic is between 0 and 100."""
        high = pd.Series([101, 103, 105, 104, 106, 108, 107, 109, 111, 110,
                         112, 114, 113, 115, 117])
        low = pd.Series([99, 101, 103, 102, 104, 106, 105, 107, 109, 108,
                        110, 112, 111, 113, 115])
        close = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                          111, 113, 112, 114, 116])
        
        stoch_k, stoch_d = TI.stochastic(high, low, close)
        
        assert stoch_k.min() >= 0
        assert stoch_k.max() <= 100


class TestVolatilityIndicators:
    """Tests for volatility indicators."""
    
    def test_bollinger_bands_order(self):
        """Test Bollinger Bands upper > middle > lower."""
        close = pd.Series(np.random.randn(50).cumsum() + 100)
        upper, middle, lower = TI.bollinger_bands(close, period=20)
        
        # After warmup period, upper should be > middle > lower
        assert (upper.iloc[20:] >= middle.iloc[20:]).all()
        assert (middle.iloc[20:] >= lower.iloc[20:]).all()
    
    def test_atr_positive(self):
        """Test ATR is always positive."""
        high = pd.Series([101, 103, 105, 104, 106, 108, 107, 109, 111, 110,
                         112, 114, 113, 115, 117, 116, 118, 120, 119, 121])
        low = pd.Series([99, 101, 103, 102, 104, 106, 105, 107, 109, 108,
                        110, 112, 111, 113, 115, 114, 116, 118, 117, 119])
        close = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
                          111, 113, 112, 114, 116, 115, 117, 119, 118, 120])
        
        atr = TI.atr(high, low, close, period=14)
        
        assert (atr.dropna() > 0).all()


class TestVolumeIndicators:
    """Tests for volume indicators."""
    
    def test_obv_calculation(self):
        """Test OBV calculation logic."""
        close = pd.Series([100, 101, 100, 102, 101])
        volume = pd.Series([1000, 1000, 1000, 1000, 1000])
        
        obv = TI.obv(close, volume)
        
        # OBV should increase on up days, decrease on down days
        assert obv.iloc[1] > obv.iloc[0]  # Price went up
        assert obv.iloc[2] < obv.iloc[1]  # Price went down
    
    def test_vwap_calculation(self):
        """Test VWAP is within price range."""
        high = pd.Series([105, 107, 106, 108, 110])
        low = pd.Series([100, 102, 101, 103, 105])
        close = pd.Series([102, 105, 103, 106, 108])
        volume = pd.Series([1000, 1500, 1200, 1800, 2000])
        
        vwap = TI.vwap(high, low, close, volume)
        
        # VWAP should be between lowest low and highest high
        assert vwap.iloc[-1] >= low.min()
        assert vwap.iloc[-1] <= high.max()


class TestAddAllIndicators:
    """Test the convenience function to add all indicators."""
    
    def test_add_all_indicators(self):
        """Test that all expected columns are added."""
        # Create sample OHLCV data
        dates = pd.date_range("2024-01-01", periods=250, freq="D")
        data = pd.DataFrame({
            "Open": np.random.randn(250).cumsum() + 100,
            "High": np.random.randn(250).cumsum() + 102,
            "Low": np.random.randn(250).cumsum() + 98,
            "Close": np.random.randn(250).cumsum() + 100,
            "Volume": np.random.randint(1000000, 5000000, 250),
        }, index=dates)
        
        # Ensure High > Low
        data["High"] = data[["Open", "Close"]].max(axis=1) + 2
        data["Low"] = data[["Open", "Close"]].min(axis=1) - 2
        
        enriched = TI.add_all_indicators(data)
        
        expected_columns = [
            "SMA_20", "SMA_50", "SMA_200",
            "EMA_12", "EMA_26",
            "RSI", "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Upper", "BB_Middle", "BB_Lower",
            "ATR", "OBV", "VWAP"
        ]
        
        for col in expected_columns:
            assert col in enriched.columns, f"Missing column: {col}"

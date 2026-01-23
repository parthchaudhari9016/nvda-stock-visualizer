"""
Technical Indicators Module - Comprehensive technical analysis calculations.

Provides a wide range of technical indicators commonly used in
quantitative trading and stock analysis.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


class TechnicalIndicators:
    """
    Comprehensive technical analysis indicator calculations.
    
    All methods are static and operate on Pandas Series/DataFrames,
    making them easy to chain and compose.
    
    Categories:
    - Trend Indicators: SMA, EMA, WMA
    - Momentum Indicators: RSI, MACD, Stochastic, ROC
    - Volatility Indicators: Bollinger Bands, ATR, Standard Deviation
    - Volume Indicators: OBV, VWAP
    
    Example:
        >>> from technical_indicators import TechnicalIndicators as TI
        >>> sma_20 = TI.sma(close_prices, period=20)
        >>> rsi = TI.rsi(close_prices, period=14)
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TREND INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def sma(series: pd.Series, period: int = 20) -> pd.Series:
        """
        Simple Moving Average (SMA).
        
        The arithmetic mean of prices over a specified period.
        
        Args:
            series: Price series (typically Close)
            period: Lookback period
            
        Returns:
            SMA series
        """
        return series.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int = 20, adjust: bool = True) -> pd.Series:
        """
        Exponential Moving Average (EMA).
        
        Weighted average that gives more importance to recent prices.
        
        Args:
            series: Price series
            period: Lookback period
            adjust: Whether to use adjusted weights
            
        Returns:
            EMA series
        """
        return series.ewm(span=period, adjust=adjust, min_periods=1).mean()
    
    @staticmethod
    def wma(series: pd.Series, period: int = 20) -> pd.Series:
        """
        Weighted Moving Average (WMA).
        
        Linear weighted average giving more weight to recent prices.
        
        Args:
            series: Price series
            period: Lookback period
            
        Returns:
            WMA series
        """
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def dema(series: pd.Series, period: int = 20) -> pd.Series:
        """
        Double Exponential Moving Average (DEMA).
        
        Reduces lag compared to traditional EMA.
        
        Args:
            series: Price series
            period: Lookback period
            
        Returns:
            DEMA series
        """
        ema1 = TechnicalIndicators.ema(series, period)
        ema2 = TechnicalIndicators.ema(ema1, period)
        return 2 * ema1 - ema2
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MOMENTUM INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI).
        
        Momentum oscillator measuring speed and magnitude of price changes.
        Values range from 0 to 100. Generally:
        - > 70: Overbought
        - < 30: Oversold
        
        Args:
            series: Price series (typically Close)
            period: Lookback period (default 14)
            
        Returns:
            RSI series (0-100)
        """
        delta = series.diff()
        
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD).
        
        Trend-following momentum indicator showing relationship
        between two moving averages.
        
        Args:
            series: Price series
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = TechnicalIndicators.ema(series, fast)
        ema_slow = TechnicalIndicators.ema(series, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.
        
        Momentum indicator comparing closing price to price range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period (default 14)
            d_period: %D smoothing period (default 3)
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
        
        return stoch_k, stoch_d
    
    @staticmethod
    def roc(series: pd.Series, period: int = 12) -> pd.Series:
        """
        Rate of Change (ROC).
        
        Percentage change between current price and price n periods ago.
        
        Args:
            series: Price series
            period: Lookback period
            
        Returns:
            ROC series (percentage)
        """
        return ((series - series.shift(period)) / series.shift(period)) * 100
    
    @staticmethod
    def momentum(series: pd.Series, period: int = 10) -> pd.Series:
        """
        Momentum Indicator.
        
        Difference between current price and price n periods ago.
        
        Args:
            series: Price series
            period: Lookback period
            
        Returns:
            Momentum series
        """
        return series - series.shift(period)
    
    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """
        Commodity Channel Index (CCI).
        
        Measures deviation of price from average.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period
            
        Returns:
            CCI series
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period, min_periods=1).mean()
        mean_dev = typical_price.rolling(window=period, min_periods=1).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        
        return (typical_price - sma_tp) / (0.015 * mean_dev + 1e-10)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VOLATILITY INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Volatility bands placed above and below a moving average.
        
        Args:
            series: Price series
            period: SMA period (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle = TechnicalIndicators.sma(series, period)
        std = series.rolling(window=period, min_periods=1).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Average True Range (ATR).
        
        Measures market volatility by decomposing the entire range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period (default 14)
            
        Returns:
            ATR series
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return true_range.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def keltner_channels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels.
        
        Volatility-based envelope set above and below an EMA.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            ema_period: EMA period
            atr_period: ATR period
            multiplier: ATR multiplier
            
        Returns:
            Tuple of (Upper, Middle, Lower)
        """
        middle = TechnicalIndicators.ema(close, ema_period)
        atr_val = TechnicalIndicators.atr(high, low, close, atr_period)
        
        upper = middle + (multiplier * atr_val)
        lower = middle - (multiplier * atr_val)
        
        return upper, middle, lower
    
    @staticmethod
    def historical_volatility(
        series: pd.Series,
        period: int = 20,
        trading_days: int = 252,
    ) -> pd.Series:
        """
        Historical Volatility (Annualized).
        
        Standard deviation of log returns, annualized.
        
        Args:
            series: Price series
            period: Lookback period
            trading_days: Trading days per year
            
        Returns:
            Annualized volatility series
        """
        log_returns = np.log(series / series.shift(1))
        volatility = log_returns.rolling(window=period, min_periods=1).std()
        return volatility * np.sqrt(trading_days)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VOLUME INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume (OBV).
        
        Cumulative volume indicator based on price direction.
        
        Args:
            close: Close prices
            volume: Volume
            
        Returns:
            OBV series
        """
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        return (volume * direction).cumsum()
    
    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP).
        
        Average price weighted by volume.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            
        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        cumulative_tp_volume = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        
        return cumulative_tp_volume / cumulative_volume
    
    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Money Flow Index (MFI).
        
        Volume-weighted RSI, oscillates between 0 and 100.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            period: Lookback period
            
        Returns:
            MFI series (0-100)
        """
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        
        tp_diff = typical_price.diff()
        
        positive_flow = raw_money_flow.where(tp_diff > 0, 0)
        negative_flow = raw_money_flow.where(tp_diff < 0, 0)
        
        positive_mf = positive_flow.rolling(window=period, min_periods=1).sum()
        negative_mf = negative_flow.rolling(window=period, min_periods=1).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        
        return mfi
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def add_all_indicators(
        df: pd.DataFrame,
        include_volume: bool = True,
    ) -> pd.DataFrame:
        """
        Add all common indicators to a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            include_volume: Whether to include volume indicators
            
        Returns:
            DataFrame with added indicator columns
        """
        result = df.copy()
        
        # Trend
        result["SMA_20"] = TechnicalIndicators.sma(df["Close"], 20)
        result["SMA_50"] = TechnicalIndicators.sma(df["Close"], 50)
        result["SMA_200"] = TechnicalIndicators.sma(df["Close"], 200)
        result["EMA_12"] = TechnicalIndicators.ema(df["Close"], 12)
        result["EMA_26"] = TechnicalIndicators.ema(df["Close"], 26)
        
        # Momentum
        result["RSI"] = TechnicalIndicators.rsi(df["Close"])
        macd, signal, hist = TechnicalIndicators.macd(df["Close"])
        result["MACD"] = macd
        result["MACD_Signal"] = signal
        result["MACD_Hist"] = hist
        
        # Volatility
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df["Close"])
        result["BB_Upper"] = bb_upper
        result["BB_Middle"] = bb_middle
        result["BB_Lower"] = bb_lower
        result["ATR"] = TechnicalIndicators.atr(df["High"], df["Low"], df["Close"])
        
        # Volume indicators
        if include_volume and "Volume" in df.columns:
            result["OBV"] = TechnicalIndicators.obv(df["Close"], df["Volume"])
            result["VWAP"] = TechnicalIndicators.vwap(
                df["High"], df["Low"], df["Close"], df["Volume"]
            )
        
        return result

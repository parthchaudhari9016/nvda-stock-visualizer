"""
Data Fetcher Module - Robust market data acquisition with caching.

Handles downloading, caching, and preprocessing of stock market data
from multiple sources with comprehensive error handling.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import yfinance as yf

from .config import get_config, Config

logger = logging.getLogger(__name__)


class DataFetcherError(Exception):
    """Custom exception for data fetching errors."""
    pass


class DataFetcher:
    """
    Robust stock data fetcher with caching and validation.
    
    Features:
    - Automatic data caching to reduce API calls
    - Data validation and cleaning
    - Multiple timeframe support
    - Error handling with informative messages
    
    Example:
        >>> fetcher = DataFetcher()
        >>> data = fetcher.fetch("NVDA", period="1y")
        >>> print(data.head())
    """
    
    VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    
    def __init__(self, config: Optional[Config] = None, cache_dir: Optional[str] = None):
        """
        Initialize the DataFetcher.
        
        Args:
            config: Configuration object (uses default if None)
            cache_dir: Directory for caching data (defaults to .cache)
        """
        self.config = config or get_config()
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache")
        
        if self.config.cache_enabled:
            self.cache_dir.mkdir(exist_ok=True)
    
    def fetch(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch stock data for the given symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., "NVDA", "AAPL")
            period: Historical data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            DataFetcherError: If data cannot be fetched or is invalid
        """
        symbol = symbol.upper().strip()
        self._validate_params(period, interval)
        
        # Check cache first
        if use_cache and self.config.cache_enabled:
            cached = self._load_from_cache(symbol, period, interval)
            if cached is not None:
                logger.info(f"Loaded {symbol} data from cache")
                return cached
        
        # Fetch from API
        logger.info(f"Fetching {symbol} data: period={period}, interval={interval}")
        
        try:
            data = yf.download(
                tickers=symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as e:
            raise DataFetcherError(f"Failed to download data for {symbol}: {e}")
        
        if data.empty:
            raise DataFetcherError(
                f"No data returned for {symbol}. Check internet connection or ticker symbol."
            )
        
        # Clean and validate data
        data = self._clean_data(data)
        
        # Cache the data
        if self.config.cache_enabled:
            self._save_to_cache(data, symbol, period, interval)
        
        return data
    
    def fetch_multiple(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            period: Historical data period
            interval: Data interval
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch(symbol, period, interval)
            except DataFetcherError as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                results[symbol] = None
        return results
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "symbol": symbol,
                "name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", "N/A"),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "avg_volume": info.get("averageVolume", 0),
            }
        except Exception as e:
            logger.warning(f"Could not fetch info for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def _validate_params(self, period: str, interval: str) -> None:
        """Validate period and interval parameters."""
        if period not in self.VALID_PERIODS:
            raise DataFetcherError(
                f"Invalid period '{period}'. Valid options: {self.VALID_PERIODS}"
            )
        if interval not in self.VALID_INTERVALS:
            raise DataFetcherError(
                f"Invalid interval '{interval}'. Valid options: {self.VALID_INTERVALS}"
            )
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the data."""
        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Ensure required columns exist
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise DataFetcherError(f"Missing required columns: {missing}")
        
        # Remove rows with NaN in critical columns
        data = data.dropna(subset=["Open", "High", "Low", "Close"])
        
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    def _get_cache_path(self, symbol: str, period: str, interval: str) -> Path:
        """Get the cache file path for given parameters."""
        return self.cache_dir / f"{symbol}_{period}_{interval}.parquet"
    
    def _load_from_cache(
        self, symbol: str, period: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        cache_path = self._get_cache_path(symbol, period, interval)
        
        if not cache_path.exists():
            return None
        
        # Check if cache is expired
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(hours=self.config.cache_expiry_hours):
            logger.info(f"Cache expired for {symbol}")
            return None
        
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(
        self, data: pd.DataFrame, symbol: str, period: str, interval: str
    ) -> None:
        """Save data to cache."""
        try:
            cache_path = self._get_cache_path(symbol, period, interval)
            data.to_parquet(cache_path)
            logger.debug(f"Saved {symbol} to cache")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """
        Clear cached data.
        
        Args:
            symbol: Specific symbol to clear, or None to clear all
            
        Returns:
            Number of cache files deleted
        """
        count = 0
        pattern = f"{symbol}_*.parquet" if symbol else "*.parquet"
        
        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            count += 1
        
        logger.info(f"Cleared {count} cache files")
        return count

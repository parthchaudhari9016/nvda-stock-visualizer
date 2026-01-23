"""
Tests for Data Fetcher module.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.data_fetcher import DataFetcher, DataFetcherError


class TestDataFetcherValidation:
    """Tests for parameter validation."""
    
    def test_invalid_period(self):
        """Test that invalid period raises error."""
        fetcher = DataFetcher()
        
        with pytest.raises(DataFetcherError, match="Invalid period"):
            fetcher._validate_params("invalid", "1d")
    
    def test_invalid_interval(self):
        """Test that invalid interval raises error."""
        fetcher = DataFetcher()
        
        with pytest.raises(DataFetcherError, match="Invalid interval"):
            fetcher._validate_params("1y", "invalid")
    
    def test_valid_params(self):
        """Test valid parameters don't raise errors."""
        fetcher = DataFetcher()
        
        # Should not raise
        fetcher._validate_params("1y", "1d")
        fetcher._validate_params("6mo", "1h")
        fetcher._validate_params("max", "1wk")


class TestDataCleaning:
    """Tests for data cleaning functionality."""
    
    def test_clean_data_with_multiindex(self):
        """Test cleaning data with multi-level columns."""
        fetcher = DataFetcher()
        
        # Create data with multi-level columns
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            ("Open", "NVDA"): [100, 101, 102, 103, 104],
            ("High", "NVDA"): [101, 102, 103, 104, 105],
            ("Low", "NVDA"): [99, 100, 101, 102, 103],
            ("Close", "NVDA"): [100.5, 101.5, 102.5, 103.5, 104.5],
            ("Volume", "NVDA"): [1000, 1100, 1200, 1300, 1400],
        }, index=dates)
        
        # Should flatten columns
        cleaned = fetcher._clean_data(data)
        
        # Check columns are flattened
        assert "Close" in cleaned.columns
        assert "Volume" in cleaned.columns
    
    def test_clean_data_removes_nan(self):
        """Test that NaN values in critical columns are removed."""
        fetcher = DataFetcher()
        
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "Open": [100, None, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100.5, 101.5, None, 103.5, 104.5],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        }, index=dates)
        
        cleaned = fetcher._clean_data(data)
        
        # Should have removed rows with NaN in Open/Close
        assert len(cleaned) == 3


class TestCaching:
    """Tests for caching functionality."""
    
    def test_cache_path_generation(self, tmp_path):
        """Test cache path is generated correctly."""
        fetcher = DataFetcher(cache_dir=str(tmp_path))
        
        path = fetcher._get_cache_path("NVDA", "1y", "1d")
        
        assert path.name == "NVDA_1y_1d.parquet"
        assert path.parent == tmp_path
    
    def test_cache_roundtrip(self, tmp_path):
        """Test saving and loading from cache."""
        fetcher = DataFetcher(cache_dir=str(tmp_path))
        
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        }, index=dates)
        
        # Save to cache
        fetcher._save_to_cache(data, "TEST", "1y", "1d")
        
        # Load from cache
        loaded = fetcher._load_from_cache("TEST", "1y", "1d")
        
        assert loaded is not None
        assert len(loaded) == 5
        pd.testing.assert_frame_equal(data, loaded)
    
    def test_clear_cache(self, tmp_path):
        """Test clearing cache."""
        fetcher = DataFetcher(cache_dir=str(tmp_path))
        
        # Create dummy cache files
        (tmp_path / "NVDA_1y_1d.parquet").touch()
        (tmp_path / "AAPL_6mo_1d.parquet").touch()
        
        # Clear specific symbol
        count = fetcher.clear_cache("NVDA")
        assert count == 1
        
        # Clear all remaining
        count = fetcher.clear_cache()
        assert count == 1


class TestFetchWithMock:
    """Tests for fetch functionality with mocked API."""
    
    @patch("src.data_fetcher.yf.download")
    def test_fetch_success(self, mock_download):
        """Test successful fetch."""
        # Setup mock
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        mock_data = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        }, index=dates)
        mock_download.return_value = mock_data
        
        fetcher = DataFetcher()
        fetcher.config.cache_enabled = False  # Disable cache for test
        
        result = fetcher.fetch("NVDA", period="1mo", interval="1d")
        
        assert not result.empty
        assert "Close" in result.columns
        mock_download.assert_called_once()
    
    @patch("src.data_fetcher.yf.download")
    def test_fetch_empty_data(self, mock_download):
        """Test handling of empty data response."""
        mock_download.return_value = pd.DataFrame()
        
        fetcher = DataFetcher()
        fetcher.config.cache_enabled = False
        
        with pytest.raises(DataFetcherError, match="No data returned"):
            fetcher.fetch("INVALID", period="1mo", interval="1d")


class TestStockInfo:
    """Tests for stock info retrieval."""
    
    @patch("src.data_fetcher.yf.Ticker")
    def test_get_stock_info(self, mock_ticker):
        """Test getting stock information."""
        mock_ticker.return_value.info = {
            "longName": "NVIDIA Corporation",
            "sector": "Technology",
            "industry": "Semiconductors",
            "marketCap": 1000000000000,
            "currency": "USD",
        }
        
        fetcher = DataFetcher()
        info = fetcher.get_stock_info("NVDA")
        
        assert info["symbol"] == "NVDA"
        assert info["name"] == "NVIDIA Corporation"
        assert info["sector"] == "Technology"

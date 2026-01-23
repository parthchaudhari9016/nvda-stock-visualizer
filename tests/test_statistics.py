"""
Tests for Statistical Analysis module.
"""

import pandas as pd
import numpy as np
import pytest

from src.statistical_analysis import StatisticalAnalysis, PerformanceMetrics, RiskMetrics


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    np.random.seed(42)
    
    # Simulate price movement
    returns = np.random.randn(252) * 0.02  # 2% daily std
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        "Open": prices * (1 + np.random.randn(252) * 0.005),
        "High": prices * (1 + np.abs(np.random.randn(252) * 0.01)),
        "Low": prices * (1 - np.abs(np.random.randn(252) * 0.01)),
        "Close": prices,
        "Volume": np.random.randint(1000000, 5000000, 252),
    }, index=dates)
    
    return data


class TestDescriptiveStats:
    """Tests for descriptive statistics."""
    
    def test_get_descriptive_stats_keys(self, sample_data):
        """Test that all expected keys are present."""
        analyzer = StatisticalAnalysis(sample_data)
        stats = analyzer.get_descriptive_stats()
        
        expected_keys = [
            "count", "start_date", "end_date",
            "start_price", "end_price",
            "min_price", "max_price", "mean_price",
            "daily_return_mean", "daily_return_std",
            "skewness", "kurtosis",
            "positive_days", "negative_days"
        ]
        
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"
    
    def test_price_summary(self, sample_data):
        """Test price summary values."""
        analyzer = StatisticalAnalysis(sample_data)
        summary = analyzer.get_price_summary()
        
        assert "current" in summary
        assert "high_52w" in summary
        assert "low_52w" in summary
        assert summary["high_52w"] >= summary["low_52w"]


class TestPerformanceMetrics:
    """Tests for performance metrics."""
    
    def test_performance_metrics_types(self, sample_data):
        """Test that metrics are of correct types."""
        analyzer = StatisticalAnalysis(sample_data)
        perf = analyzer.calculate_performance_metrics()
        
        assert isinstance(perf, PerformanceMetrics)
        assert isinstance(perf.total_return, float)
        assert isinstance(perf.sharpe_ratio, float)
        assert isinstance(perf.max_drawdown, float)
    
    def test_max_drawdown_negative(self, sample_data):
        """Test that max drawdown is negative or zero."""
        analyzer = StatisticalAnalysis(sample_data)
        perf = analyzer.calculate_performance_metrics()
        
        assert perf.max_drawdown <= 0
    
    def test_win_rate_bounds(self, sample_data):
        """Test that win rate is between 0 and 100."""
        analyzer = StatisticalAnalysis(sample_data)
        perf = analyzer.calculate_performance_metrics()
        
        assert 0 <= perf.win_rate <= 100


class TestRiskMetrics:
    """Tests for risk metrics."""
    
    def test_risk_metrics_types(self, sample_data):
        """Test that risk metrics are of correct types."""
        analyzer = StatisticalAnalysis(sample_data)
        risk = analyzer.calculate_risk_metrics()
        
        assert isinstance(risk, RiskMetrics)
        assert isinstance(risk.var_95, float)
        assert isinstance(risk.var_99, float)
    
    def test_var_ordering(self, sample_data):
        """Test VaR 99% is more extreme than VaR 95%."""
        analyzer = StatisticalAnalysis(sample_data)
        risk = analyzer.calculate_risk_metrics()
        
        # VaR 99% should be more negative than VaR 95%
        assert risk.var_99 <= risk.var_95
    
    def test_cvar_vs_var(self, sample_data):
        """Test CVaR is more extreme than VaR."""
        analyzer = StatisticalAnalysis(sample_data)
        risk = analyzer.calculate_risk_metrics()
        
        # CVaR (expected shortfall) should be more extreme
        assert risk.cvar_95 <= risk.var_95


class TestReturnsAnalysis:
    """Tests for returns calculations."""
    
    def test_daily_returns_length(self, sample_data):
        """Test daily returns length."""
        analyzer = StatisticalAnalysis(sample_data)
        daily = analyzer.calculate_returns("daily")
        
        # Should be one less than prices (first return is NaN dropped)
        assert len(daily) == len(sample_data) - 1
    
    def test_cumulative_returns(self, sample_data):
        """Test cumulative returns calculation."""
        analyzer = StatisticalAnalysis(sample_data)
        cumulative = analyzer.calculate_cumulative_returns()
        
        # First cumulative return should match first daily return
        daily = analyzer.daily_returns
        assert abs(cumulative.iloc[0] - daily.iloc[0]) < 0.0001
    
    def test_weekly_returns(self, sample_data):
        """Test weekly returns calculation."""
        analyzer = StatisticalAnalysis(sample_data)
        weekly = analyzer.calculate_returns("weekly")
        
        # Weekly returns should have fewer data points
        assert len(weekly) < len(sample_data)


class TestDrawdownAnalysis:
    """Tests for drawdown analysis."""
    
    def test_max_drawdown_range(self, sample_data):
        """Test max drawdown is in valid range."""
        analyzer = StatisticalAnalysis(sample_data)
        max_dd = analyzer.calculate_max_drawdown()
        
        assert -1.0 <= max_dd <= 0.0
    
    def test_drawdown_series(self, sample_data):
        """Test drawdown series properties."""
        analyzer = StatisticalAnalysis(sample_data)
        dd_series = analyzer.calculate_drawdown_series()
        
        # All drawdowns should be <= 0
        assert (dd_series <= 0).all()
        
        # Should have same length as returns
        assert len(dd_series) == len(analyzer.daily_returns)


class TestGenerateSummary:
    """Tests for summary generation."""
    
    def test_summary_structure(self, sample_data):
        """Test summary has all required sections."""
        analyzer = StatisticalAnalysis(sample_data)
        summary = analyzer.generate_summary()
        
        expected_sections = [
            "descriptive_stats",
            "price_summary",
            "performance",
            "risk",
            "return_distribution"
        ]
        
        for section in expected_sections:
            assert section in summary, f"Missing section: {section}"

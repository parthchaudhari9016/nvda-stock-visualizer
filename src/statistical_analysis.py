"""
Statistical Analysis Module - Quantitative risk and return metrics.

Provides comprehensive statistical analysis tools for evaluating
stock performance, risk metrics, and return characteristics.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import get_config


@dataclass
class PerformanceMetrics:
    """Container for performance analysis results."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float


@dataclass
class RiskMetrics:
    """Container for risk analysis results."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    beta: Optional[float]
    alpha: Optional[float]
    r_squared: Optional[float]


class StatisticalAnalysis:
    """
    Comprehensive statistical and risk analysis for stock data.
    
    Provides methods for:
    - Descriptive statistics
    - Return analysis (daily, weekly, monthly)
    - Risk metrics (VaR, CVaR, Sharpe, Sortino)
    - Drawdown analysis
    - Correlation analysis
    
    Example:
        >>> analyzer = StatisticalAnalysis(data)
        >>> stats = analyzer.get_descriptive_stats()
        >>> perf = analyzer.calculate_performance_metrics()
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        price_column: str = "Close",
        risk_free_rate: Optional[float] = None,
        trading_days: int = 252,
    ):
        """
        Initialize the statistical analyzer.
        
        Args:
            data: DataFrame with OHLCV data
            price_column: Column to use for analysis
            risk_free_rate: Annual risk-free rate (defaults to config)
            trading_days: Trading days per year
        """
        self.data = data.copy()
        self.price_column = price_column
        self.prices = data[price_column]
        
        config = get_config()
        self.risk_free_rate = risk_free_rate or config.analysis.risk_free_rate
        self.trading_days = trading_days
        
        # Pre-calculate returns
        self.daily_returns = self.prices.pct_change().dropna()
        self.log_returns = np.log(self.prices / self.prices.shift(1)).dropna()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DESCRIPTIVE STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_descriptive_stats(self) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics.
        
        Returns:
            Dictionary with statistical measures
        """
        returns = self.daily_returns
        
        return {
            "count": len(self.prices),
            "start_date": str(self.prices.index[0].date()),
            "end_date": str(self.prices.index[-1].date()),
            "start_price": round(self.prices.iloc[0], 2),
            "end_price": round(self.prices.iloc[-1], 2),
            "min_price": round(self.prices.min(), 2),
            "max_price": round(self.prices.max(), 2),
            "mean_price": round(self.prices.mean(), 2),
            "median_price": round(self.prices.median(), 2),
            "std_price": round(self.prices.std(), 2),
            "daily_return_mean": round(returns.mean() * 100, 4),
            "daily_return_std": round(returns.std() * 100, 4),
            "skewness": round(returns.skew(), 4),
            "kurtosis": round(returns.kurtosis(), 4),
            "positive_days": (returns > 0).sum(),
            "negative_days": (returns < 0).sum(),
            "positive_ratio": round((returns > 0).mean() * 100, 2),
        }
    
    def get_price_summary(self) -> Dict[str, float]:
        """Get price range summary."""
        return {
            "current": round(self.prices.iloc[-1], 2),
            "high_52w": round(self.prices.tail(self.trading_days).max(), 2),
            "low_52w": round(self.prices.tail(self.trading_days).min(), 2),
            "avg_30d": round(self.prices.tail(30).mean(), 2),
            "avg_90d": round(self.prices.tail(90).mean(), 2),
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RETURN ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_returns(
        self,
        period: str = "daily",
    ) -> pd.Series:
        """
        Calculate returns for different periods.
        
        Args:
            period: "daily", "weekly", "monthly", or "yearly"
            
        Returns:
            Series of returns
        """
        if period == "daily":
            return self.daily_returns
        elif period == "weekly":
            weekly_prices = self.prices.resample("W").last()
            return weekly_prices.pct_change().dropna()
        elif period == "monthly":
            monthly_prices = self.prices.resample("ME").last()
            return monthly_prices.pct_change().dropna()
        elif period == "yearly":
            yearly_prices = self.prices.resample("YE").last()
            return yearly_prices.pct_change().dropna()
        else:
            raise ValueError(f"Invalid period: {period}")
    
    def calculate_cumulative_returns(self) -> pd.Series:
        """Calculate cumulative returns from start."""
        return (1 + self.daily_returns).cumprod() - 1
    
    def calculate_rolling_returns(
        self,
        window: int = 30,
    ) -> pd.Series:
        """
        Calculate rolling returns.
        
        Args:
            window: Rolling window in days
            
        Returns:
            Series of rolling returns
        """
        return self.prices.pct_change(periods=window)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERFORMANCE METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            PerformanceMetrics dataclass with all metrics
        """
        returns = self.daily_returns
        
        # Total and annualized returns
        total_return = (self.prices.iloc[-1] / self.prices.iloc[0]) - 1
        n_years = len(returns) / self.trading_days
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(self.trading_days)
        
        # Sharpe Ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino Ratio (uses downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.trading_days)
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # Maximum Drawdown
        max_drawdown = self.calculate_max_drawdown()
        
        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate and profit factor
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        gross_profits = wins.sum()
        gross_losses = abs(losses.sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        return PerformanceMetrics(
            total_return=round(total_return * 100, 2),
            annualized_return=round(annualized_return * 100, 2),
            volatility=round(volatility * 100, 2),
            sharpe_ratio=round(sharpe_ratio, 3),
            sortino_ratio=round(sortino_ratio, 3),
            max_drawdown=round(max_drawdown * 100, 2),
            calmar_ratio=round(calmar_ratio, 3),
            win_rate=round(win_rate * 100, 2),
            avg_win=round(avg_win * 100, 4),
            avg_loss=round(avg_loss * 100, 4),
            profit_factor=round(profit_factor, 3),
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RISK METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_risk_metrics(
        self,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            benchmark_returns: Optional benchmark for beta/alpha calculation
            
        Returns:
            RiskMetrics dataclass with all metrics
        """
        returns = self.daily_returns
        
        # Value at Risk (VaR) - Historical method
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional VaR (CVaR / Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Beta, Alpha, R-squared (if benchmark provided)
        beta = None
        alpha = None
        r_squared = None
        
        if benchmark_returns is not None:
            # Align returns
            aligned = pd.DataFrame({
                "stock": returns,
                "benchmark": benchmark_returns
            }).dropna()
            
            if len(aligned) > 10:
                cov_matrix = aligned.cov()
                beta = cov_matrix.loc["stock", "benchmark"] / aligned["benchmark"].var()
                
                stock_ann = aligned["stock"].mean() * self.trading_days
                bench_ann = aligned["benchmark"].mean() * self.trading_days
                alpha = stock_ann - (self.risk_free_rate + beta * (bench_ann - self.risk_free_rate))
                
                corr = aligned.corr().loc["stock", "benchmark"]
                r_squared = corr ** 2
        
        return RiskMetrics(
            var_95=round(var_95 * 100, 4),
            var_99=round(var_99 * 100, 4),
            cvar_95=round(cvar_95 * 100, 4) if not np.isnan(cvar_95) else 0,
            cvar_99=round(cvar_99 * 100, 4) if not np.isnan(cvar_99) else 0,
            beta=round(beta, 4) if beta is not None else None,
            alpha=round(alpha * 100, 4) if alpha is not None else None,
            r_squared=round(r_squared, 4) if r_squared is not None else None,
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DRAWDOWN ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        
        Returns:
            Maximum drawdown as decimal (negative)
        """
        cumulative = (1 + self.daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_drawdown_series(self) -> pd.Series:
        """
        Calculate drawdown series over time.
        
        Returns:
            Series of drawdowns
        """
        cumulative = (1 + self.daily_returns).cumprod()
        running_max = cumulative.cummax()
        return (cumulative - running_max) / running_max
    
    def get_drawdown_periods(
        self,
        threshold: float = -0.05,
    ) -> pd.DataFrame:
        """
        Get significant drawdown periods.
        
        Args:
            threshold: Minimum drawdown to include (negative)
            
        Returns:
            DataFrame with drawdown periods
        """
        drawdown = self.calculate_drawdown_series()
        
        # Find drawdown periods
        in_drawdown = drawdown < threshold
        
        periods = []
        start = None
        
        for i, (date, is_dd) in enumerate(in_drawdown.items()):
            if is_dd and start is None:
                start = date
            elif not is_dd and start is not None:
                end = drawdown.index[i - 1]
                min_dd = drawdown[start:end].min()
                periods.append({
                    "start": start,
                    "end": end,
                    "duration_days": (end - start).days,
                    "max_drawdown": round(min_dd * 100, 2),
                })
                start = None
        
        return pd.DataFrame(periods)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CORRELATION ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_rolling_correlation(
        self,
        other_returns: pd.Series,
        window: int = 30,
    ) -> pd.Series:
        """
        Calculate rolling correlation with another series.
        
        Args:
            other_returns: Return series to correlate with
            window: Rolling window
            
        Returns:
            Rolling correlation series
        """
        return self.daily_returns.rolling(window).corr(other_returns)
    
    def calculate_return_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of returns.
        
        Returns:
            Dictionary with distribution metrics
        """
        returns = self.daily_returns
        
        return {
            "mean": round(returns.mean() * 100, 4),
            "std": round(returns.std() * 100, 4),
            "skew": round(returns.skew(), 4),
            "kurtosis": round(returns.kurtosis(), 4),
            "min": round(returns.min() * 100, 4),
            "max": round(returns.max() * 100, 4),
            "percentile_5": round(np.percentile(returns, 5) * 100, 4),
            "percentile_25": round(np.percentile(returns, 25) * 100, 4),
            "percentile_50": round(np.percentile(returns, 50) * 100, 4),
            "percentile_75": round(np.percentile(returns, 75) * 100, 4),
            "percentile_95": round(np.percentile(returns, 95) * 100, 4),
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis summary.
        
        Returns:
            Dictionary with complete analysis
        """
        return {
            "descriptive_stats": self.get_descriptive_stats(),
            "price_summary": self.get_price_summary(),
            "performance": self.calculate_performance_metrics().__dict__,
            "risk": self.calculate_risk_metrics().__dict__,
            "return_distribution": self.calculate_return_distribution(),
        }

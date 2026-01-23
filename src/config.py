"""
Configuration management for NVDA Stock Analyzer.
Centralized settings, themes, and default parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Theme(Enum):
    """Chart theme options."""
    DARK = "dark"
    LIGHT = "light"
    PROFESSIONAL = "professional"


class OutputFormat(Enum):
    """Report output format options."""
    PDF = "pdf"
    EXCEL = "excel"
    HTML = "html"
    MARKDOWN = "markdown"
    PNG = "png"


@dataclass
class ChartConfig:
    """Chart visualization configuration."""
    theme: Theme = Theme.PROFESSIONAL
    figsize: tuple = (14, 10)
    dpi: int = 150
    grid_alpha: float = 0.3
    title_fontsize: int = 14
    label_fontsize: int = 11
    legend_fontsize: int = 10


@dataclass
class ColorPalette:
    """Professional color schemes for charts."""
    # Price colors
    price_up: str = "#26a69a"       # Green for bullish
    price_down: str = "#ef5350"     # Red for bearish
    price_line: str = "#2196f3"     # Blue for close price
    
    # Indicator colors
    sma_short: str = "#ff9800"      # Orange
    sma_long: str = "#9c27b0"       # Purple  
    ema: str = "#00bcd4"            # Cyan
    macd_line: str = "#2196f3"      # Blue
    macd_signal: str = "#ff5722"    # Deep orange
    macd_histogram_pos: str = "#4caf50"
    macd_histogram_neg: str = "#f44336"
    
    # Bollinger Bands
    bb_upper: str = "#90a4ae"
    bb_lower: str = "#90a4ae"
    bb_fill: str = "#e3f2fd"
    
    # RSI colors
    rsi_line: str = "#673ab7"
    rsi_overbought: str = "#ef5350"
    rsi_oversold: str = "#26a69a"
    
    # Volume
    volume_up: str = "#26a69a80"    # Semi-transparent green
    volume_down: str = "#ef535080"  # Semi-transparent red
    
    # Background
    background: str = "#1e1e2e"
    grid: str = "#333344"
    text: str = "#cdd6f4"


@dataclass
class AnalysisConfig:
    """Analysis parameters configuration."""
    # Default symbol and periods
    default_symbol: str = "NVDA"
    default_period: str = "1y"
    default_interval: str = "1d"
    
    # Moving averages
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    
    # RSI
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # ATR
    atr_period: int = 14
    
    # Risk metrics
    risk_free_rate: float = 0.05  # 5% annual
    var_confidence: float = 0.95
    trading_days: int = 252


@dataclass
class Config:
    """Main configuration container."""
    chart: ChartConfig = field(default_factory=ChartConfig)
    colors: ColorPalette = field(default_factory=ColorPalette)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output_dir: str = "output"
    cache_enabled: bool = True
    cache_expiry_hours: int = 1


# Global default configuration
DEFAULT_CONFIG = Config()


def get_config() -> Config:
    """Get the default configuration instance."""
    return DEFAULT_CONFIG

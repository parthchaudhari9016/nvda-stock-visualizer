"""
Stock Visualizer Module - Professional-grade chart generation.

Creates publication-ready visualizations including candlestick charts,
technical indicator overlays, and multi-panel dashboards.
"""

from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from .config import get_config, Config, ColorPalette
from .technical_indicators import TechnicalIndicators


class StockVisualizer:
    """
    Professional stock chart visualization engine.
    
    Creates high-quality, publication-ready charts with:
    - Candlestick/OHLC charts
    - Technical indicator overlays
    - Multi-panel layouts
    - Customizable themes
    
    Example:
        >>> viz = StockVisualizer(data, symbol="NVDA")
        >>> viz.plot_technical_analysis()
        >>> viz.save("nvda_analysis.png")
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        symbol: str = "NVDA",
        config: Optional[Config] = None,
    ):
        """
        Initialize the visualizer.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock ticker symbol
            config: Configuration object
        """
        self.data = data.copy()
        self.symbol = symbol
        self.config = config or get_config()
        self.colors = self.config.colors
        
        # Apply theme
        self._apply_theme()
        
        # Current figure reference
        self.fig = None
        self.axes = None
    
    def _apply_theme(self) -> None:
        """Apply the configured theme to matplotlib."""
        plt.style.use("dark_background")
        
        plt.rcParams.update({
            "figure.facecolor": self.colors.background,
            "axes.facecolor": self.colors.background,
            "axes.edgecolor": self.colors.grid,
            "axes.labelcolor": self.colors.text,
            "text.color": self.colors.text,
            "xtick.color": self.colors.text,
            "ytick.color": self.colors.text,
            "grid.color": self.colors.grid,
            "grid.alpha": 0.3,
            "legend.facecolor": self.colors.background,
            "legend.edgecolor": self.colors.grid,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
        })
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BASIC CHARTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def plot_price(
        self,
        show_volume: bool = True,
        show_sma: List[int] = [20, 50],
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Create a basic price chart with optional indicators.
        
        Args:
            show_volume: Whether to show volume subplot
            show_sma: List of SMA periods to display
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure object
        """
        figsize = figsize or self.config.chart.figsize
        
        if show_volume:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=figsize,
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True
            )
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
            ax2 = None
        
        # Price line
        ax1.plot(
            self.data.index,
            self.data["Close"],
            color=self.colors.price_line,
            linewidth=1.5,
            label="Close"
        )
        
        # Add SMAs
        sma_colors = [self.colors.sma_short, self.colors.sma_long, "#9c27b0", "#00bcd4"]
        for i, period in enumerate(show_sma):
            sma = TechnicalIndicators.sma(self.data["Close"], period)
            color = sma_colors[i % len(sma_colors)]
            ax1.plot(
                self.data.index,
                sma,
                color=color,
                linewidth=1,
                alpha=0.8,
                label=f"SMA {period}"
            )
        
        ax1.set_title(f"{self.symbol} Price Analysis", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Price (USD)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")
        
        # Volume subplot
        if ax2 is not None:
            colors = [
                self.colors.volume_up if c >= o else self.colors.volume_down
                for o, c in zip(self.data["Open"], self.data["Close"])
            ]
            ax2.bar(self.data.index, self.data["Volume"], color=colors, width=0.8)
            ax2.set_ylabel("Volume")
            ax2.set_xlabel("Date")
            ax2.grid(True, alpha=0.3)
        else:
            ax1.set_xlabel("Date")
        
        # Format x-axis
        if len(self.data) > 100:
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        
        plt.tight_layout()
        self.fig = fig
        return fig
    
    def plot_candlestick(
        self,
        show_volume: bool = True,
        show_sma: List[int] = [20, 50],
        tail_days: Optional[int] = 90,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Create a candlestick chart.
        
        Args:
            show_volume: Whether to show volume subplot
            show_sma: List of SMA periods to display
            tail_days: Number of recent days to show (None for all)
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        figsize = figsize or self.config.chart.figsize
        data = self.data.tail(tail_days) if tail_days else self.data
        
        if show_volume:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=figsize,
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True
            )
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
            ax2 = None
        
        # Draw candlesticks
        width = 0.6
        width2 = 0.1
        
        for i, (idx, row) in enumerate(data.iterrows()):
            is_up = row["Close"] >= row["Open"]
            color = self.colors.price_up if is_up else self.colors.price_down
            
            # Body
            body_bottom = min(row["Open"], row["Close"])
            body_height = abs(row["Close"] - row["Open"])
            ax1.add_patch(Rectangle(
                (mdates.date2num(idx) - width/2, body_bottom),
                width, body_height,
                facecolor=color,
                edgecolor=color
            ))
            
            # Wicks
            ax1.plot(
                [mdates.date2num(idx), mdates.date2num(idx)],
                [row["Low"], body_bottom],
                color=color,
                linewidth=1
            )
            ax1.plot(
                [mdates.date2num(idx), mdates.date2num(idx)],
                [body_bottom + body_height, row["High"]],
                color=color,
                linewidth=1
            )
        
        # Add SMAs
        sma_colors = [self.colors.sma_short, self.colors.sma_long]
        for i, period in enumerate(show_sma[:2]):
            sma = TechnicalIndicators.sma(data["Close"], period)
            ax1.plot(
                data.index,
                sma,
                color=sma_colors[i],
                linewidth=1.5,
                label=f"SMA {period}"
            )
        
        ax1.set_title(f"{self.symbol} Candlestick Chart", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Price (USD)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")
        ax1.set_xlim(mdates.date2num(data.index[0]) - 1, mdates.date2num(data.index[-1]) + 1)
        
        # Volume
        if ax2 is not None:
            colors = [
                self.colors.price_up if c >= o else self.colors.price_down
                for o, c in zip(data["Open"], data["Close"])
            ]
            ax2.bar(data.index, data["Volume"], color=colors, width=0.8, alpha=0.7)
            ax2.set_ylabel("Volume")
            ax2.set_xlabel("Date")
            ax2.grid(True, alpha=0.3)
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        self.fig = fig
        return fig
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TECHNICAL ANALYSIS CHARTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def plot_technical_analysis(
        self,
        indicators: List[str] = ["macd", "rsi", "volume"],
        tail_days: Optional[int] = 180,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Create a comprehensive technical analysis chart.
        
        Args:
            indicators: List of indicators to show ["macd", "rsi", "volume", "bollinger"]
            tail_days: Number of recent days to show
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        figsize = figsize or (14, 12)
        data = self.data.tail(tail_days) if tail_days else self.data
        
        n_panels = 1 + len(indicators)
        height_ratios = [3] + [1] * len(indicators)
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_panels, 1, height_ratios=height_ratios, hspace=0.05)
        
        axes = []
        for i in range(n_panels):
            ax = fig.add_subplot(gs[i])
            axes.append(ax)
            if i < n_panels - 1:
                ax.tick_params(labelbottom=False)
        
        # Main price panel with Bollinger Bands
        ax_price = axes[0]
        
        if "bollinger" in indicators:
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(data["Close"])
            ax_price.fill_between(
                data.index, bb_upper, bb_lower,
                color=self.colors.bb_fill, alpha=0.3
            )
            ax_price.plot(data.index, bb_upper, color=self.colors.bb_upper, linewidth=0.8, alpha=0.7)
            ax_price.plot(data.index, bb_lower, color=self.colors.bb_lower, linewidth=0.8, alpha=0.7)
        
        # Price and SMAs
        ax_price.plot(
            data.index, data["Close"],
            color=self.colors.price_line, linewidth=1.5, label="Close"
        )
        
        sma_20 = TechnicalIndicators.sma(data["Close"], 20)
        sma_50 = TechnicalIndicators.sma(data["Close"], 50)
        ax_price.plot(data.index, sma_20, color=self.colors.sma_short, linewidth=1, label="SMA 20")
        ax_price.plot(data.index, sma_50, color=self.colors.sma_long, linewidth=1, label="SMA 50")
        
        ax_price.set_title(
            f"{self.symbol} Technical Analysis",
            fontsize=14, fontweight="bold", pad=10
        )
        ax_price.set_ylabel("Price (USD)")
        ax_price.legend(loc="upper left", fontsize=9)
        ax_price.grid(True, alpha=0.3)
        
        # Additional indicator panels
        panel_idx = 1
        for indicator in indicators:
            if indicator == "bollinger":
                continue  # Already shown on price panel
            
            ax = axes[panel_idx]
            
            if indicator == "macd":
                self._plot_macd(ax, data)
            elif indicator == "rsi":
                self._plot_rsi(ax, data)
            elif indicator == "volume":
                self._plot_volume(ax, data)
            
            panel_idx += 1
        
        # Format dates on last axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axes[-1].set_xlabel("Date")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        self.fig = fig
        self.axes = axes
        return fig
    
    def _plot_macd(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        """Plot MACD indicator."""
        macd_line, signal_line, histogram = TechnicalIndicators.macd(data["Close"])
        
        # Histogram
        colors = [
            self.colors.macd_histogram_pos if h >= 0 else self.colors.macd_histogram_neg
            for h in histogram
        ]
        ax.bar(data.index, histogram, color=colors, width=0.8, alpha=0.7)
        
        # Lines
        ax.plot(data.index, macd_line, color=self.colors.macd_line, linewidth=1, label="MACD")
        ax.plot(data.index, signal_line, color=self.colors.macd_signal, linewidth=1, label="Signal")
        
        ax.axhline(y=0, color=self.colors.text, linewidth=0.5, alpha=0.5)
        ax.set_ylabel("MACD")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_rsi(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        """Plot RSI indicator."""
        rsi = TechnicalIndicators.rsi(data["Close"])
        
        ax.plot(data.index, rsi, color=self.colors.rsi_line, linewidth=1)
        ax.axhline(y=70, color=self.colors.rsi_overbought, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(y=30, color=self.colors.rsi_oversold, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(y=50, color=self.colors.text, linewidth=0.5, alpha=0.3)
        
        ax.fill_between(data.index, 70, 100, color=self.colors.rsi_overbought, alpha=0.1)
        ax.fill_between(data.index, 0, 30, color=self.colors.rsi_oversold, alpha=0.1)
        
        ax.set_ylim(0, 100)
        ax.set_ylabel("RSI")
        ax.grid(True, alpha=0.3)
    
    def _plot_volume(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        """Plot volume bars."""
        colors = [
            self.colors.price_up if c >= o else self.colors.price_down
            for o, c in zip(data["Open"], data["Close"])
        ]
        ax.bar(data.index, data["Volume"], color=colors, width=0.8, alpha=0.7)
        ax.set_ylabel("Volume")
        ax.grid(True, alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMPARISON CHARTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def plot_performance_comparison(
        self,
        other_data: Dict[str, pd.DataFrame],
        normalize: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Compare performance across multiple stocks.
        
        Args:
            other_data: Dictionary of {symbol: DataFrame}
            normalize: Normalize to 100 at start
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        figsize = figsize or self.config.chart.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = [
            self.colors.price_line, self.colors.sma_short, self.colors.sma_long,
            "#9c27b0", "#00bcd4", "#e91e63", "#cddc39"
        ]
        
        all_data = {self.symbol: self.data, **other_data}
        
        for i, (symbol, data) in enumerate(all_data.items()):
            if data is None:
                continue
            
            prices = data["Close"]
            if normalize:
                prices = (prices / prices.iloc[0]) * 100
            
            ax.plot(
                data.index, prices,
                color=colors[i % len(colors)],
                linewidth=1.5,
                label=symbol
            )
        
        ax.set_title("Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_ylabel("Normalized Price" if normalize else "Price (USD)")
        ax.set_xlabel("Date")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig = fig
        return fig
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def save(
        self,
        filepath: str,
        dpi: Optional[int] = None,
        transparent: bool = False,
    ) -> Path:
        """
        Save the current figure to file.
        
        Args:
            filepath: Output file path
            dpi: Resolution (defaults to config)
            transparent: Whether to use transparent background
            
        Returns:
            Path to saved file
        """
        if self.fig is None:
            raise ValueError("No figure to save. Create a plot first.")
        
        dpi = dpi or self.config.chart.dpi
        path = Path(filepath)
        
        self.fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=self.fig.get_facecolor(),
            transparent=transparent
        )
        
        return path
    
    def show(self) -> None:
        """Display the current figure."""
        if self.fig is None:
            raise ValueError("No figure to show. Create a plot first.")
        plt.show()
    
    def close(self) -> None:
        """Close the current figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None

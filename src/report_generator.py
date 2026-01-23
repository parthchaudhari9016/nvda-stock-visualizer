"""
Report Generator Module - Multi-format report exports.

Generates professional analysis reports in various formats
including Markdown, HTML, and text summaries.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import json

import pandas as pd

from .config import get_config
from .data_fetcher import DataFetcher
from .technical_indicators import TechnicalIndicators
from .statistical_analysis import StatisticalAnalysis
from .visualizer import StockVisualizer


class ReportGenerator:
    """
    Professional analysis report generator.
    
    Generates comprehensive reports in multiple formats:
    - Markdown (for documentation)
    - HTML (for web viewing)
    - Text (for terminal/email)
    - JSON (for programmatic access)
    
    Example:
        >>> generator = ReportGenerator(data, symbol="NVDA")
        >>> generator.generate_markdown_report("report.md")
        >>> generator.generate_html_report("report.html")
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        symbol: str = "NVDA",
        period: str = "1y",
    ):
        """
        Initialize the report generator.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock ticker symbol
            period: Data period for context
        """
        self.data = data.copy()
        self.symbol = symbol
        self.period = period
        self.config = get_config()
        
        # Pre-calculate analysis
        self.analyzer = StatisticalAnalysis(data)
        self.stats = self.analyzer.get_descriptive_stats()
        self.performance = self.analyzer.calculate_performance_metrics()
        self.risk = self.analyzer.calculate_risk_metrics()
        
        # Add technical indicators
        self.data_with_indicators = TechnicalIndicators.add_all_indicators(data)
    
    def generate_markdown_report(
        self,
        filepath: str,
        include_charts: bool = True,
        chart_dir: Optional[str] = None,
    ) -> Path:
        """
        Generate a Markdown format report.
        
        Args:
            filepath: Output file path
            include_charts: Whether to generate and embed charts
            chart_dir: Directory to save charts
            
        Returns:
            Path to generated report
        """
        path = Path(filepath)
        chart_dir = Path(chart_dir) if chart_dir else path.parent
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate charts if requested
        chart_filename = None
        if include_charts:
            viz = StockVisualizer(self.data, symbol=self.symbol)
            viz.plot_technical_analysis()
            chart_filename = f"{self.symbol.lower()}_analysis.png"
            viz.save(str(chart_dir / chart_filename))
            viz.close()
        
        # Build report
        report = self._build_markdown_content(chart_filename)
        
        path.write_text(report, encoding="utf-8")
        return path
    
    def _build_markdown_content(self, chart_filename: Optional[str] = None) -> str:
        """Build the Markdown report content."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            f"# {self.symbol} Stock Analysis Report",
            "",
            f"**Generated:** {now}  ",
            f"**Period:** {self.period}  ",
            f"**Data Points:** {self.stats['count']}  ",
            "",
            "---",
            "",
        ]
        
        # Executive Summary
        lines.extend([
            "## 📋 Executive Summary",
            "",
            f"This report provides a comprehensive analysis of **{self.symbol}** stock ",
            f"covering the period from {self.stats['start_date']} to {self.stats['end_date']}.",
            "",
            "### Key Highlights",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Return | **{self.performance.total_return}%** |",
            f"| Annualized Return | {self.performance.annualized_return}% |",
            f"| Volatility | {self.performance.volatility}% |",
            f"| Sharpe Ratio | {self.performance.sharpe_ratio} |",
            f"| Max Drawdown | {self.performance.max_drawdown}% |",
            "",
        ])
        
        # Chart
        if chart_filename:
            lines.extend([
                "---",
                "",
                "## 📊 Technical Analysis Chart",
                "",
                f"![{self.symbol} Technical Analysis]({chart_filename})",
                "",
            ])
        
        # Price Analysis
        lines.extend([
            "---",
            "",
            "## 💰 Price Analysis",
            "",
            "### Price Range",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Start Price | ${self.stats['start_price']} |",
            f"| End Price | ${self.stats['end_price']} |",
            f"| High | ${self.stats['max_price']} |",
            f"| Low | ${self.stats['min_price']} |",
            f"| Mean | ${self.stats['mean_price']} |",
            f"| Median | ${self.stats['median_price']} |",
            "",
        ])
        
        # Performance Metrics
        lines.extend([
            "---",
            "",
            "## 📈 Performance Metrics",
            "",
            "### Return Analysis",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Return | {self.performance.total_return}% |",
            f"| Annualized Return | {self.performance.annualized_return}% |",
            f"| Daily Return (Avg) | {self.stats['daily_return_mean']}% |",
            f"| Daily Return (Std) | {self.stats['daily_return_std']}% |",
            "",
            "### Risk-Adjusted Returns",
            "",
            f"| Metric | Value | Interpretation |",
            f"|--------|-------|----------------|",
            f"| Sharpe Ratio | {self.performance.sharpe_ratio} | {'Excellent' if self.performance.sharpe_ratio > 1 else 'Good' if self.performance.sharpe_ratio > 0.5 else 'Poor'} |",
            f"| Sortino Ratio | {self.performance.sortino_ratio} | {'Excellent' if self.performance.sortino_ratio > 1.5 else 'Good' if self.performance.sortino_ratio > 0.5 else 'Poor'} |",
            f"| Calmar Ratio | {self.performance.calmar_ratio} | - |",
            "",
            "### Trading Statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Win Rate | {self.performance.win_rate}% |",
            f"| Average Win | {self.performance.avg_win}% |",
            f"| Average Loss | {self.performance.avg_loss}% |",
            f"| Profit Factor | {self.performance.profit_factor} |",
            f"| Positive Days | {self.stats['positive_days']} |",
            f"| Negative Days | {self.stats['negative_days']} |",
            "",
        ])
        
        # Risk Metrics
        lines.extend([
            "---",
            "",
            "## ⚠️ Risk Analysis",
            "",
            "### Volatility Measures",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Annualized Volatility | {self.performance.volatility}% |",
            f"| Max Drawdown | {self.performance.max_drawdown}% |",
            "",
            "### Value at Risk (VaR)",
            "",
            f"| Confidence | Daily VaR | Interpretation |",
            f"|------------|-----------|----------------|",
            f"| 95% | {self.risk.var_95}% | 5% chance of losing more than this in a day |",
            f"| 99% | {self.risk.var_99}% | 1% chance of losing more than this in a day |",
            "",
            "### Conditional VaR (Expected Shortfall)",
            "",
            f"| Confidence | CVaR |",
            f"|------------|------|",
            f"| 95% | {self.risk.cvar_95}% |",
            f"| 99% | {self.risk.cvar_99}% |",
            "",
        ])
        
        # Distribution Analysis
        dist = self.analyzer.calculate_return_distribution()
        lines.extend([
            "---",
            "",
            "## 📉 Return Distribution",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Skewness | {dist['skew']} |",
            f"| Kurtosis | {dist['kurtosis']} |",
            f"| Min Daily Return | {dist['min']}% |",
            f"| Max Daily Return | {dist['max']}% |",
            "",
            "### Percentiles",
            "",
            f"| Percentile | Return |",
            f"|------------|--------|",
            f"| 5th | {dist['percentile_5']}% |",
            f"| 25th | {dist['percentile_25']}% |",
            f"| 50th (Median) | {dist['percentile_50']}% |",
            f"| 75th | {dist['percentile_75']}% |",
            f"| 95th | {dist['percentile_95']}% |",
            "",
        ])
        
        # Technical Indicators
        latest = self.data_with_indicators.iloc[-1]
        lines.extend([
            "---",
            "",
            "## 🔧 Current Technical Indicators",
            "",
            f"*As of {self.stats['end_date']}*",
            "",
            "### Moving Averages",
            "",
            f"| Indicator | Value | Signal |",
            f"|-----------|-------|--------|",
            f"| SMA 20 | ${latest.get('SMA_20', 0):.2f} | {'Bullish' if latest['Close'] > latest.get('SMA_20', 0) else 'Bearish'} |",
            f"| SMA 50 | ${latest.get('SMA_50', 0):.2f} | {'Bullish' if latest['Close'] > latest.get('SMA_50', 0) else 'Bearish'} |",
            f"| SMA 200 | ${latest.get('SMA_200', 0):.2f} | {'Bullish' if latest['Close'] > latest.get('SMA_200', 0) else 'Bearish'} |",
            "",
            "### Momentum Indicators",
            "",
            f"| Indicator | Value | Signal |",
            f"|-----------|-------|--------|",
        ])
        
        rsi_val = latest.get('RSI', 50)
        rsi_signal = 'Overbought' if rsi_val > 70 else 'Oversold' if rsi_val < 30 else 'Neutral'
        lines.append(f"| RSI (14) | {rsi_val:.2f} | {rsi_signal} |")
        
        macd_val = latest.get('MACD', 0)
        macd_signal_val = latest.get('MACD_Signal', 0)
        macd_signal = 'Bullish' if macd_val > macd_signal_val else 'Bearish'
        lines.append(f"| MACD | {macd_val:.4f} | {macd_signal} |")
        
        lines.extend([
            "",
            "### Volatility Indicators",
            "",
            f"| Indicator | Value |",
            f"|-----------|-------|",
            f"| Bollinger Upper | ${latest.get('BB_Upper', 0):.2f} |",
            f"| Bollinger Middle | ${latest.get('BB_Middle', 0):.2f} |",
            f"| Bollinger Lower | ${latest.get('BB_Lower', 0):.2f} |",
            f"| ATR (14) | ${latest.get('ATR', 0):.2f} |",
            "",
        ])
        
        # Disclaimer
        lines.extend([
            "---",
            "",
            "## ⚖️ Disclaimer",
            "",
            "*This report is for informational purposes only and should not be considered ",
            "as financial advice. Past performance does not guarantee future results. ",
            "Always conduct your own research and consult with a qualified financial ",
            "advisor before making investment decisions.*",
            "",
            "---",
            "",
            f"*Report generated by NVDA Stock Analyzer v2.0.0*",
        ])
        
        return "\n".join(lines)
    
    def generate_html_report(
        self,
        filepath: str,
        include_charts: bool = True,
    ) -> Path:
        """
        Generate an HTML format report.
        
        Args:
            filepath: Output file path
            include_charts: Whether to embed charts
            
        Returns:
            Path to generated report
        """
        path = Path(filepath)
        
        # Generate chart as base64 if needed
        chart_base64 = None
        if include_charts:
            import base64
            from io import BytesIO
            
            viz = StockVisualizer(self.data, symbol=self.symbol)
            viz.plot_technical_analysis()
            
            buf = BytesIO()
            viz.fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                           facecolor=viz.fig.get_facecolor())
            buf.seek(0)
            chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
            viz.close()
        
        html = self._build_html_content(chart_base64)
        path.write_text(html, encoding="utf-8")
        
        return path
    
    def _build_html_content(self, chart_base64: Optional[str] = None) -> str:
        """Build HTML report content."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        chart_html = ""
        if chart_base64:
            chart_html = f'''
            <div class="chart-container">
                <h2>📊 Technical Analysis Chart</h2>
                <img src="data:image/png;base64,{chart_base64}" alt="{self.symbol} Chart" class="chart-image">
            </div>
            '''
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.symbol} Stock Analysis Report</title>
    <style>
        :root {{
            --bg-primary: #1e1e2e;
            --bg-secondary: #313244;
            --text-primary: #cdd6f4;
            --text-secondary: #a6adc8;
            --accent: #89b4fa;
            --green: #a6e3a1;
            --red: #f38ba8;
            --yellow: #f9e2af;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        h1, h2, h3 {{
            color: var(--accent);
            margin-bottom: 1rem;
        }}
        
        h1 {{
            font-size: 2.5rem;
            text-align: center;
            padding: 2rem 0;
            border-bottom: 2px solid var(--bg-secondary);
        }}
        
        .meta {{
            text-align: center;
            color: var(--text-secondary);
            margin-bottom: 2rem;
        }}
        
        .section {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--bg-primary);
        }}
        
        th {{
            color: var(--accent);
            font-weight: 600;
        }}
        
        .positive {{
            color: var(--green);
        }}
        
        .negative {{
            color: var(--red);
        }}
        
        .chart-container {{
            text-align: center;
            margin: 2rem 0;
        }}
        
        .chart-image {{
            max-width: 100%;
            border-radius: 8px;
        }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .kpi-card {{
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}
        
        .kpi-value {{
            font-size: 1.75rem;
            font-weight: bold;
            color: var(--accent);
        }}
        
        .kpi-label {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        .disclaimer {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-style: italic;
            padding: 1rem;
            border-top: 1px solid var(--bg-secondary);
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 {self.symbol} Stock Analysis</h1>
        
        <div class="meta">
            <p>Generated: {now} | Period: {self.period} | Data Points: {self.stats['count']}</p>
        </div>
        
        <div class="section">
            <h2>📋 Key Performance Indicators</h2>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-value {'positive' if self.performance.total_return >= 0 else 'negative'}">{self.performance.total_return}%</div>
                    <div class="kpi-label">Total Return</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{self.performance.sharpe_ratio}</div>
                    <div class="kpi-label">Sharpe Ratio</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{self.performance.volatility}%</div>
                    <div class="kpi-label">Volatility</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value negative">{self.performance.max_drawdown}%</div>
                    <div class="kpi-label">Max Drawdown</div>
                </div>
            </div>
        </div>
        
        {chart_html}
        
        <div class="section">
            <h2>💰 Price Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Start Price</td><td>${self.stats['start_price']}</td></tr>
                <tr><td>End Price</td><td>${self.stats['end_price']}</td></tr>
                <tr><td>High</td><td>${self.stats['max_price']}</td></tr>
                <tr><td>Low</td><td>${self.stats['min_price']}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>📈 Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Annualized Return</td><td class="{'positive' if self.performance.annualized_return >= 0 else 'negative'}">{self.performance.annualized_return}%</td></tr>
                <tr><td>Sortino Ratio</td><td>{self.performance.sortino_ratio}</td></tr>
                <tr><td>Win Rate</td><td>{self.performance.win_rate}%</td></tr>
                <tr><td>Profit Factor</td><td>{self.performance.profit_factor}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>⚠️ Risk Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>VaR (95%)</td><td class="negative">{self.risk.var_95}%</td></tr>
                <tr><td>VaR (99%)</td><td class="negative">{self.risk.var_99}%</td></tr>
                <tr><td>CVaR (95%)</td><td class="negative">{self.risk.cvar_95}%</td></tr>
            </table>
        </div>
        
        <div class="disclaimer">
            ⚖️ This report is for informational purposes only and should not be considered 
            as financial advice. Past performance does not guarantee future results.
        </div>
    </div>
</body>
</html>'''
    
    def generate_json_report(self, filepath: str) -> Path:
        """
        Generate a JSON format report.
        
        Args:
            filepath: Output file path
            
        Returns:
            Path to generated report
        """
        path = Path(filepath)
        
        report = {
            "symbol": self.symbol,
            "period": self.period,
            "generated_at": datetime.now().isoformat(),
            "descriptive_stats": self.stats,
            "performance": self.performance.__dict__,
            "risk": self.risk.__dict__,
            "distribution": self.analyzer.calculate_return_distribution(),
        }
        
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        
        return path
    
    def generate_text_summary(self) -> str:
        """
        Generate a plain text summary.
        
        Returns:
            Text summary string
        """
        lines = [
            "=" * 60,
            f"  {self.symbol} STOCK ANALYSIS SUMMARY",
            "=" * 60,
            "",
            f"Period: {self.stats['start_date']} to {self.stats['end_date']}",
            "",
            "PRICE:",
            f"  Start: ${self.stats['start_price']}  →  End: ${self.stats['end_price']}",
            f"  High: ${self.stats['max_price']}  |  Low: ${self.stats['min_price']}",
            "",
            "RETURNS:",
            f"  Total Return: {self.performance.total_return}%",
            f"  Annualized: {self.performance.annualized_return}%",
            f"  Volatility: {self.performance.volatility}%",
            "",
            "RISK-ADJUSTED:",
            f"  Sharpe Ratio: {self.performance.sharpe_ratio}",
            f"  Sortino Ratio: {self.performance.sortino_ratio}",
            f"  Max Drawdown: {self.performance.max_drawdown}%",
            "",
            "RISK:",
            f"  VaR (95%): {self.risk.var_95}%",
            f"  CVaR (95%): {self.risk.cvar_95}%",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)

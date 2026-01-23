"""
Command-Line Interface for NVDA Stock Analyzer.

Provides a comprehensive CLI for running analyses, generating reports,
and visualizing stock data.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from . import __version__
from .config import get_config
from .data_fetcher import DataFetcher, DataFetcherError
from .technical_indicators import TechnicalIndicators
from .statistical_analysis import StatisticalAnalysis
from .visualizer import StockVisualizer


def create_console():
    """Create console for output."""
    if RICH_AVAILABLE:
        return Console()
    return None


def print_header(console):
    """Print application header."""
    if console:
        console.print(Panel.fit(
            f"[bold cyan]NVDA Stock Analyzer[/bold cyan] v{__version__}\n"
            "[dim]Professional Quantitative Analysis Toolkit[/dim]",
            border_style="cyan"
        ))
    else:
        print(f"\n{'='*50}")
        print(f"  NVDA Stock Analyzer v{__version__}")
        print(f"  Professional Quantitative Analysis Toolkit")
        print(f"{'='*50}\n")


def print_stats_table(console, stats: dict, title: str):
    """Print statistics as a formatted table."""
    if console:
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        
        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)
    else:
        print(f"\n{title}")
        print("-" * 40)
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")


def run_analyze(args):
    """Run full analysis on a stock."""
    console = create_console()
    print_header(console)
    
    symbol = args.symbol.upper()
    period = args.period
    interval = args.interval
    
    if console:
        console.print(f"\n[bold]Analyzing {symbol}[/bold] - Period: {period}, Interval: {interval}\n")
    else:
        print(f"\nAnalyzing {symbol} - Period: {period}, Interval: {interval}\n")
    
    # Fetch data
    try:
        fetcher = DataFetcher()
        
        if console and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching market data...", total=None)
                data = fetcher.fetch(symbol, period, interval)
                progress.update(task, completed=True)
        else:
            print("Fetching market data...")
            data = fetcher.fetch(symbol, period, interval)
        
    except DataFetcherError as e:
        if console:
            console.print(f"[red]Error:[/red] {e}")
        else:
            print(f"Error: {e}")
        return 1
    
    # Statistical Analysis
    analyzer = StatisticalAnalysis(data)
    
    # Print descriptive stats
    desc_stats = analyzer.get_descriptive_stats()
    print_stats_table(console, desc_stats, "📊 Descriptive Statistics")
    
    # Print performance metrics
    perf = analyzer.calculate_performance_metrics()
    perf_dict = {
        "Total Return": f"{perf.total_return}%",
        "Annualized Return": f"{perf.annualized_return}%",
        "Volatility": f"{perf.volatility}%",
        "Sharpe Ratio": f"{perf.sharpe_ratio}",
        "Sortino Ratio": f"{perf.sortino_ratio}",
        "Max Drawdown": f"{perf.max_drawdown}%",
        "Win Rate": f"{perf.win_rate}%",
    }
    print_stats_table(console, perf_dict, "📈 Performance Metrics")
    
    # Print risk metrics
    risk = analyzer.calculate_risk_metrics()
    risk_dict = {
        "VaR (95%)": f"{risk.var_95}%",
        "VaR (99%)": f"{risk.var_99}%",
        "CVaR (95%)": f"{risk.cvar_95}%",
        "CVaR (99%)": f"{risk.cvar_99}%",
    }
    print_stats_table(console, risk_dict, "⚠️ Risk Metrics")
    
    # Generate chart if requested
    if args.output:
        if console:
            console.print("\n[bold]Generating visualization...[/bold]")
        else:
            print("\nGenerating visualization...")
        
        viz = StockVisualizer(data, symbol=symbol)
        viz.plot_technical_analysis()
        
        output_path = Path(args.output)
        viz.save(str(output_path))
        
        if console:
            console.print(f"[green]Chart saved to:[/green] {output_path.absolute()}")
        else:
            print(f"Chart saved to: {output_path.absolute()}")
    
    return 0


def run_chart(args):
    """Generate chart only."""
    console = create_console()
    print_header(console)
    
    symbol = args.symbol.upper()
    chart_type = args.type
    
    if console:
        console.print(f"\n[bold]Generating {chart_type} chart for {symbol}[/bold]\n")
    else:
        print(f"\nGenerating {chart_type} chart for {symbol}\n")
    
    try:
        fetcher = DataFetcher()
        data = fetcher.fetch(symbol, args.period, args.interval)
    except DataFetcherError as e:
        if console:
            console.print(f"[red]Error:[/red] {e}")
        else:
            print(f"Error: {e}")
        return 1
    
    viz = StockVisualizer(data, symbol=symbol)
    
    if chart_type == "price":
        viz.plot_price()
    elif chart_type == "candlestick":
        viz.plot_candlestick(tail_days=args.days)
    elif chart_type == "technical":
        viz.plot_technical_analysis(tail_days=args.days)
    else:
        viz.plot_technical_analysis()
    
    # Save or show
    if args.output:
        output_path = Path(args.output)
        viz.save(str(output_path))
        if console:
            console.print(f"[green]Chart saved to:[/green] {output_path.absolute()}")
        else:
            print(f"Chart saved to: {output_path.absolute()}")
    else:
        viz.show()
    
    return 0


def run_info(args):
    """Get stock information."""
    console = create_console()
    print_header(console)
    
    symbol = args.symbol.upper()
    
    fetcher = DataFetcher()
    info = fetcher.get_stock_info(symbol)
    
    print_stats_table(console, info, f"ℹ️ {symbol} Information")
    
    return 0


def main(argv=None):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="nvda-analyzer",
        description="Professional Stock Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nvda-analyzer analyze NVDA --period 1y --output analysis.png
  nvda-analyzer chart NVDA --type candlestick --days 60
  nvda-analyzer info AAPL
        """
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run full analysis on a stock"
    )
    analyze_parser.add_argument("symbol", help="Stock ticker symbol (e.g., NVDA)")
    analyze_parser.add_argument(
        "--period", "-p",
        default="1y",
        help="Historical period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)"
    )
    analyze_parser.add_argument(
        "--interval", "-i",
        default="1d",
        help="Data interval (1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        help="Output file path for chart"
    )
    analyze_parser.set_defaults(func=run_analyze)
    
    # Chart command
    chart_parser = subparsers.add_parser(
        "chart",
        help="Generate a chart"
    )
    chart_parser.add_argument("symbol", help="Stock ticker symbol")
    chart_parser.add_argument(
        "--type", "-t",
        choices=["price", "candlestick", "technical"],
        default="technical",
        help="Chart type"
    )
    chart_parser.add_argument(
        "--period", "-p",
        default="1y",
        help="Historical period"
    )
    chart_parser.add_argument(
        "--interval", "-i",
        default="1d",
        help="Data interval"
    )
    chart_parser.add_argument(
        "--days", "-d",
        type=int,
        default=180,
        help="Number of days to show"
    )
    chart_parser.add_argument(
        "--output", "-o",
        help="Output file path (if not specified, displays chart)"
    )
    chart_parser.set_defaults(func=run_chart)
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Get stock information"
    )
    info_parser.add_argument("symbol", help="Stock ticker symbol")
    info_parser.set_defaults(func=run_info)
    
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

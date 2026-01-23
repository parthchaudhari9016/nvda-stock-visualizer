"""
NVDA Stock Analyzer - Professional Quantitative Analysis Toolkit

A comprehensive Python package for stock market analysis featuring
technical indicators, statistical metrics, and professional-grade
visualizations.
"""

__version__ = "2.0.0"
__author__ = "Parth Chaudhari"

from .data_fetcher import DataFetcher
from .technical_indicators import TechnicalIndicators
from .statistical_analysis import StatisticalAnalysis
from .visualizer import StockVisualizer

__all__ = [
    "DataFetcher",
    "TechnicalIndicators",
    "StatisticalAnalysis",
    "StockVisualizer",
    "__version__",
]

"""
FinAgent Analysis Module

This module contains analysis components:
- TechnicalAnalyzer: TA-Lib based technical indicators
- FundamentalAnalyzer: Financial ratio analysis
- ChainOfThoughtSynthesizer: Structured reasoning engine
- NewsScreener: News-based stock screening and catalyst detection
"""

from .technical import TechnicalAnalyzer
from .fundamental import FundamentalAnalyzer
from .synthesizer import ChainOfThoughtSynthesizer
from .news_screener import NewsScreener, TimeHorizon, CatalystType, StockRecommendation

__all__ = [
    "TechnicalAnalyzer",
    "FundamentalAnalyzer",
    "ChainOfThoughtSynthesizer",
    "NewsScreener",
    "TimeHorizon",
    "CatalystType",
    "StockRecommendation",
]

"""
FinAgent Analysis Module

This module contains analysis components:
- TechnicalAnalyzer: TA-Lib based technical indicators
- FundamentalAnalyzer: Financial ratio analysis
- ChainOfThoughtSynthesizer: Structured reasoning engine
- NewsScreener: News-based stock screening and catalyst detection
- FinancialSentimentAnalyzer: NLP-based sentiment analysis (FinBERT/VADER)
"""

from .technical import TechnicalAnalyzer
from .fundamental import FundamentalAnalyzer
from .synthesizer import ChainOfThoughtSynthesizer
from .news_screener import NewsScreener, TimeHorizon, CatalystType, StockRecommendation
from .sentiment_analyzer import (
    FinancialSentimentAnalyzer,
    SentimentResult,
    SentimentLabel,
    analyze_sentiment,
)

__all__ = [
    "TechnicalAnalyzer",
    "FundamentalAnalyzer",
    "ChainOfThoughtSynthesizer",
    "NewsScreener",
    "TimeHorizon",
    "CatalystType",
    "StockRecommendation",
    "FinancialSentimentAnalyzer",
    "SentimentResult",
    "SentimentLabel",
    "analyze_sentiment",
]

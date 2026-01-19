"""
FinAgent Analysis Module

This module contains analysis components:
- TechnicalAnalyzer: TA-Lib based technical indicators
- FundamentalAnalyzer: Financial ratio analysis
- ChainOfThoughtSynthesizer: Structured reasoning engine
- NewsScreener: News-based stock screening and catalyst detection
- FinancialSentimentAnalyzer: NLP-based sentiment analysis (FinBERT/VADER)
- LiveNewsFetcher: Real-time news fetching from Yahoo Finance and Google News
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
from .live_news_fetcher import (
    LiveNewsFetcher,
    NewsItem,
    fetch_live_news,
    get_live_news_fetcher,
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
    "LiveNewsFetcher",
    "NewsItem",
    "fetch_live_news",
    "get_live_news_fetcher",
]

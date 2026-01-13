"""
FinAgent Data Pipelines Module

This module contains data ingestion pipelines:
- MarketDataPipeline: Structured market data (OHLCV)
- CircularPipeline: Unstructured regulatory documents
- SentimentPipeline: News and sentiment data
"""

from .structured_data import MarketDataPipeline
from .unstructured_data import CircularPipeline
from .sentiment_data import SentimentPipeline

__all__ = ["MarketDataPipeline", "CircularPipeline", "SentimentPipeline"]

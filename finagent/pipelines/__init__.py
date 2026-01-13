"""
FinAgent Data Pipelines Module

This module contains data ingestion pipelines:
- MarketDataPipeline: Structured market data (OHLCV)
- CircularPipeline: Unstructured regulatory documents (BSE/NSE circulars)
- SentimentPipeline: News and sentiment data
- PressReleasePipeline: Company press releases
- IngestionOrchestrator: Batch ingestion coordinator
"""

from .structured_data import MarketDataPipeline
from .unstructured_data import CircularPipeline, CircularDocument, DocumentType
from .sentiment_data import SentimentPipeline
from .press_releases import PressReleasePipeline, PressRelease
from .ingestion_orchestrator import IngestionOrchestrator, IngestionJob, IngestionResult

__all__ = [
    "MarketDataPipeline",
    "CircularPipeline",
    "CircularDocument",
    "DocumentType",
    "SentimentPipeline",
    "PressReleasePipeline",
    "PressRelease",
    "IngestionOrchestrator",
    "IngestionJob",
    "IngestionResult",
]

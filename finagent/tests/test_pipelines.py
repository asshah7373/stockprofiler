"""
Tests for Data Pipelines
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMarketDataPipeline:
    """Tests for structured market data pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance with mocked DB."""
        from pipelines.structured_data import MarketDataPipeline
        return MarketDataPipeline(cache_db=":memory:")

    def test_normalize_ticker(self, pipeline):
        """Test ticker normalization."""
        assert pipeline._normalize_ticker("RELIANCE") == "RELIANCE.NS"
        assert pipeline._normalize_ticker("RELIANCE.NS") == "RELIANCE.NS"
        assert pipeline._normalize_ticker("reliance") == "RELIANCE.NS"
        assert pipeline._normalize_ticker("TCS.BO") == "TCS.BO"

    def test_market_cap_categorization(self, pipeline):
        """Test market cap categorization."""
        # Large cap (>20000 Cr = >200 billion)
        assert pipeline._categorize_market_cap(300_000_000_000) == "LARGE"

        # Mid cap (5000-20000 Cr)
        assert pipeline._categorize_market_cap(100_000_000_000) == "MID"

        # Small cap (<5000 Cr)
        assert pipeline._categorize_market_cap(10_000_000_000) == "SMALL"

        # None
        assert pipeline._categorize_market_cap(None) is None

    @patch('yfinance.Ticker')
    def test_get_live_quote(self, mock_ticker, pipeline):
        """Test live quote fetching."""
        mock_info = {
            'regularMarketPrice': 2500.0,
            'regularMarketPreviousClose': 2480.0,
            'regularMarketVolume': 1000000,
            'dayHigh': 2520.0,
            'dayLow': 2470.0
        }
        mock_ticker.return_value.info = mock_info

        quote = pipeline.get_live_quote("RELIANCE.NS")

        assert quote['ticker'] == "RELIANCE.NS"
        assert quote['price'] == 2500.0
        assert quote['source'] == 'yfinance'


class TestSentimentPipeline:
    """Tests for sentiment analysis pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance."""
        from pipelines.sentiment_data import SentimentPipeline
        return SentimentPipeline()

    def test_calculate_article_sentiment_positive(self, pipeline):
        """Test positive sentiment detection."""
        headline = "Stock price surges after strong quarterly results"
        score = pipeline._calculate_article_sentiment(headline)
        assert score > 0

    def test_calculate_article_sentiment_negative(self, pipeline):
        """Test negative sentiment detection."""
        headline = "Company shares crash amid fraud investigation"
        score = pipeline._calculate_article_sentiment(headline)
        assert score < 0

    def test_calculate_article_sentiment_neutral(self, pipeline):
        """Test neutral sentiment."""
        headline = "Company announces board meeting"
        score = pipeline._calculate_article_sentiment(headline)
        assert score == 0

    def test_aggregate_sentiment(self, pipeline):
        """Test sentiment aggregation."""
        articles = [
            {"sentiment_score": 0.5, "source": "reuters", "published_at": datetime.now().isoformat()},
            {"sentiment_score": -0.3, "source": "moneycontrol", "published_at": datetime.now().isoformat()},
            {"sentiment_score": 0.2, "source": "unknown", "published_at": datetime.now().isoformat()}
        ]

        score = pipeline.aggregate_sentiment(articles)
        assert -1 <= score <= 1

    def test_aggregate_sentiment_empty(self, pipeline):
        """Test aggregation with empty list."""
        assert pipeline.aggregate_sentiment([]) == 0.0


class TestCircularPipeline:
    """Tests for circular/document pipeline."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create pipeline with temp directories."""
        from pipelines.unstructured_data import CircularPipeline
        return CircularPipeline(
            data_dir=str(tmp_path / "circulars"),
            db_path=str(tmp_path / "circulars.db")
        )

    def test_generate_doc_id(self, pipeline):
        """Test document ID generation."""
        id1 = pipeline._generate_doc_id("http://example.com/doc1", "Title 1")
        id2 = pipeline._generate_doc_id("http://example.com/doc2", "Title 2")
        id3 = pipeline._generate_doc_id("http://example.com/doc1", "Title 1")

        assert id1 != id2
        assert id1 == id3  # Same input = same ID
        assert len(id1) == 16

    def test_chunk_for_rag(self, pipeline):
        """Test document chunking."""
        content = {
            "text": "This is sentence one. This is sentence two. " * 50,
            "metadata": {"pages": 1}
        }

        chunks = pipeline.chunk_for_rag(content, "test_doc", chunk_size=100, overlap=20)

        assert len(chunks) > 1
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["doc_id"] == "test_doc"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

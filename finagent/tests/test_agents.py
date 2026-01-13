"""
Tests for Agent Modules
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAnalystAgent:
    """Tests for Analyst Agent."""

    @pytest.fixture
    def analyst(self):
        """Create analyst instance."""
        from agents.analyst import AnalystAgent
        return AnalystAgent()

    def test_macro_analysis_positive(self, analyst):
        """Test macro analysis with positive sentiment."""
        company_info = {"sector": "Technology", "industry": "Software"}
        sentiment_data = {
            "overall_sentiment": 0.5,
            "sentiment_trend": "POSITIVE"
        }

        result = analyst._analyze_macro_context(company_info, sentiment_data)

        assert result["sector"] == "Technology"
        assert result["market_mood"] == "POSITIVE"
        assert result["sentiment_impact"] == 1

    def test_macro_analysis_negative(self, analyst):
        """Test macro analysis with negative sentiment."""
        company_info = {"sector": "Banking"}
        sentiment_data = {
            "overall_sentiment": -0.5,
            "sentiment_trend": "NEGATIVE"
        }

        result = analyst._analyze_macro_context(company_info, sentiment_data)

        assert result["market_mood"] == "NEGATIVE"
        assert result["sentiment_impact"] == -1

    def test_fundamental_analysis(self, analyst):
        """Test fundamental analysis."""
        company_info = {
            "pe_ratio": 15.0,
            "eps": 50.0,
            "revenue_growth": 0.20,
            "profit_margin": 0.18
        }
        rag_context = {"sources": [{"doc_id": "test123"}]}

        result = analyst._analyze_fundamentals(company_info, rag_context)

        assert result["pe_ratio"] == 15.0
        assert result["valuation_score"] > 0  # Low PE is positive
        assert "test123" in result["sources"]

    def test_technical_analysis_bullish(self, analyst):
        """Test technical analysis with bullish signals."""
        technical_data = {
            "rsi_14": 25.0,  # Oversold
            "macd": {"value": 1.0, "signal": 0.5, "histogram": 0.5},
            "signals": {
                "rsi_signal": "OVERSOLD",
                "macd_signal": "BULLISH",
                "trend": "UPTREND"
            }
        }

        result = analyst._analyze_technicals(technical_data)

        assert result["tech_score"] > 0
        assert result["assessment"] in ["BULLISH", "STRONGLY_BULLISH"]

    def test_risk_assessment(self, analyst):
        """Test risk assessment."""
        technical_data = {
            "beta": 1.8,
            "volatility_30d": 0.4
        }
        company_info = {
            "market_cap_category": "SMALL",
            "debt_to_equity": 120
        }
        user_profile = {"risk_category": "CONSERVATIVE"}

        result = analyst._assess_risk(technical_data, company_info, user_profile)

        assert result["risk_level"] in ["HIGH", "VERY_HIGH"]
        assert result["suitable_for_profile"] is False


class TestReviewerAgent:
    """Tests for Reviewer Agent."""

    @pytest.fixture
    def reviewer(self):
        """Create reviewer instance."""
        from agents.reviewer import ReviewerAgent
        return ReviewerAgent()

    def test_logical_consistency_valid(self, reviewer):
        """Test logical consistency check with valid data."""
        rationale = {
            "technical": "bullish trend, oversold conditions",
            "fundamental": "undervalued, strong growth"
        }

        result = reviewer._check_logical_consistency(rationale, "BUY", 0.7)

        assert len(result["errors"]) == 0

    def test_logical_consistency_invalid(self, reviewer):
        """Test detection of logical inconsistency."""
        rationale = {
            "technical": "bearish trend, overbought, weak",
            "fundamental": "overvalued, declining, negative"
        }

        result = reviewer._check_logical_consistency(rationale, "BUY", 0.8)

        assert len(result["errors"]) > 0

    def test_verify_technicals(self, reviewer):
        """Test technical indicator verification."""
        rationale = {
            "technical_analysis": "RSI at 45.0 shows NEUTRAL conditions"
        }
        calculated = {
            "rsi_14": 45.0,
            "signals": {"macd_signal": "NEUTRAL"}
        }

        result = reviewer._verify_technicals(rationale, calculated)

        assert len(result["verified"]) > 0
        assert len(result["errors"]) == 0

    def test_hallucination_check(self, reviewer):
        """Test hallucination detection."""
        recommendation = {
            "rationale": {
                "analysis": "Target price of Rs 2500 by next month"
            },
            "sources_cited": []
        }

        result = reviewer._check_for_hallucinations(recommendation)

        assert len(result["warnings"]) > 0


class TestPerceiverAgent:
    """Tests for Perceiver Agent."""

    @pytest.fixture
    def perceiver(self):
        """Create perceiver with mocked dependencies."""
        from agents.perceiver import PerceiverAgent
        from unittest.mock import Mock

        perceiver = PerceiverAgent(
            market_pipeline=Mock(),
            circular_pipeline=Mock(),
            sentiment_pipeline=Mock(),
            vector_store=Mock()
        )
        return perceiver

    def test_perceive_structure(self, perceiver):
        """Test perceive returns correct structure."""
        # Mock the fetch methods
        perceiver.market.get_live_quote.return_value = {"price": 100}
        perceiver.market.get_company_info.return_value = {"name": "Test Co"}
        perceiver.market.get_historical_data.return_value = Mock(empty=True)
        perceiver.sentiment.get_sentiment_summary.return_value = {}

        result = perceiver.perceive("TEST.NS")

        assert "ticker" in result
        assert "market_data" in result
        assert "company_info" in result
        assert "perceived_at" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

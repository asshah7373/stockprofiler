"""
Tests for Compliance and Risk Profile Modules
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRiskProfiler:
    """Tests for risk profiling questionnaire."""

    @pytest.fixture
    def profiler(self, tmp_path):
        """Create profiler with temp database."""
        from risk_profile.questionnaire import RiskProfiler
        return RiskProfiler(db_path=str(tmp_path / "profiles.db"))

    def test_conduct_profiling_conservative(self, profiler):
        """Test conservative profile creation."""
        # All lowest risk answers
        responses = {
            "age": 3,  # 60+
            "income": 0,  # Below 5 Lakhs
            "net_worth": 0,
            "experience": 0,  # None
            "knowledge": 0,
            "horizon": 0,  # Less than 1 year
            "risk_scenario": 0,  # Sell everything
            "capital": 0,  # More than 50%
            "goal": 0,  # Capital preservation
            "loss_tolerance": 0  # 0-5%
        }

        profile = profiler.conduct_profiling(responses)

        assert profile.risk_category == "CONSERVATIVE"
        assert profile.risk_score <= 7

    def test_conduct_profiling_aggressive(self, profiler):
        """Test aggressive profile creation."""
        # All highest risk answers
        responses = {
            "age": 0,  # 18-30
            "income": 3,  # Above 50 Lakhs
            "net_worth": 3,
            "experience": 3,  # 7+ years
            "knowledge": 3,
            "horizon": 3,  # More than 5 years
            "risk_scenario": 3,  # Buy more
            "capital": 3,  # Less than 10%
            "goal": 3,  # Aggressive growth
            "loss_tolerance": 3  # More than 30%
        }

        profile = profiler.conduct_profiling(responses)

        assert profile.risk_category == "AGGRESSIVE"
        assert profile.risk_score >= 15

    def test_constraints_conservative(self, profiler):
        """Test conservative profile constraints."""
        responses = {q["id"]: 0 for q in profiler.QUESTIONS}
        profile = profiler.conduct_profiling(responses)

        constraints = profile.constraints

        assert constraints["max_beta"] <= 1.0
        assert constraints["min_market_cap"] == "LARGE"
        assert "F&O" in constraints["excluded_instruments"]
        assert constraints["prefer_dividend"] is True

    def test_profile_expiry(self, profiler):
        """Test profile expiry check."""
        responses = {q["id"]: 1 for q in profiler.QUESTIONS}
        profile = profiler.conduct_profiling(responses)

        # Fresh profile should not be expired
        assert not profiler.is_profile_expired(profile)


class TestConstraintApplier:
    """Tests for constraint application."""

    @pytest.fixture
    def applier(self):
        """Create constraint applier."""
        from risk_profile.constraints import ConstraintApplier
        return ConstraintApplier()

    @pytest.fixture
    def conservative_profile(self, tmp_path):
        """Create a conservative profile."""
        from risk_profile.questionnaire import RiskProfiler
        profiler = RiskProfiler(db_path=str(tmp_path / "profiles.db"))
        responses = {q["id"]: 0 for q in profiler.QUESTIONS}
        return profiler.conduct_profiling(responses)

    def test_check_beta_violation(self, applier):
        """Test beta constraint violation detection."""
        result = applier._check_beta(1.5, 0.8)

        assert result is not None
        assert result["is_violation"] is True

    def test_check_beta_pass(self, applier):
        """Test beta constraint pass."""
        result = applier._check_beta(0.7, 0.8)

        assert result is None

    def test_check_market_cap_violation(self, applier):
        """Test market cap constraint violation."""
        result = applier._check_market_cap("SMALL", "LARGE")

        assert result is not None
        assert result["is_violation"] is True

    def test_check_market_cap_pass(self, applier):
        """Test market cap constraint pass."""
        result = applier._check_market_cap("LARGE", "LARGE")

        assert result is None

    def test_suitability_check_fail(self, applier, conservative_profile):
        """Test suitability check with unsuitable stock."""
        stock_data = {
            "beta": 1.8,
            "market_cap_category": "SMALL",
            "sector": "Crypto",
            "debt_to_equity": 200
        }

        result = applier.check_suitability(stock_data, conservative_profile)

        assert result.is_suitable is False
        assert len(result.violations) > 0

    def test_suitability_check_pass(self, applier, conservative_profile):
        """Test suitability check with suitable stock."""
        stock_data = {
            "beta": 0.6,
            "market_cap_category": "LARGE",
            "sector": "FMCG",
            "debt_to_equity": 30,
            "dividend_yield": 0.03
        }

        result = applier.check_suitability(stock_data, conservative_profile)

        assert result.is_suitable is True
        assert len(result.violations) == 0


class TestComplianceLogger:
    """Tests for compliance logging."""

    @pytest.fixture
    def logger(self, tmp_path):
        """Create compliance logger with temp database."""
        from utils.compliance_logger import ComplianceLogger
        return ComplianceLogger(db_path=str(tmp_path / "compliance.db"))

    def test_log_interaction(self, logger):
        """Test interaction logging."""
        interaction_id = logger.log_interaction(
            interaction_type="analysis",
            query="Analyze RELIANCE.NS",
            response="Analysis complete",
            recommendation={"action": "BUY", "ticker": "RELIANCE.NS"},
            sources=["yfinance", "newsapi"],
            risk_profile={"profile_id": "test123", "risk_category": "MODERATE"}
        )

        assert interaction_id is not None
        assert len(interaction_id) == 36  # UUID length

    def test_verify_integrity(self, logger):
        """Test data integrity verification."""
        interaction_id = logger.log_interaction(
            interaction_type="test",
            query="Test query",
            response="Test response"
        )

        assert logger.verify_integrity(interaction_id) is True

    def test_get_audit_trail(self, logger):
        """Test audit trail retrieval."""
        # Log some interactions
        for i in range(5):
            logger.log_interaction(
                interaction_type="test",
                query=f"Query {i}",
                response=f"Response {i}"
            )

        trail = logger.get_audit_trail(limit=10)

        assert len(trail) == 5

    def test_compliance_summary(self, logger):
        """Test compliance summary generation."""
        logger.log_interaction(
            interaction_type="analysis",
            query="Test",
            response="Response"
        )

        summary = logger.get_compliance_summary()

        assert "total_interactions" in summary
        assert summary["total_interactions"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

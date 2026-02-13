"""
Reviewer Agent - Fact-Checker
Validates Analyst recommendations against source documents.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Result of reviewing an analysis."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    verified_claims: List[str]
    review_details: Dict[str, Any]


class ReviewerAgent:
    """
    Fact-Checking Agent (Reviewer)

    Responsibilities:
    1. Verify all cited figures exist in source documents
    2. Check for numerical contradictions
    3. Validate technical indicators match calculations
    4. Ensure recommendation aligns with stated rationale

    CONSTRAINTS:
    - ONLY validate, do not generate recommendations
    - Flag any unsupported claims
    - Maximum 3 regeneration requests before escalation
    """

    # Tolerance for floating-point comparisons
    NUMERIC_TOLERANCE = 0.01

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def review(
        self,
        recommendation: Dict,
        source_documents: Dict,
        calculated_technicals: Dict,
        perceived_data: Optional[Dict] = None
    ) -> ReviewResult:
        """
        Validate recommendation against sources.

        Args:
            recommendation: Analyst's recommendation (AnalysisResult as dict)
            source_documents: RAG retrieved documents
            calculated_technicals: Technical analysis results
            perceived_data: Raw data from Perceiver

        Returns:
            ReviewResult with validation status
        """
        errors = []
        warnings = []
        verified_claims = []
        review_details = {}

        # 1. Verify citations exist
        citation_result = self._verify_citations(
            recommendation.get("sources_cited", []),
            source_documents
        )
        verified_claims.extend(citation_result["verified"])
        errors.extend(citation_result["errors"])
        warnings.extend(citation_result["warnings"])
        review_details["citations"] = citation_result

        # 2. Verify technical indicators
        if calculated_technicals:
            tech_result = self._verify_technicals(
                recommendation.get("rationale", {}),
                calculated_technicals
            )
            verified_claims.extend(tech_result["verified"])
            errors.extend(tech_result["errors"])
            review_details["technicals"] = tech_result

        # 3. Check logical consistency
        logic_result = self._check_logical_consistency(
            recommendation.get("rationale", {}),
            recommendation.get("action"),
            recommendation.get("confidence_score")
        )
        errors.extend(logic_result["errors"])
        warnings.extend(logic_result["warnings"])
        review_details["logic"] = logic_result

        # 4. Verify market data if available
        if perceived_data:
            data_result = self._verify_market_data(
                recommendation,
                perceived_data
            )
            verified_claims.extend(data_result["verified"])
            errors.extend(data_result["errors"])
            review_details["market_data"] = data_result

        # 5. Check for hallucination indicators
        hallucination_result = self._check_for_hallucinations(recommendation)
        errors.extend(hallucination_result["errors"])
        warnings.extend(hallucination_result["warnings"])
        review_details["hallucination_check"] = hallucination_result

        # Determine overall validity
        is_valid = len(errors) == 0

        return ReviewResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            verified_claims=verified_claims,
            review_details=review_details
        )

    def _verify_citations(
        self,
        cited_sources: List[str],
        source_documents: Dict
    ) -> Dict:
        """
        Verify each cited source exists.

        Returns:
            {
                "verified": List of verified source IDs,
                "errors": List of errors,
                "warnings": List of warnings
            }
        """
        verified = []
        errors = []
        warnings = []

        available_sources = set()

        # Get available sources from RAG context
        if source_documents:
            for source in source_documents.get("sources", []):
                available_sources.add(source.get("doc_id", ""))

            # Also check chunks
            for chunk in source_documents.get("chunks", []):
                available_sources.add(chunk.get("doc_id", ""))

        for source_id in cited_sources:
            if source_id in available_sources:
                verified.append(f"Source {source_id} exists in document store")
            else:
                # Not an error if we don't have source documents
                if source_documents:
                    warnings.append(f"Cited source {source_id} not found in retrieved documents")

        if not cited_sources:
            warnings.append("No sources cited in the recommendation")

        return {
            "verified": verified,
            "errors": errors,
            "warnings": warnings,
            "cited_count": len(cited_sources),
            "verified_count": len(verified)
        }

    def _verify_technicals(
        self,
        rationale: Dict,
        calculated: Dict
    ) -> Dict:
        """
        Verify technical values match calculations.

        Returns:
            {
                "verified": List of verified claims,
                "errors": List of discrepancies
            }
        """
        verified = []
        errors = []

        # Get technical analysis reasoning
        technical_reasoning = rationale.get("technical_analysis", "")

        # Check RSI
        if calculated.get("rsi_14") is not None:
            stated_rsi = self._extract_number(technical_reasoning, "RSI")
            if stated_rsi is not None:
                if self._values_match(stated_rsi, calculated["rsi_14"]):
                    verified.append(f"RSI value {stated_rsi:.1f} matches calculation")
                else:
                    errors.append(
                        f"RSI discrepancy: stated {stated_rsi:.1f}, "
                        f"calculated {calculated['rsi_14']:.1f}"
                    )

        # Check MACD
        if calculated.get("macd", {}).get("value") is not None:
            # Just verify signal is mentioned correctly
            macd_signal = calculated.get("signals", {}).get("macd_signal", "")
            if macd_signal.upper() in technical_reasoning.upper():
                verified.append(f"MACD signal {macd_signal} correctly stated")

        # Check trend
        trend = calculated.get("signals", {}).get("trend", "")
        if trend and trend.upper() in technical_reasoning.upper():
            verified.append(f"Trend {trend} correctly stated")

        return {
            "verified": verified,
            "errors": errors
        }

    def _extract_number(self, text: str, keyword: str) -> Optional[float]:
        """Extract a number following a keyword from text."""
        pattern = rf'{keyword}\s*[:=]?\s*(\d+\.?\d*)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    def _values_match(self, value1: float, value2: float) -> bool:
        """Check if two values are approximately equal."""
        if value1 == 0 and value2 == 0:
            return True
        if value1 == 0 or value2 == 0:
            return abs(value1 - value2) < self.NUMERIC_TOLERANCE
        return abs(value1 - value2) / max(abs(value1), abs(value2)) < self.NUMERIC_TOLERANCE

    def _check_logical_consistency(
        self,
        rationale: Dict,
        action: str,
        confidence: float
    ) -> Dict:
        """
        Check if conclusion follows from rationale.

        E.g., bullish rationale shouldn't lead to SELL.
        """
        errors = []
        warnings = []

        # Combine all rationale text
        all_reasoning = " ".join(str(v) for v in rationale.values()).lower()

        # Check for contradictions
        bullish_indicators = [
            "bullish", "uptrend", "oversold", "undervalued",
            "strong", "positive", "growth", "buy signal"
        ]
        bearish_indicators = [
            "bearish", "downtrend", "overbought", "overvalued",
            "weak", "negative", "decline", "sell signal"
        ]

        bullish_count = sum(1 for ind in bullish_indicators if ind in all_reasoning)
        bearish_count = sum(1 for ind in bearish_indicators if ind in all_reasoning)

        # Check for action/reasoning mismatch
        if action == "BUY" and bearish_count > bullish_count + 2:
            errors.append(
                f"Logic inconsistency: BUY recommendation but rationale "
                f"has more bearish ({bearish_count}) than bullish ({bullish_count}) signals"
            )

        if action == "SELL" and bullish_count > bearish_count + 2:
            errors.append(
                f"Logic inconsistency: SELL recommendation but rationale "
                f"has more bullish ({bullish_count}) than bearish ({bearish_count}) signals"
            )

        # Check confidence alignment
        if confidence > 0.7 and abs(bullish_count - bearish_count) < 2:
            warnings.append(
                f"High confidence ({confidence:.0%}) but mixed signals in rationale"
            )

        if confidence < 0.4 and action != "HOLD":
            warnings.append(
                f"Low confidence ({confidence:.0%}) but recommending {action}"
            )

        return {
            "errors": errors,
            "warnings": warnings,
            "bullish_signals": bullish_count,
            "bearish_signals": bearish_count
        }

    def _verify_market_data(
        self,
        recommendation: Dict,
        perceived_data: Dict
    ) -> Dict:
        """
        Verify market data in recommendation matches source.
        """
        verified = []
        errors = []

        market_data = perceived_data.get("market_data", {})
        company_info = perceived_data.get("company_info", {})

        # Check current price
        rec_price = recommendation.get("current_price")
        actual_price = market_data.get("price")

        if rec_price is not None and actual_price is not None:
            if self._values_match(rec_price, actual_price):
                verified.append(f"Current price {rec_price} verified")
            else:
                errors.append(
                    f"Price mismatch: recommendation shows {rec_price}, "
                    f"actual price is {actual_price}"
                )

        # Check company name
        rec_name = recommendation.get("company_name")
        actual_name = company_info.get("name")

        if rec_name and actual_name and rec_name.lower() != actual_name.lower():
            errors.append(
                f"Company name mismatch: {rec_name} vs {actual_name}"
            )

        return {
            "verified": verified,
            "errors": errors
        }

    def _check_for_hallucinations(self, recommendation: Dict) -> Dict:
        """
        Check for signs of hallucinated content.
        """
        errors = []
        warnings = []

        rationale = recommendation.get("rationale", {})

        # Check for specific prices without sources
        for key, value in rationale.items():
            if isinstance(value, str):
                # Look for specific price predictions
                price_pattern = r'target\s+(?:price\s+)?(?:of\s+)?(?:Rs\.?\s*)?(\d{2,})'
                if re.search(price_pattern, value, re.IGNORECASE):
                    if not recommendation.get("sources_cited"):
                        warnings.append(
                            f"Price target mentioned in {key} without cited sources"
                        )

                # Look for specific dates
                date_pattern = r'on\s+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'
                if re.search(date_pattern, value):
                    warnings.append(
                        f"Specific date mentioned in {key} - verify accuracy"
                    )

                # Look for "approximately" or estimates
                estimate_words = ["approximately", "around", "roughly", "about", "estimated"]
                for word in estimate_words:
                    if word in value.lower():
                        warnings.append(
                            f"Estimated/approximate language used in {key} - "
                            "ensure actual figures are used"
                        )

        # Check for target price without methodology
        if recommendation.get("target_price") is not None:
            if not recommendation.get("sources_cited"):
                errors.append(
                    "Target price provided without cited sources or methodology"
                )

        return {
            "errors": errors,
            "warnings": warnings
        }

    def generate_review_report(self, result: ReviewResult) -> str:
        """Generate a human-readable review report."""
        report = f"""
{'='*60}
REVIEW REPORT
{'='*60}

Overall Status: {'APPROVED' if result.is_valid else 'REJECTED'}

{'─'*60}
ERRORS ({len(result.errors)})
{'─'*60}
"""
        if result.errors:
            for i, error in enumerate(result.errors, 1):
                report += f"{i}. {error}\n"
        else:
            report += "No errors found.\n"

        report += f"""
{'─'*60}
WARNINGS ({len(result.warnings)})
{'─'*60}
"""
        if result.warnings:
            for i, warning in enumerate(result.warnings, 1):
                report += f"{i}. {warning}\n"
        else:
            report += "No warnings.\n"

        report += f"""
{'─'*60}
VERIFIED CLAIMS ({len(result.verified_claims)})
{'─'*60}
"""
        if result.verified_claims:
            for i, claim in enumerate(result.verified_claims[:10], 1):
                report += f"{i}. {claim}\n"
            if len(result.verified_claims) > 10:
                report += f"... and {len(result.verified_claims) - 10} more\n"
        else:
            report += "No claims verified.\n"

        report += f"""
{'='*60}
"""
        return report

    def requires_regeneration(self, result: ReviewResult) -> bool:
        """Check if analysis needs to be regenerated."""
        # Regenerate if there are errors
        if result.errors:
            return True

        # Also regenerate if there are too many warnings
        if len(result.warnings) > 5:
            return True

        return False

    def get_regeneration_guidance(self, result: ReviewResult) -> str:
        """Get guidance for regenerating the analysis."""
        guidance_parts = ["Please regenerate the analysis addressing the following issues:\n"]

        for i, error in enumerate(result.errors, 1):
            guidance_parts.append(f"{i}. ERROR: {error}")

        for i, warning in enumerate(result.warnings[:3], len(result.errors) + 1):
            guidance_parts.append(f"{i}. WARNING: {warning}")

        guidance_parts.append("\nEnsure all claims are supported by cited sources.")
        guidance_parts.append("Use only calculated technical indicators, not estimates.")
        guidance_parts.append("Verify all numerical values against source data.")

        return "\n".join(guidance_parts)

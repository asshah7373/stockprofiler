"""
Fundamental Analysis Module
Analyzes company financial metrics from various sources.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValuationRating(Enum):
    DEEPLY_UNDERVALUED = "DEEPLY_UNDERVALUED"
    UNDERVALUED = "UNDERVALUED"
    FAIRLY_VALUED = "FAIRLY_VALUED"
    OVERVALUED = "OVERVALUED"
    DEEPLY_OVERVALUED = "DEEPLY_OVERVALUED"


class QualityRating(Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    AVERAGE = "AVERAGE"
    POOR = "POOR"
    VERY_POOR = "VERY_POOR"


@dataclass
class FundamentalScore:
    """Comprehensive fundamental analysis score."""
    ticker: str
    valuation_rating: str
    quality_rating: str
    valuation_score: float  # 0-100
    quality_score: float    # 0-100
    overall_score: float    # 0-100
    metrics: Dict[str, Any]
    analysis: Dict[str, str]
    sources: List[str]


class FundamentalAnalyzer:
    """
    Analyzes fundamental metrics from company data.

    All calculations are based on actual data - no estimation or hallucination.
    """

    # Sector-specific P/E benchmarks (approximate averages for Indian markets)
    SECTOR_PE_BENCHMARKS = {
        "Technology": 25,
        "Financial Services": 15,
        "Healthcare": 30,
        "Consumer Cyclical": 25,
        "Consumer Defensive": 35,
        "Industrials": 20,
        "Basic Materials": 12,
        "Energy": 10,
        "Utilities": 12,
        "Real Estate": 15,
        "Communication Services": 18,
        "default": 20
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(
        self,
        company_info: Dict[str, Any],
        rag_context: Optional[Dict] = None
    ) -> FundamentalScore:
        """
        Perform comprehensive fundamental analysis.

        Args:
            company_info: Company data from PerceiverAgent
            rag_context: Retrieved document context

        Returns:
            FundamentalScore with detailed analysis
        """
        ticker = company_info.get("ticker", "UNKNOWN")
        sector = company_info.get("sector", "default")

        # Extract and validate metrics
        metrics = self._extract_metrics(company_info)

        # Calculate scores
        valuation_result = self._analyze_valuation(metrics, sector)
        quality_result = self._analyze_quality(metrics)
        growth_result = self._analyze_growth(metrics)
        financial_health = self._analyze_financial_health(metrics)

        # Combine into overall scores
        valuation_score = valuation_result.get("score", 50)
        quality_score = (
            quality_result.get("score", 50) * 0.4 +
            growth_result.get("score", 50) * 0.3 +
            financial_health.get("score", 50) * 0.3
        )

        overall_score = (valuation_score * 0.4 + quality_score * 0.6)

        # Determine ratings
        valuation_rating = self._score_to_valuation_rating(valuation_score)
        quality_rating = self._score_to_quality_rating(quality_score)

        # Build analysis commentary
        analysis = {
            "valuation": valuation_result.get("analysis", ""),
            "profitability": quality_result.get("analysis", ""),
            "growth": growth_result.get("analysis", ""),
            "financial_health": financial_health.get("analysis", "")
        }

        # Collect sources
        sources = []
        if rag_context:
            sources = [s.get("doc_id", "") for s in rag_context.get("sources", [])]

        return FundamentalScore(
            ticker=ticker,
            valuation_rating=valuation_rating.value,
            quality_rating=quality_rating.value,
            valuation_score=round(valuation_score, 1),
            quality_score=round(quality_score, 1),
            overall_score=round(overall_score, 1),
            metrics=metrics,
            analysis=analysis,
            sources=sources
        )

    def _extract_metrics(self, company_info: Dict) -> Dict[str, Any]:
        """Extract and organize key financial metrics."""
        return {
            # Valuation
            "pe_ratio": company_info.get("pe_ratio"),
            "forward_pe": company_info.get("forward_pe"),
            "pb_ratio": company_info.get("price_to_book"),
            "ps_ratio": company_info.get("price_to_sales"),
            "peg_ratio": company_info.get("peg_ratio"),
            "ev_to_ebitda": company_info.get("enterprise_to_ebitda"),
            "ev_to_revenue": company_info.get("enterprise_to_revenue"),

            # Profitability
            "profit_margin": company_info.get("profit_margin"),
            "operating_margin": company_info.get("operating_margin"),
            "gross_margin": company_info.get("gross_margin"),
            "roe": company_info.get("return_on_equity"),
            "roa": company_info.get("return_on_assets"),

            # Per share
            "eps": company_info.get("eps"),
            "forward_eps": company_info.get("forward_eps"),
            "book_value": company_info.get("book_value"),
            "revenue_per_share": company_info.get("revenue_per_share"),

            # Growth
            "revenue_growth": company_info.get("revenue_growth"),

            # Financial health
            "debt_to_equity": company_info.get("debt_to_equity"),
            "current_ratio": company_info.get("current_ratio"),
            "quick_ratio": company_info.get("quick_ratio"),
            "total_debt": company_info.get("total_debt"),
            "total_cash": company_info.get("total_cash"),

            # Dividends
            "dividend_yield": company_info.get("dividend_yield"),
            "payout_ratio": company_info.get("payout_ratio"),

            # Size
            "market_cap": company_info.get("market_cap"),
            "market_cap_category": company_info.get("market_cap_category"),

            # Misc
            "beta": company_info.get("beta"),
            "sector": company_info.get("sector"),
            "industry": company_info.get("industry")
        }

    def _analyze_valuation(self, metrics: Dict, sector: str) -> Dict:
        """Analyze company valuation."""
        score = 50  # Start at neutral
        analysis_parts = []

        # Get sector benchmark
        benchmark_pe = self.SECTOR_PE_BENCHMARKS.get(sector, self.SECTOR_PE_BENCHMARKS["default"])

        # P/E Analysis
        pe = metrics.get("pe_ratio")
        if pe is not None:
            if pe < 0:
                score -= 20
                analysis_parts.append(f"Negative P/E ({pe:.1f}) indicates losses")
            elif pe < benchmark_pe * 0.6:
                score += 25
                analysis_parts.append(f"P/E of {pe:.1f} is well below sector average of {benchmark_pe}")
            elif pe < benchmark_pe:
                score += 10
                analysis_parts.append(f"P/E of {pe:.1f} is below sector average")
            elif pe > benchmark_pe * 1.5:
                score -= 20
                analysis_parts.append(f"P/E of {pe:.1f} is significantly above sector average")
            elif pe > benchmark_pe:
                score -= 5
                analysis_parts.append(f"P/E of {pe:.1f} is above sector average")

        # P/B Analysis
        pb = metrics.get("pb_ratio")
        if pb is not None:
            if pb < 1:
                score += 15
                analysis_parts.append(f"P/B of {pb:.2f} suggests undervaluation")
            elif pb > 5:
                score -= 10
                analysis_parts.append(f"High P/B of {pb:.2f}")

        # PEG Analysis
        peg = metrics.get("peg_ratio")
        if peg is not None:
            if 0 < peg < 1:
                score += 20
                analysis_parts.append(f"Attractive PEG ratio of {peg:.2f}")
            elif peg > 2:
                score -= 10
                analysis_parts.append(f"High PEG ratio of {peg:.2f}")

        # EV/EBITDA
        ev_ebitda = metrics.get("ev_to_ebitda")
        if ev_ebitda is not None:
            if ev_ebitda < 8:
                score += 10
                analysis_parts.append(f"Low EV/EBITDA of {ev_ebitda:.1f}")
            elif ev_ebitda > 15:
                score -= 10
                analysis_parts.append(f"High EV/EBITDA of {ev_ebitda:.1f}")

        # Normalize score to 0-100
        score = max(0, min(100, score))

        return {
            "score": score,
            "analysis": ". ".join(analysis_parts) if analysis_parts else "Valuation data not available."
        }

    def _analyze_quality(self, metrics: Dict) -> Dict:
        """Analyze company quality (profitability)."""
        score = 50
        analysis_parts = []

        # Return on Equity
        roe = metrics.get("roe")
        if roe is not None:
            roe_pct = roe * 100 if roe < 1 else roe
            if roe_pct > 20:
                score += 25
                analysis_parts.append(f"Excellent ROE of {roe_pct:.1f}%")
            elif roe_pct > 15:
                score += 15
                analysis_parts.append(f"Good ROE of {roe_pct:.1f}%")
            elif roe_pct > 10:
                score += 5
                analysis_parts.append(f"Average ROE of {roe_pct:.1f}%")
            elif roe_pct < 5:
                score -= 15
                analysis_parts.append(f"Low ROE of {roe_pct:.1f}%")

        # Profit Margin
        margin = metrics.get("profit_margin")
        if margin is not None:
            margin_pct = margin * 100 if margin < 1 else margin
            if margin_pct > 20:
                score += 20
                analysis_parts.append(f"Strong profit margin of {margin_pct:.1f}%")
            elif margin_pct > 10:
                score += 10
                analysis_parts.append(f"Good profit margin of {margin_pct:.1f}%")
            elif margin_pct < 5:
                score -= 15
                analysis_parts.append(f"Low profit margin of {margin_pct:.1f}%")

        # Operating Margin
        op_margin = metrics.get("operating_margin")
        if op_margin is not None:
            op_pct = op_margin * 100 if op_margin < 1 else op_margin
            if op_pct > 25:
                score += 10
                analysis_parts.append(f"High operating margin of {op_pct:.1f}%")
            elif op_pct < 10:
                score -= 5
                analysis_parts.append(f"Low operating margin of {op_pct:.1f}%")

        # Normalize
        score = max(0, min(100, score))

        return {
            "score": score,
            "analysis": ". ".join(analysis_parts) if analysis_parts else "Quality data not available."
        }

    def _analyze_growth(self, metrics: Dict) -> Dict:
        """Analyze company growth metrics."""
        score = 50
        analysis_parts = []

        # Revenue Growth
        rev_growth = metrics.get("revenue_growth")
        if rev_growth is not None:
            growth_pct = rev_growth * 100 if abs(rev_growth) < 5 else rev_growth
            if growth_pct > 20:
                score += 30
                analysis_parts.append(f"Strong revenue growth of {growth_pct:.1f}%")
            elif growth_pct > 10:
                score += 15
                analysis_parts.append(f"Good revenue growth of {growth_pct:.1f}%")
            elif growth_pct > 0:
                score += 5
                analysis_parts.append(f"Modest revenue growth of {growth_pct:.1f}%")
            elif growth_pct < -10:
                score -= 25
                analysis_parts.append(f"Revenue declining at {growth_pct:.1f}%")
            else:
                score -= 10
                analysis_parts.append(f"Slight revenue decline of {growth_pct:.1f}%")

        # EPS Growth (compare current to forward)
        eps = metrics.get("eps")
        forward_eps = metrics.get("forward_eps")
        if eps is not None and forward_eps is not None and eps > 0:
            eps_growth = (forward_eps - eps) / eps * 100
            if eps_growth > 15:
                score += 15
                analysis_parts.append(f"EPS expected to grow {eps_growth:.1f}%")
            elif eps_growth < -10:
                score -= 15
                analysis_parts.append(f"EPS expected to decline {eps_growth:.1f}%")

        # Normalize
        score = max(0, min(100, score))

        return {
            "score": score,
            "analysis": ". ".join(analysis_parts) if analysis_parts else "Growth data not available."
        }

    def _analyze_financial_health(self, metrics: Dict) -> Dict:
        """Analyze company financial health."""
        score = 50
        analysis_parts = []

        # Debt to Equity
        de = metrics.get("debt_to_equity")
        if de is not None:
            if de < 30:
                score += 20
                analysis_parts.append(f"Low debt with D/E of {de:.1f}%")
            elif de < 80:
                score += 5
                analysis_parts.append(f"Moderate debt with D/E of {de:.1f}%")
            elif de > 150:
                score -= 25
                analysis_parts.append(f"High debt with D/E of {de:.1f}%")
            else:
                score -= 10
                analysis_parts.append(f"Elevated debt with D/E of {de:.1f}%")

        # Current Ratio
        current_ratio = metrics.get("current_ratio")
        if current_ratio is not None:
            if current_ratio > 2:
                score += 15
                analysis_parts.append(f"Strong current ratio of {current_ratio:.2f}")
            elif current_ratio > 1.5:
                score += 10
                analysis_parts.append(f"Good current ratio of {current_ratio:.2f}")
            elif current_ratio < 1:
                score -= 20
                analysis_parts.append(f"Concerning current ratio of {current_ratio:.2f}")

        # Cash position
        total_cash = metrics.get("total_cash")
        total_debt = metrics.get("total_debt")
        if total_cash is not None and total_debt is not None and total_debt > 0:
            cash_to_debt = total_cash / total_debt
            if cash_to_debt > 1:
                score += 15
                analysis_parts.append("Cash exceeds debt")
            elif cash_to_debt < 0.1:
                score -= 10
                analysis_parts.append("Limited cash relative to debt")

        # Normalize
        score = max(0, min(100, score))

        return {
            "score": score,
            "analysis": ". ".join(analysis_parts) if analysis_parts else "Financial health data not available."
        }

    def _score_to_valuation_rating(self, score: float) -> ValuationRating:
        """Convert valuation score to rating."""
        if score >= 80:
            return ValuationRating.DEEPLY_UNDERVALUED
        elif score >= 65:
            return ValuationRating.UNDERVALUED
        elif score >= 45:
            return ValuationRating.FAIRLY_VALUED
        elif score >= 30:
            return ValuationRating.OVERVALUED
        else:
            return ValuationRating.DEEPLY_OVERVALUED

    def _score_to_quality_rating(self, score: float) -> QualityRating:
        """Convert quality score to rating."""
        if score >= 80:
            return QualityRating.EXCELLENT
        elif score >= 65:
            return QualityRating.GOOD
        elif score >= 45:
            return QualityRating.AVERAGE
        elif score >= 30:
            return QualityRating.POOR
        else:
            return QualityRating.VERY_POOR

    def get_sector_comparison(
        self,
        ticker_metrics: Dict,
        sector_peers: List[Dict]
    ) -> Dict:
        """
        Compare a stock's fundamentals against sector peers.

        Args:
            ticker_metrics: Target company metrics
            sector_peers: List of peer company metrics

        Returns:
            Comparison analysis
        """
        if not sector_peers:
            return {"error": "No sector peers available for comparison"}

        comparison = {}

        # Compare key metrics
        key_metrics = ["pe_ratio", "profit_margin", "roe", "debt_to_equity", "revenue_growth"]

        for metric in key_metrics:
            ticker_value = ticker_metrics.get(metric)
            if ticker_value is None:
                continue

            peer_values = [p.get(metric) for p in sector_peers if p.get(metric) is not None]
            if not peer_values:
                continue

            avg_peer = sum(peer_values) / len(peer_values)
            percentile = sum(1 for v in peer_values if v < ticker_value) / len(peer_values) * 100

            comparison[metric] = {
                "value": ticker_value,
                "peer_average": avg_peer,
                "percentile": percentile,
                "vs_avg": "above" if ticker_value > avg_peer else "below"
            }

        return comparison

    def format_report(self, score: FundamentalScore) -> str:
        """Format fundamental analysis as a readable report."""
        report = f"""
{'='*60}
FUNDAMENTAL ANALYSIS REPORT
{'='*60}

Ticker: {score.ticker}

RATINGS
{'─'*60}
Valuation: {score.valuation_rating} (Score: {score.valuation_score}/100)
Quality: {score.quality_rating} (Score: {score.quality_score}/100)
Overall: {score.overall_score}/100

KEY METRICS
{'─'*60}
"""

        # Add key metrics
        key_display = [
            ("P/E Ratio", score.metrics.get("pe_ratio")),
            ("P/B Ratio", score.metrics.get("pb_ratio")),
            ("ROE", score.metrics.get("roe")),
            ("Profit Margin", score.metrics.get("profit_margin")),
            ("Revenue Growth", score.metrics.get("revenue_growth")),
            ("Debt/Equity", score.metrics.get("debt_to_equity")),
            ("Current Ratio", score.metrics.get("current_ratio")),
            ("Dividend Yield", score.metrics.get("dividend_yield")),
        ]

        for name, value in key_display:
            if value is not None:
                if isinstance(value, float):
                    report += f"{name}: {value:.2f}\n"
                else:
                    report += f"{name}: {value}\n"

        report += f"""
ANALYSIS
{'─'*60}
"""
        for category, analysis in score.analysis.items():
            if analysis:
                report += f"\n{category.replace('_', ' ').title()}:\n{analysis}\n"

        report += f"""
{'='*60}
"""
        return report

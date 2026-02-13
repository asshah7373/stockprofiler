"""
Constraint Applier
Filters recommendations based on risk profile.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

from .questionnaire import RiskProfile, RiskCategory

logger = logging.getLogger(__name__)


@dataclass
class SuitabilityResult:
    """Result of suitability check."""
    is_suitable: bool
    violations: List[str]
    warnings: List[str]
    details: Dict[str, Any]


class ConstraintApplier:
    """
    Applies risk profile constraints to filter recommendations.

    Ensures all recommendations comply with user's risk tolerance
    as per SEBI Investment Adviser regulations.
    """

    # Market cap thresholds (in Crores)
    MARKET_CAP_HIERARCHY = {
        "LARGE": 20000,
        "MID": 5000,
        "SMALL": 0
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def filter_recommendations(
        self,
        recommendations: List[Dict],
        profile: RiskProfile
    ) -> List[Dict]:
        """
        Filter recommendations by profile constraints.

        Removes any recommendation that violates constraints.

        Args:
            recommendations: List of stock recommendations
            profile: User's risk profile

        Returns:
            Filtered list of suitable recommendations
        """
        filtered = []
        constraints = profile.constraints

        for rec in recommendations:
            result = self.check_suitability(rec, profile)

            if result.is_suitable:
                # Add suitability info to recommendation
                rec["suitability"] = {
                    "suitable": True,
                    "warnings": result.warnings,
                    "profile_used": profile.profile_id
                }
                filtered.append(rec)
            else:
                self.logger.info(
                    f"Filtered out {rec.get('ticker')}: {result.violations}"
                )

        return filtered

    def check_suitability(
        self,
        stock_data: Dict,
        profile: RiskProfile
    ) -> SuitabilityResult:
        """
        Check if a stock is suitable for profile.

        Args:
            stock_data: Stock data including fundamentals and technicals
            profile: User's risk profile

        Returns:
            SuitabilityResult with violations and warnings
        """
        violations = []
        warnings = []
        details = {}
        constraints = profile.constraints

        # Check beta
        beta_result = self._check_beta(
            stock_data.get("beta"),
            constraints.get("max_beta")
        )
        if beta_result:
            if beta_result["is_violation"]:
                violations.append(beta_result["message"])
            else:
                warnings.append(beta_result["message"])
        details["beta"] = beta_result

        # Check market cap
        cap_result = self._check_market_cap(
            stock_data.get("market_cap_category"),
            constraints.get("min_market_cap")
        )
        if cap_result:
            if cap_result["is_violation"]:
                violations.append(cap_result["message"])
            else:
                warnings.append(cap_result["message"])
        details["market_cap"] = cap_result

        # Check sector
        sector_result = self._check_sector(
            stock_data.get("sector"),
            constraints.get("allowed_sectors")
        )
        if sector_result:
            if sector_result["is_violation"]:
                violations.append(sector_result["message"])
        details["sector"] = sector_result

        # Check excluded instruments
        instrument_result = self._check_excluded_instruments(
            stock_data.get("instrument_type"),
            constraints.get("excluded_instruments", [])
        )
        if instrument_result:
            violations.append(instrument_result["message"])
        details["instrument"] = instrument_result

        # Check dividend preference
        if constraints.get("prefer_dividend"):
            div_result = self._check_dividend(
                stock_data.get("dividend_yield"),
                constraints.get("min_dividend_yield", 0)
            )
            if div_result:
                warnings.append(div_result["message"])
            details["dividend"] = div_result

        # Check debt/leverage
        leverage_result = self._check_leverage(
            stock_data.get("debt_to_equity"),
            profile.risk_category
        )
        if leverage_result:
            if leverage_result["is_violation"]:
                violations.append(leverage_result["message"])
            else:
                warnings.append(leverage_result["message"])
        details["leverage"] = leverage_result

        return SuitabilityResult(
            is_suitable=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            details=details
        )

    def _check_beta(
        self,
        beta: Optional[float],
        max_beta: Optional[float]
    ) -> Optional[Dict]:
        """Check beta constraint."""
        if beta is None or max_beta is None:
            return None

        if beta > max_beta:
            return {
                "is_violation": True,
                "message": f"Beta ({beta:.2f}) exceeds maximum allowed ({max_beta})",
                "stock_value": beta,
                "constraint": max_beta
            }
        elif beta > max_beta * 0.9:
            return {
                "is_violation": False,
                "message": f"Beta ({beta:.2f}) is close to maximum ({max_beta})",
                "stock_value": beta,
                "constraint": max_beta
            }
        return None

    def _check_market_cap(
        self,
        stock_cap_category: Optional[str],
        min_cap_category: Optional[str]
    ) -> Optional[Dict]:
        """Check market cap constraint."""
        if stock_cap_category is None or min_cap_category is None:
            return None

        cap_order = ["SMALL", "MID", "LARGE"]

        stock_index = cap_order.index(stock_cap_category) if stock_cap_category in cap_order else -1
        min_index = cap_order.index(min_cap_category) if min_cap_category in cap_order else -1

        if stock_index < min_index:
            return {
                "is_violation": True,
                "message": f"Market cap ({stock_cap_category}) below minimum ({min_cap_category})",
                "stock_value": stock_cap_category,
                "constraint": min_cap_category
            }
        return None

    def _check_sector(
        self,
        sector: Optional[str],
        allowed_sectors: Any
    ) -> Optional[Dict]:
        """Check sector constraint."""
        if sector is None or allowed_sectors is None:
            return None

        if allowed_sectors == "ALL":
            return None

        if isinstance(allowed_sectors, list) and sector not in allowed_sectors:
            return {
                "is_violation": True,
                "message": f"Sector ({sector}) not in allowed list",
                "stock_value": sector,
                "constraint": allowed_sectors
            }
        return None

    def _check_excluded_instruments(
        self,
        instrument_type: Optional[str],
        excluded: List[str]
    ) -> Optional[Dict]:
        """Check excluded instruments constraint."""
        if instrument_type is None or not excluded:
            return None

        for exc in excluded:
            if exc.lower() in instrument_type.lower():
                return {
                    "is_violation": True,
                    "message": f"Instrument type ({instrument_type}) is excluded",
                    "stock_value": instrument_type,
                    "constraint": excluded
                }
        return None

    def _check_dividend(
        self,
        dividend_yield: Optional[float],
        min_yield: float
    ) -> Optional[Dict]:
        """Check dividend preference."""
        if dividend_yield is None:
            return {
                "is_violation": False,
                "message": "No dividend data available",
                "stock_value": None,
                "constraint": min_yield
            }

        if dividend_yield < min_yield:
            return {
                "is_violation": False,
                "message": f"Dividend yield ({dividend_yield*100:.2f}%) below preferred minimum ({min_yield*100:.1f}%)",
                "stock_value": dividend_yield,
                "constraint": min_yield
            }
        return None

    def _check_leverage(
        self,
        debt_to_equity: Optional[float],
        risk_category: str
    ) -> Optional[Dict]:
        """Check leverage based on risk category."""
        if debt_to_equity is None:
            return None

        # Define thresholds based on risk category
        thresholds = {
            "CONSERVATIVE": {"warning": 50, "violation": 80},
            "MODERATE": {"warning": 100, "violation": 150},
            "AGGRESSIVE": {"warning": 150, "violation": 250}
        }

        limits = thresholds.get(risk_category, thresholds["MODERATE"])

        if debt_to_equity > limits["violation"]:
            return {
                "is_violation": True,
                "message": f"Debt/Equity ({debt_to_equity:.0f}%) too high for {risk_category} profile",
                "stock_value": debt_to_equity,
                "constraint": limits["violation"]
            }
        elif debt_to_equity > limits["warning"]:
            return {
                "is_violation": False,
                "message": f"Elevated Debt/Equity ({debt_to_equity:.0f}%)",
                "stock_value": debt_to_equity,
                "constraint": limits["warning"]
            }
        return None

    def get_allocation_limit(
        self,
        stock_data: Dict,
        profile: RiskProfile,
        portfolio_size: float
    ) -> float:
        """
        Calculate maximum allocation for a stock.

        Args:
            stock_data: Stock fundamentals
            profile: Risk profile
            portfolio_size: Total portfolio value

        Returns:
            Maximum recommended allocation in absolute terms
        """
        max_allocation_pct = profile.constraints.get("max_single_stock_allocation", 0.10)

        # Reduce allocation for riskier stocks
        beta = stock_data.get("beta", 1.0)
        if beta > 1.5:
            max_allocation_pct *= 0.7
        elif beta > 1.2:
            max_allocation_pct *= 0.85

        # Reduce for small caps
        if stock_data.get("market_cap_category") == "SMALL":
            max_allocation_pct *= 0.8

        # Reduce for high leverage
        de = stock_data.get("debt_to_equity")
        if de and de > 100:
            max_allocation_pct *= 0.9

        return portfolio_size * max_allocation_pct

    def generate_suitability_report(
        self,
        result: SuitabilityResult,
        stock_ticker: str,
        profile: RiskProfile
    ) -> str:
        """Generate a suitability report."""
        report = f"""
{'='*60}
SUITABILITY ASSESSMENT
{'='*60}

Stock: {stock_ticker}
Risk Profile: {profile.risk_category}
Profile ID: {profile.profile_id}

RESULT: {'SUITABLE' if result.is_suitable else 'NOT SUITABLE'}

"""

        if result.violations:
            report += "VIOLATIONS:\n"
            for v in result.violations:
                report += f"  ❌ {v}\n"
            report += "\n"

        if result.warnings:
            report += "WARNINGS:\n"
            for w in result.warnings:
                report += f"  ⚠️ {w}\n"
            report += "\n"

        if not result.violations and not result.warnings:
            report += "✅ All constraints satisfied\n"

        return report

    def recommend_profile_update(self, profile: RiskProfile) -> Optional[str]:
        """
        Check if profile needs updating.

        Returns recommendation message if update needed.
        """
        from datetime import datetime

        created = datetime.fromisoformat(profile.created_at)
        expiry = datetime.fromisoformat(profile.expiry_date)
        now = datetime.now()

        # Check if expired
        if now > expiry:
            return "Your risk profile has expired. Please complete a new assessment."

        # Check if close to expiry (within 30 days)
        days_to_expiry = (expiry - now).days
        if days_to_expiry < 30:
            return f"Your risk profile expires in {days_to_expiry} days. Consider updating it soon."

        # Check if very old (over 6 months)
        days_old = (now - created).days
        if days_old > 180:
            return "Your risk profile is over 6 months old. Consider reviewing it to ensure it reflects your current situation."

        return None

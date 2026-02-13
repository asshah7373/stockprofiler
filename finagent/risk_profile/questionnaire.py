"""
Risk Profiling Questionnaire
SEBI-mandated investor profiling.
"""

from typing import Dict, Optional, List, Any
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
from datetime import datetime
import sqlite3
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class RiskCategory(Enum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"


@dataclass
class RiskProfile:
    """User's risk profile based on SEBI questionnaire."""
    profile_id: str
    created_at: str
    risk_category: str
    risk_score: int
    constraints: Dict
    responses: Dict
    expiry_date: str  # Profiles should be refreshed periodically

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class RiskProfiler:
    """
    Conducts SEBI-mandated risk profiling.

    Questions based on:
    - SEBI (Investment Advisers) Regulations, 2013
    - AMFI Risk Profiling Guidelines

    Note: Risk profiling is MANDATORY before any recommendation
    """

    QUESTIONS = [
        {
            "id": "age",
            "question": "What is your age bracket?",
            "options": [
                ("18-30", 3),
                ("31-45", 2),
                ("46-60", 1),
                ("60+", 0)
            ],
            "weight": 1.0,
            "category": "demographics"
        },
        {
            "id": "income",
            "question": "What is your annual income?",
            "options": [
                ("Below 5 Lakhs", 0),
                ("5-15 Lakhs", 1),
                ("15-50 Lakhs", 2),
                ("Above 50 Lakhs", 3)
            ],
            "weight": 1.0,
            "category": "financial"
        },
        {
            "id": "net_worth",
            "question": "What is your approximate net worth (excluding primary residence)?",
            "options": [
                ("Below 10 Lakhs", 0),
                ("10-50 Lakhs", 1),
                ("50 Lakhs - 2 Crores", 2),
                ("Above 2 Crores", 3)
            ],
            "weight": 1.0,
            "category": "financial"
        },
        {
            "id": "experience",
            "question": "How many years of investment experience do you have?",
            "options": [
                ("None / New to investing", 0),
                ("1-3 years", 1),
                ("3-7 years", 2),
                ("More than 7 years", 3)
            ],
            "weight": 1.0,
            "category": "experience"
        },
        {
            "id": "knowledge",
            "question": "How would you rate your knowledge of equity markets?",
            "options": [
                ("Basic - I know stocks go up and down", 0),
                ("Intermediate - I understand fundamentals and technicals", 1),
                ("Advanced - I actively trade and understand derivatives", 2),
                ("Expert - I have professional experience in finance", 3)
            ],
            "weight": 0.5,
            "category": "experience"
        },
        {
            "id": "horizon",
            "question": "What is your investment time horizon?",
            "options": [
                ("Less than 1 year", 0),
                ("1-3 years", 1),
                ("3-5 years", 2),
                ("More than 5 years", 3)
            ],
            "weight": 1.5,
            "category": "goals"
        },
        {
            "id": "risk_scenario",
            "question": "If your portfolio drops 20% in a month, you would:",
            "options": [
                ("Sell everything immediately to prevent further losses", 0),
                ("Sell some holdings to reduce risk exposure", 1),
                ("Hold and wait for recovery", 2),
                ("Buy more at lower prices", 3)
            ],
            "weight": 2.0,
            "category": "behavior"
        },
        {
            "id": "capital",
            "question": "What portion of your total savings is this investment?",
            "options": [
                ("More than 50%", 0),
                ("25-50%", 1),
                ("10-25%", 2),
                ("Less than 10%", 3)
            ],
            "weight": 1.5,
            "category": "financial"
        },
        {
            "id": "goal",
            "question": "What is your primary investment goal?",
            "options": [
                ("Capital preservation - safety is most important", 0),
                ("Regular income - stable returns with dividends", 1),
                ("Balanced growth - moderate growth with some safety", 2),
                ("Aggressive growth - maximum returns, willing to take risks", 3)
            ],
            "weight": 1.5,
            "category": "goals"
        },
        {
            "id": "loss_tolerance",
            "question": "What is the maximum loss you can tolerate in a year?",
            "options": [
                ("0-5% - I cannot afford to lose money", 0),
                ("5-15% - Some loss is acceptable", 1),
                ("15-30% - I can handle significant volatility", 2),
                ("More than 30% - High risk for high rewards", 3)
            ],
            "weight": 2.0,
            "category": "behavior"
        }
    ]

    # Constraints for each risk category
    CATEGORY_CONSTRAINTS = {
        RiskCategory.CONSERVATIVE: {
            "max_beta": 0.8,
            "min_market_cap": "LARGE",
            "allowed_sectors": ["FMCG", "IT", "Pharma", "Banking", "Utilities"],
            "excluded_instruments": ["F&O", "Penny Stocks", "SME", "IPO"],
            "max_single_stock_allocation": 0.05,
            "prefer_dividend": True,
            "min_credit_rating": "AA",
            "max_portfolio_turnover": 0.25,
            "min_dividend_yield": 0.02
        },
        RiskCategory.MODERATE: {
            "max_beta": 1.2,
            "min_market_cap": "MID",
            "allowed_sectors": "ALL",
            "excluded_instruments": ["Penny Stocks", "Unlisted Shares"],
            "max_single_stock_allocation": 0.10,
            "prefer_dividend": False,
            "min_credit_rating": "A",
            "max_portfolio_turnover": 0.50,
            "min_dividend_yield": None
        },
        RiskCategory.AGGRESSIVE: {
            "max_beta": 2.0,
            "min_market_cap": "SMALL",
            "allowed_sectors": "ALL",
            "excluded_instruments": [],
            "max_single_stock_allocation": 0.15,
            "prefer_dividend": False,
            "min_credit_rating": "BBB",
            "max_portfolio_turnover": 1.0,
            "min_dividend_yield": None
        }
    }

    def __init__(self, db_path: str = "data/cache/risk_profiles.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._init_db()

    def _init_db(self):
        """Initialize database for storing risk profiles."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_profiles (
                profile_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                risk_category TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                constraints_json TEXT NOT NULL,
                responses_json TEXT NOT NULL,
                expiry_date TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def get_questions(self) -> List[Dict]:
        """Get the list of profiling questions."""
        return [
            {
                "id": q["id"],
                "question": q["question"],
                "options": [opt[0] for opt in q["options"]],
                "category": q["category"]
            }
            for q in self.QUESTIONS
        ]

    def conduct_profiling(self, responses: Dict[str, int]) -> RiskProfile:
        """
        Calculate risk profile from responses.

        Args:
            responses: {question_id: selected_option_index}

        Returns:
            RiskProfile with category and constraints
        """
        # Validate responses
        for q in self.QUESTIONS:
            if q["id"] not in responses:
                raise ValueError(f"Missing response for question: {q['id']}")
            if responses[q["id"]] < 0 or responses[q["id"]] >= len(q["options"]):
                raise ValueError(f"Invalid option for question: {q['id']}")

        # Calculate weighted score
        score = self._calculate_score(responses)

        # Determine category
        category = self._determine_category(score)

        # Generate constraints
        constraints = self._generate_constraints(category)

        # Create profile
        profile = RiskProfile(
            profile_id=str(uuid.uuid4()),
            created_at=datetime.now().isoformat(),
            risk_category=category.value,
            risk_score=score,
            constraints=constraints,
            responses=responses,
            expiry_date=self._calculate_expiry()
        )

        # Save to database
        self._save_profile(profile)

        return profile

    def _calculate_score(self, responses: Dict[str, int]) -> int:
        """Calculate total weighted risk score."""
        total_score = 0
        total_weight = 0

        for q in self.QUESTIONS:
            option_index = responses.get(q["id"], 0)
            option_score = q["options"][option_index][1]
            weight = q.get("weight", 1.0)

            total_score += option_score * weight
            total_weight += weight

        # Normalize to 0-21 range (as per original specification)
        normalized_score = int((total_score / total_weight) * 7)
        return min(21, max(0, normalized_score))

    def _determine_category(self, score: int) -> RiskCategory:
        """
        Map score to risk category.

        0-7: Conservative
        8-14: Moderate
        15-21: Aggressive
        """
        if score <= 7:
            return RiskCategory.CONSERVATIVE
        elif score <= 14:
            return RiskCategory.MODERATE
        else:
            return RiskCategory.AGGRESSIVE

    def _generate_constraints(self, category: RiskCategory) -> Dict:
        """Generate investment constraints for category."""
        return self.CATEGORY_CONSTRAINTS[category].copy()

    def _calculate_expiry(self, months: int = 12) -> str:
        """Calculate profile expiry date (1 year by default)."""
        from datetime import timedelta
        expiry = datetime.now() + timedelta(days=months * 30)
        return expiry.isoformat()

    def _save_profile(self, profile: RiskProfile):
        """Save profile to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO risk_profiles
            (profile_id, created_at, risk_category, risk_score,
             constraints_json, responses_json, expiry_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            profile.profile_id,
            profile.created_at,
            profile.risk_category,
            profile.risk_score,
            json.dumps(profile.constraints),
            json.dumps(profile.responses),
            profile.expiry_date
        ))

        conn.commit()
        conn.close()

    def get_profile(self, profile_id: str) -> Optional[RiskProfile]:
        """Retrieve a risk profile by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM risk_profiles WHERE profile_id = ?",
            (profile_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return RiskProfile(
                profile_id=row[0],
                created_at=row[1],
                risk_category=row[2],
                risk_score=row[3],
                constraints=json.loads(row[4]),
                responses=json.loads(row[5]),
                expiry_date=row[6]
            )
        return None

    def get_latest_profile(self) -> Optional[RiskProfile]:
        """Get the most recent valid profile."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM risk_profiles
            WHERE expiry_date > ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (datetime.now().isoformat(),))

        row = cursor.fetchone()
        conn.close()

        if row:
            return RiskProfile(
                profile_id=row[0],
                created_at=row[1],
                risk_category=row[2],
                risk_score=row[3],
                constraints=json.loads(row[4]),
                responses=json.loads(row[5]),
                expiry_date=row[6]
            )
        return None

    def is_profile_expired(self, profile: RiskProfile) -> bool:
        """Check if a profile has expired."""
        expiry = datetime.fromisoformat(profile.expiry_date)
        return datetime.now() > expiry

    def get_profile_summary(self, profile: RiskProfile) -> str:
        """Generate a human-readable profile summary."""
        constraints = profile.constraints

        summary = f"""
{'='*60}
RISK PROFILE SUMMARY
{'='*60}

Profile ID: {profile.profile_id}
Created: {profile.created_at[:10]}
Expires: {profile.expiry_date[:10]}

Risk Category: {profile.risk_category}
Risk Score: {profile.risk_score}/21

{'─'*60}
INVESTMENT CONSTRAINTS
{'─'*60}
Maximum Beta: {constraints.get('max_beta', 'N/A')}
Minimum Market Cap: {constraints.get('min_market_cap', 'N/A')}
Allowed Sectors: {constraints.get('allowed_sectors', 'ALL')}
Excluded Instruments: {', '.join(constraints.get('excluded_instruments', [])) or 'None'}
Max Single Stock Allocation: {constraints.get('max_single_stock_allocation', 0) * 100:.0f}%
Dividend Preference: {'Yes' if constraints.get('prefer_dividend') else 'No'}

{'─'*60}
RECOMMENDATIONS
{'─'*60}
"""

        if profile.risk_category == "CONSERVATIVE":
            summary += """
- Focus on large-cap, blue-chip stocks
- Prefer dividend-paying companies
- Avoid high-volatility stocks and derivatives
- Consider debt instruments for stability
"""
        elif profile.risk_category == "MODERATE":
            summary += """
- Balance between large and mid-cap stocks
- Diversify across sectors
- Consider both growth and value stocks
- Limited exposure to small-caps
"""
        else:  # AGGRESSIVE
            summary += """
- Can include small and mid-cap stocks
- Higher allocation to growth stocks
- Can consider IPOs and new listings
- May include sectoral/thematic exposure
"""

        return summary

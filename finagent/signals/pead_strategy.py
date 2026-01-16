"""
PEAD (Post-Earnings Announcement Drift) Strategy

A quantitative strategy based on the well-documented market anomaly where:
- Stocks with positive earnings surprises tend to drift upward for 30-60 days
- Stocks with negative earnings surprises tend to drift downward for 30-60 days

This strategy:
1. Identifies stocks with recent earnings announcements
2. Calculates earnings surprise (YoY growth, beat/miss)
3. Generates signals for stocks in the "drift window"
4. Scores based on surprise magnitude and days since announcement

Research Reference:
- Bernard & Thomas (1989): "Post-Earnings-Announcement Drift"
- Optimal holding period: Entry within 3 days of announcement, hold for 60 days
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class EarningsSurprise(Enum):
    """Earnings surprise classification."""
    STRONG_BEAT = "strong_beat"     # > 20% YoY growth or beat
    BEAT = "beat"                   # 5-20% beat
    INLINE = "inline"               # -5% to +5%
    MISS = "miss"                   # -5% to -20% miss
    STRONG_MISS = "strong_miss"     # > 20% miss


@dataclass
class EarningsEvent:
    """Represents an earnings announcement event."""
    ticker: str
    company_name: str
    announcement_date: datetime
    quarter: str                     # e.g., "Q3FY25"

    # Earnings data
    revenue: Optional[float] = None
    revenue_yoy: Optional[float] = None       # YoY growth %
    net_profit: Optional[float] = None
    profit_yoy: Optional[float] = None        # YoY growth %
    eps: Optional[float] = None
    eps_yoy: Optional[float] = None           # YoY growth %

    # Surprise metrics
    surprise_type: EarningsSurprise = EarningsSurprise.INLINE
    surprise_magnitude: float = 0.0           # % surprise

    # Price reaction
    price_at_announcement: Optional[float] = None
    current_price: Optional[float] = None
    price_change_since: Optional[float] = None  # % change since announcement

    # PEAD metrics
    days_since_announcement: int = 0
    drift_window_remaining: int = 60          # Days left in typical drift window
    pead_score: float = 0.0                   # -100 to +100

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'announcement_date': self.announcement_date.isoformat(),
            'quarter': self.quarter,
            'revenue': self.revenue,
            'revenue_yoy': self.revenue_yoy,
            'net_profit': self.net_profit,
            'profit_yoy': self.profit_yoy,
            'eps': self.eps,
            'eps_yoy': self.eps_yoy,
            'surprise_type': self.surprise_type.value,
            'surprise_magnitude': round(self.surprise_magnitude, 2),
            'price_at_announcement': self.price_at_announcement,
            'current_price': self.current_price,
            'price_change_since': round(self.price_change_since, 2) if self.price_change_since else None,
            'days_since_announcement': self.days_since_announcement,
            'drift_window_remaining': self.drift_window_remaining,
            'pead_score': round(self.pead_score, 1)
        }


@dataclass
class PEADSignal:
    """PEAD trading signal."""
    ticker: str
    company_name: str
    action: str                      # BUY, SELL, HOLD
    confidence: float                # 0-100

    # Event details
    earnings_event: EarningsEvent

    # Trade parameters
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    expected_drift: float = 0.0      # Expected % drift remaining
    holding_period_days: int = 60

    # Analysis
    reasons: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'action': self.action,
            'confidence': round(self.confidence, 1),
            'entry_price': round(self.entry_price, 2) if self.entry_price else None,
            'stop_loss': round(self.stop_loss, 2) if self.stop_loss else None,
            'target_price': round(self.target_price, 2) if self.target_price else None,
            'expected_drift': round(self.expected_drift, 2),
            'holding_period_days': self.holding_period_days,
            'earnings_event': self.earnings_event.to_dict(),
            'reasons': self.reasons,
            'risks': self.risks
        }


class PEADStrategy:
    """
    Post-Earnings Announcement Drift Strategy.

    Identifies trading opportunities based on earnings surprises
    and the documented tendency for prices to drift in the
    direction of the surprise for 30-60 days.
    """

    # PEAD parameters (based on academic research)
    DRIFT_WINDOW_DAYS = 60           # Typical drift window
    OPTIMAL_ENTRY_DAYS = 3           # Best to enter within 3 days
    MIN_SURPRISE_PCT = 5.0           # Minimum surprise to trigger signal
    STRONG_SURPRISE_PCT = 20.0       # Threshold for strong signal

    # Expected drift by surprise magnitude (empirical)
    EXPECTED_DRIFT = {
        EarningsSurprise.STRONG_BEAT: 8.0,    # Expect 8% additional drift
        EarningsSurprise.BEAT: 4.0,           # Expect 4% drift
        EarningsSurprise.INLINE: 0.0,
        EarningsSurprise.MISS: -4.0,
        EarningsSurprise.STRONG_MISS: -8.0,
    }

    def __init__(self, db_path: Optional[str] = None):
        """Initialize PEAD strategy."""
        if db_path:
            self.db_path = Path(db_path)
        else:
            # Try common locations for the circulars database
            possible_paths = [
                Path("data/cache/circulars.db"),
                Path.home() / ".finagent" / "circulars.db",
                Path(__file__).parent.parent / "data" / "cache" / "circulars.db",
            ]
            self.db_path = None
            for p in possible_paths:
                if p.exists():
                    self.db_path = p
                    break
            if self.db_path is None:
                self.db_path = Path("data/cache/circulars.db")  # Default

        self.logger = logging.getLogger(__name__)

    def _get_db_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. "
                "Run 'finagent ingest' first to fetch earnings data."
            )
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def fetch_recent_earnings(
        self,
        days_back: int = 60,
        ticker: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch recent earnings announcements from ingested data.

        Looks for quarterly results in the circulars database.
        """
        conn = self._get_db_connection()
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        # Query for quarterly results
        query = """
            SELECT
                ticker, company_name, filing_date, subject,
                content, doc_type, exchange
            FROM circulars
            WHERE filing_date >= ?
            AND (
                LOWER(subject) LIKE '%quarterly%'
                OR LOWER(subject) LIKE '%results%'
                OR LOWER(subject) LIKE '%quarter%'
                OR LOWER(subject) LIKE '%financial%'
                OR LOWER(doc_type) LIKE '%result%'
            )
        """
        params = [since_date]

        if ticker:
            query += " AND UPPER(ticker) = ?"
            params.append(ticker.upper())

        query += " ORDER BY filing_date DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def _parse_earnings_data(self, filing: Dict) -> Optional[Dict]:
        """
        Parse earnings data from filing content.

        Extracts key metrics like revenue, profit, EPS from announcement text.
        """
        content = filing.get('content', '') or ''
        subject = filing.get('subject', '') or ''

        data = {
            'ticker': filing.get('ticker'),
            'company_name': filing.get('company_name'),
            'announcement_date': filing.get('filing_date'),
            'quarter': self._extract_quarter(subject + ' ' + content),
            'revenue': None,
            'revenue_yoy': None,
            'net_profit': None,
            'profit_yoy': None,
            'eps': None,
            'eps_yoy': None,
        }

        # Try to extract YoY growth percentages from content
        import re

        # Look for YoY growth patterns
        yoy_patterns = [
            r'(?:profit|PAT|net income).*?(?:up|grew|increased|rose)\s*(?:by\s*)?(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:YoY|y-o-y|year.on.year)',
            r'(?:revenue|sales|income).*?(?:up|grew|increased)\s*(?:by\s*)?(\d+(?:\.\d+)?)\s*%',
        ]

        for pattern in yoy_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                growth = float(match.group(1))
                if 'profit' in pattern.lower() or 'PAT' in pattern:
                    data['profit_yoy'] = growth
                else:
                    data['revenue_yoy'] = growth
                break

        # Look for decline patterns
        decline_patterns = [
            r'(?:profit|PAT).*?(?:down|fell|declined|decreased)\s*(?:by\s*)?(\d+(?:\.\d+)?)\s*%',
            r'(?:revenue|sales).*?(?:down|fell|declined)\s*(?:by\s*)?(\d+(?:\.\d+)?)\s*%',
        ]

        for pattern in decline_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                decline = -float(match.group(1))
                if 'profit' in pattern.lower() or 'PAT' in pattern:
                    data['profit_yoy'] = decline
                else:
                    data['revenue_yoy'] = decline
                break

        # Check for "beat" or "miss" keywords
        if re.search(r'beat\s*(?:street\s*)?(?:estimates?|expectations?)', content, re.IGNORECASE):
            if data['profit_yoy'] is None:
                data['profit_yoy'] = 10.0  # Assume moderate beat
        elif re.search(r'miss(?:ed)?\s*(?:street\s*)?(?:estimates?|expectations?)', content, re.IGNORECASE):
            if data['profit_yoy'] is None:
                data['profit_yoy'] = -10.0  # Assume moderate miss

        return data

    def _extract_quarter(self, text: str) -> str:
        """Extract quarter information from text."""
        import re

        # Patterns like Q1FY25, Q3 FY2025, etc.
        patterns = [
            r'Q([1-4])\s*FY\s*(\d{2,4})',
            r'([1-4])Q\s*FY\s*(\d{2,4})',
            r'quarter\s*([1-4])',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                q = match.group(1)
                fy = match.group(2) if len(match.groups()) > 1 else datetime.now().year
                return f"Q{q}FY{fy}"

        return "Unknown"

    def _classify_surprise(self, yoy_growth: Optional[float]) -> Tuple[EarningsSurprise, float]:
        """Classify earnings surprise based on YoY growth."""
        if yoy_growth is None:
            return EarningsSurprise.INLINE, 0.0

        if yoy_growth >= self.STRONG_SURPRISE_PCT:
            return EarningsSurprise.STRONG_BEAT, yoy_growth
        elif yoy_growth >= self.MIN_SURPRISE_PCT:
            return EarningsSurprise.BEAT, yoy_growth
        elif yoy_growth <= -self.STRONG_SURPRISE_PCT:
            return EarningsSurprise.STRONG_MISS, yoy_growth
        elif yoy_growth <= -self.MIN_SURPRISE_PCT:
            return EarningsSurprise.MISS, yoy_growth
        else:
            return EarningsSurprise.INLINE, yoy_growth

    def _calculate_pead_score(
        self,
        surprise_type: EarningsSurprise,
        surprise_magnitude: float,
        days_since: int
    ) -> float:
        """
        Calculate PEAD score based on surprise and timing.

        Score ranges from -100 to +100:
        - Positive = bullish drift expected
        - Negative = bearish drift expected
        - Magnitude reflects confidence
        """
        # Base score from surprise type
        base_scores = {
            EarningsSurprise.STRONG_BEAT: 80,
            EarningsSurprise.BEAT: 50,
            EarningsSurprise.INLINE: 0,
            EarningsSurprise.MISS: -50,
            EarningsSurprise.STRONG_MISS: -80,
        }

        base = base_scores.get(surprise_type, 0)

        # Adjust for magnitude (stronger surprise = higher score)
        magnitude_factor = min(abs(surprise_magnitude) / 30, 1.0)  # Cap at 30%
        base = base * (0.5 + 0.5 * magnitude_factor)

        # Decay factor based on days since announcement
        # Drift is strongest in first 30 days, decays after
        if days_since <= 3:
            timing_factor = 1.0  # Optimal entry window
        elif days_since <= 30:
            timing_factor = 0.8  # Good entry
        elif days_since <= 45:
            timing_factor = 0.5  # Moderate
        elif days_since <= 60:
            timing_factor = 0.3  # Late entry
        else:
            timing_factor = 0.1  # Drift likely complete

        return base * timing_factor

    def analyze_earnings_event(
        self,
        filing: Dict,
        current_price: Optional[float] = None
    ) -> EarningsEvent:
        """Analyze an earnings announcement and create EarningsEvent."""
        data = self._parse_earnings_data(filing)

        # Calculate days since announcement
        try:
            ann_date = datetime.fromisoformat(data['announcement_date'].replace('Z', '+00:00'))
            if ann_date.tzinfo:
                ann_date = ann_date.replace(tzinfo=None)
        except:
            ann_date = datetime.now()

        days_since = (datetime.now() - ann_date).days

        # Use profit YoY as primary surprise metric
        yoy = data.get('profit_yoy') or data.get('revenue_yoy')
        surprise_type, surprise_mag = self._classify_surprise(yoy)

        # Calculate PEAD score
        pead_score = self._calculate_pead_score(surprise_type, surprise_mag or 0, days_since)

        return EarningsEvent(
            ticker=data['ticker'] or 'UNKNOWN',
            company_name=data['company_name'] or 'Unknown Company',
            announcement_date=ann_date,
            quarter=data['quarter'],
            revenue=data.get('revenue'),
            revenue_yoy=data.get('revenue_yoy'),
            net_profit=data.get('net_profit'),
            profit_yoy=data.get('profit_yoy'),
            eps=data.get('eps'),
            eps_yoy=data.get('eps_yoy'),
            surprise_type=surprise_type,
            surprise_magnitude=surprise_mag or 0,
            current_price=current_price,
            days_since_announcement=days_since,
            drift_window_remaining=max(0, self.DRIFT_WINDOW_DAYS - days_since),
            pead_score=pead_score
        )

    def generate_signals(
        self,
        days_back: int = 60,
        min_score: float = 30.0,
        ticker: Optional[str] = None
    ) -> List[PEADSignal]:
        """
        Generate PEAD trading signals from recent earnings.

        Args:
            days_back: Look back period for earnings
            min_score: Minimum PEAD score to generate signal
            ticker: Filter by specific ticker

        Returns:
            List of PEADSignal objects sorted by score
        """
        signals = []

        # Fetch recent earnings
        filings = self.fetch_recent_earnings(days_back=days_back, ticker=ticker)
        self.logger.info(f"Found {len(filings)} earnings announcements")

        # Group by ticker (take most recent per ticker)
        ticker_filings = {}
        for filing in filings:
            t = filing.get('ticker')
            if t and t not in ticker_filings:
                ticker_filings[t] = filing

        # Analyze each ticker
        for ticker_sym, filing in ticker_filings.items():
            try:
                event = self.analyze_earnings_event(filing)

                # Skip if score below threshold
                if abs(event.pead_score) < min_score:
                    continue

                # Skip if drift window expired
                if event.drift_window_remaining <= 0:
                    continue

                # Generate signal
                signal = self._create_signal(event)
                if signal:
                    signals.append(signal)

            except Exception as e:
                self.logger.warning(f"Error analyzing {ticker_sym}: {e}")
                continue

        # Sort by absolute score (strongest signals first)
        signals.sort(key=lambda s: abs(s.earnings_event.pead_score), reverse=True)

        return signals

    def _create_signal(self, event: EarningsEvent) -> Optional[PEADSignal]:
        """Create a PEADSignal from an EarningsEvent."""
        score = event.pead_score

        # Determine action
        if score >= 40:
            action = "BUY"
        elif score >= 20:
            action = "BUY"  # Weak buy
        elif score <= -40:
            action = "SELL"
        elif score <= -20:
            action = "SELL"  # Weak sell
        else:
            return None  # No signal

        # Calculate confidence
        confidence = min(abs(score), 100)

        # Expected drift based on surprise type
        expected_drift = self.EXPECTED_DRIFT.get(event.surprise_type, 0)
        # Adjust for already elapsed time
        time_factor = event.drift_window_remaining / self.DRIFT_WINDOW_DAYS
        expected_drift = expected_drift * time_factor

        # Trade parameters
        current = event.current_price or 100  # Placeholder if no price
        if action == "BUY":
            entry = current
            stop_loss = current * 0.95  # 5% stop
            target = current * (1 + expected_drift / 100)
        else:
            entry = current
            stop_loss = current * 1.05  # 5% stop for short
            target = current * (1 + expected_drift / 100)  # Negative drift

        # Build reasons and risks
        reasons = []
        risks = []

        if event.surprise_type in [EarningsSurprise.STRONG_BEAT, EarningsSurprise.BEAT]:
            reasons.append(f"Earnings surprise: {event.surprise_magnitude:+.1f}% YoY growth")
            reasons.append(f"PEAD drift window: {event.drift_window_remaining} days remaining")
            if event.days_since_announcement <= 3:
                reasons.append("Optimal entry: within 3 days of announcement")
            elif event.days_since_announcement <= 30:
                reasons.append("Good entry timing: early in drift window")
        else:
            reasons.append(f"Earnings miss: {event.surprise_magnitude:.1f}% YoY decline")
            reasons.append("PEAD suggests continued downward drift")

        # Risks
        if event.days_since_announcement > 30:
            risks.append(f"Late entry: {event.days_since_announcement} days since announcement")
        if abs(event.surprise_magnitude) < 10:
            risks.append("Moderate surprise magnitude - weaker signal")
        risks.append("Market conditions may override PEAD effect")
        risks.append("Individual stock factors not considered")

        return PEADSignal(
            ticker=event.ticker,
            company_name=event.company_name,
            action=action,
            confidence=confidence,
            earnings_event=event,
            entry_price=entry,
            stop_loss=stop_loss,
            target_price=target,
            expected_drift=expected_drift,
            holding_period_days=event.drift_window_remaining,
            reasons=reasons,
            risks=risks
        )

    def get_pead_summary(self, days_back: int = 60) -> Dict[str, Any]:
        """Get summary of PEAD opportunities."""
        signals = self.generate_signals(days_back=days_back, min_score=20.0)

        buy_signals = [s for s in signals if s.action == "BUY"]
        sell_signals = [s for s in signals if s.action == "SELL"]

        return {
            'total_earnings_events': len(self.fetch_recent_earnings(days_back)),
            'actionable_signals': len(signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'top_buys': [s.to_dict() for s in buy_signals[:5]],
            'top_sells': [s.to_dict() for s in sell_signals[:5]],
            'average_confidence': sum(s.confidence for s in signals) / len(signals) if signals else 0,
        }

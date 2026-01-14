"""
News-Based Stock Screener
Analyzes ingested news and press releases to identify investment opportunities.

This module detects positive catalysts from:
- Large contract wins
- Product launches
- Expansion announcements
- Favorable regulatory changes
- Commodity price movements
- Strategic partnerships
- Strong financial results
"""

import sqlite3
import re
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class TimeHorizon(Enum):
    """Investment time horizons."""
    INTRADAY = "intraday"      # Same day
    SHORT = "1d"               # 1 day
    WEEKLY = "1w"              # 1 week
    MONTHLY = "1m"             # 1 month
    QUARTERLY = "3m"           # 3 months
    LONG = "1y"                # 1 year+


class CatalystType(Enum):
    """Types of positive catalysts."""
    CONTRACT_WIN = "contract_win"
    PRODUCT_LAUNCH = "product_launch"
    EXPANSION = "expansion"
    PARTNERSHIP = "partnership"
    ACQUISITION = "acquisition"
    STRONG_RESULTS = "strong_results"
    DIVIDEND = "dividend"
    BUYBACK = "buyback"
    ORDER_WIN = "order_win"
    REGULATORY_APPROVAL = "regulatory_approval"
    COMMODITY_BENEFIT = "commodity_benefit"
    RATING_UPGRADE = "rating_upgrade"
    MANAGEMENT_POSITIVE = "management_positive"
    SECTOR_TAILWIND = "sector_tailwind"


@dataclass
class StockCatalyst:
    """Represents a detected catalyst for a stock."""
    ticker: str
    company_name: str
    catalyst_type: CatalystType
    headline: str
    description: str
    source: str
    date: str
    confidence: float  # 0-1 confidence score
    impact_score: float  # Expected impact (0-10)
    time_relevance: str  # intraday, short, medium, long


@dataclass
class StockRecommendation:
    """A stock recommendation with reasoning."""
    ticker: str
    company_name: str
    score: float
    catalysts: List[StockCatalyst]
    time_horizon: TimeHorizon
    reasoning: str
    risk_level: str
    sector: str


class NewsScreener:
    """
    Screens news and announcements for investment opportunities.

    Features:
    - Catalyst detection from text
    - Time-horizon based filtering
    - Sentiment scoring
    - Sector-aware recommendations
    """

    # Catalyst detection patterns
    CATALYST_PATTERNS = {
        CatalystType.CONTRACT_WIN: [
            r'(?:secured?|won|awarded?|bagged?|received?)\s+(?:a\s+)?(?:major\s+|large\s+|significant\s+)?(?:contract|order|deal)',
            r'contract\s+(?:worth|valued?\s+at|of)\s+(?:Rs\.?|INR|₹)?\s*[\d,]+',
            r'order\s+(?:worth|valued?\s+at|of)\s+(?:Rs\.?|INR|₹)?\s*[\d,]+\s*(?:cr|crore|lakh|million|billion)',
            r'(?:multi-year|long-term)\s+(?:contract|agreement|deal)',
        ],
        CatalystType.PRODUCT_LAUNCH: [
            r'(?:launch(?:ed|es|ing)?|introduc(?:ed|es|ing)?|unveil(?:ed|s|ing)?)\s+(?:new\s+)?(?:product|service|solution|platform)',
            r'(?:new|innovative|revolutionary|breakthrough)\s+(?:product|technology|solution)',
            r'(?:first\s+of\s+its\s+kind|industry-first|pioneering)',
        ],
        CatalystType.EXPANSION: [
            r'(?:expansion|expand(?:s|ed|ing)?)\s+(?:into|in|to)\s+(?:new\s+)?(?:market|region|segment)',
            r'(?:new|additional)\s+(?:plant|facility|factory|warehouse|office)',
            r'(?:capacity\s+)?expansion\s+(?:plan|project)',
            r'(?:capex|capital\s+expenditure)\s+(?:of|worth)\s+(?:Rs\.?|INR|₹)?\s*[\d,]+',
        ],
        CatalystType.PARTNERSHIP: [
            r'(?:strategic\s+)?(?:partnership|alliance|collaboration|tie-up|joint\s+venture)\s+with',
            r'(?:partner(?:ed|s|ing)?|collaborat(?:ed|es|ing)?)\s+with\s+(?:\w+\s+){1,3}(?:limited|ltd|inc|corp)',
            r'(?:mou|memorandum\s+of\s+understanding)\s+(?:signed|executed)',
        ],
        CatalystType.ACQUISITION: [
            r'(?:acquir(?:ed|es|ing)|bought|purchas(?:ed|es|ing))\s+(?:\d+%?\s+)?(?:stake|share|interest)',
            r'acquisition\s+of\s+(?:\w+\s+){1,4}(?:for|worth|valued)',
            r'(?:merger|amalgamation)\s+(?:with|of)',
        ],
        CatalystType.STRONG_RESULTS: [
            r'(?:profit|revenue|income|earnings|pat|ebitda)\s+(?:up|rose|increased|jumped|surged)\s+(?:by\s+)?[\d.]+%',
            r'(?:record|highest|best)\s+(?:ever\s+)?(?:profit|revenue|quarter|performance)',
            r'(?:beat|exceeded|surpassed)\s+(?:estimates|expectations|guidance)',
            r'(?:margin|profitability)\s+(?:improved|expanded|increased)',
        ],
        CatalystType.DIVIDEND: [
            r'(?:declared?|announced?|recommended?)\s+(?:a\s+)?(?:final\s+|interim\s+|special\s+)?dividend',
            r'dividend\s+(?:of\s+)?(?:Rs\.?|INR|₹)?\s*[\d.]+\s*(?:per\s+share)?',
            r'(?:high|attractive|generous)\s+dividend\s+(?:yield|payout)',
        ],
        CatalystType.BUYBACK: [
            r'(?:buyback|buy-back|share\s+repurchase)\s+(?:program|scheme|offer)',
            r'(?:announced?|approved?)\s+(?:a\s+)?buyback',
        ],
        CatalystType.ORDER_WIN: [
            r'(?:received?|won|bagged?|secured?)\s+(?:orders?|contracts?)\s+(?:worth|of|valued)',
            r'order\s+book\s+(?:stands?\s+at|reached|crossed)\s+(?:Rs\.?|INR|₹)?\s*[\d,]+',
            r'(?:strong|robust|healthy)\s+order\s+(?:book|pipeline|inflow)',
        ],
        CatalystType.REGULATORY_APPROVAL: [
            r'(?:received?|got|obtained?)\s+(?:regulatory\s+)?(?:approval|clearance|nod|license)',
            r'(?:fda|usfda|dcgi|sebi|rbi|cci)\s+(?:approval|clearance)',
            r'(?:approved?|cleared?)\s+(?:by|from)\s+(?:regulator|authority)',
        ],
        CatalystType.RATING_UPGRADE: [
            r'(?:rating\s+)?(?:upgrade|upgraded|raised)\s+(?:to|by)',
            r'(?:outlook|view)\s+(?:revised|changed|upgraded)\s+to\s+(?:positive|buy|outperform)',
            r'(?:target\s+price|tp)\s+(?:raised|increased|upgraded)\s+(?:to|by)',
        ],
    }

    # Commodity-related sectors and keywords
    COMMODITY_SECTORS = {
        'gold': ['jewellery', 'gold', 'precious metals', 'titan', 'kalyan', 'tanishq', 'malabar'],
        'oil': ['oil', 'petroleum', 'refinery', 'ongc', 'reliance', 'bpcl', 'hpcl', 'iocl'],
        'steel': ['steel', 'iron ore', 'tata steel', 'jsw', 'sail', 'jindal'],
        'copper': ['copper', 'hindalco', 'vedanta', 'hindustan copper'],
        'agriculture': ['fertilizer', 'seeds', 'agrochemical', 'upl', 'coromandel', 'chambal'],
    }

    # Sector classification
    SECTOR_KEYWORDS = {
        'IT': ['software', 'technology', 'digital', 'cloud', 'saas', 'it services', 'tech'],
        'BANKING': ['bank', 'nbfc', 'financial services', 'lending', 'credit'],
        'PHARMA': ['pharma', 'pharmaceutical', 'drug', 'medicine', 'healthcare', 'hospital'],
        'AUTO': ['automobile', 'automotive', 'vehicle', 'car', 'two-wheeler', 'ev', 'electric vehicle'],
        'FMCG': ['fmcg', 'consumer', 'food', 'beverage', 'personal care'],
        'INFRA': ['infrastructure', 'construction', 'cement', 'real estate', 'roads', 'power'],
        'METALS': ['metal', 'steel', 'aluminium', 'copper', 'mining'],
        'OIL_GAS': ['oil', 'gas', 'petroleum', 'refinery', 'energy'],
        'TELECOM': ['telecom', 'telecommunications', '5g', 'mobile', 'spectrum'],
    }

    # Time relevance scoring based on catalyst type
    TIME_RELEVANCE = {
        CatalystType.CONTRACT_WIN: 'medium',      # Affects next few quarters
        CatalystType.PRODUCT_LAUNCH: 'medium',    # Revenue impact over time
        CatalystType.EXPANSION: 'long',           # Long-term impact
        CatalystType.PARTNERSHIP: 'medium',
        CatalystType.ACQUISITION: 'long',
        CatalystType.STRONG_RESULTS: 'short',     # Immediate market reaction
        CatalystType.DIVIDEND: 'short',           # Near-term event
        CatalystType.BUYBACK: 'short',
        CatalystType.ORDER_WIN: 'medium',
        CatalystType.REGULATORY_APPROVAL: 'short',  # Immediate positive
        CatalystType.RATING_UPGRADE: 'short',
    }

    def __init__(
        self,
        circulars_db: str = "data/cache/circulars.db",
        press_releases_db: str = "data/cache/press_releases.db"
    ):
        self.circulars_db = Path(circulars_db)
        self.press_releases_db = Path(press_releases_db)
        self.logger = logging.getLogger(__name__)

    def _get_time_delta(self, horizon: TimeHorizon) -> timedelta:
        """Convert time horizon to timedelta for filtering."""
        deltas = {
            TimeHorizon.INTRADAY: timedelta(hours=12),
            TimeHorizon.SHORT: timedelta(days=1),
            TimeHorizon.WEEKLY: timedelta(days=7),
            TimeHorizon.MONTHLY: timedelta(days=30),
            TimeHorizon.QUARTERLY: timedelta(days=90),
            TimeHorizon.LONG: timedelta(days=365),
        }
        return deltas.get(horizon, timedelta(days=7))

    def _fetch_recent_news(
        self,
        days_back: int = 7,
        ticker: Optional[str] = None
    ) -> List[Dict]:
        """Fetch recent news from both databases."""
        all_news = []
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()

        # Fetch from circulars
        if self.circulars_db.exists():
            try:
                conn = sqlite3.connect(self.circulars_db)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                sql = """
                    SELECT id, ticker, company_name, title, description,
                           doc_type, exchange, filing_date, category,
                           parsed_text
                    FROM circulars
                    WHERE filing_date >= ?
                """
                params = [cutoff_date]

                if ticker:
                    sql += " AND ticker LIKE ?"
                    params.append(f"%{ticker.replace('.NS', '').replace('.BO', '')}%")

                sql += " ORDER BY filing_date DESC LIMIT 1000"

                cursor.execute(sql, params)
                for row in cursor.fetchall():
                    all_news.append({
                        'id': row['id'],
                        'ticker': row['ticker'],
                        'company_name': row['company_name'],
                        'title': row['title'],
                        'description': row['description'] or '',
                        'content': row['parsed_text'] or '',
                        'date': row['filing_date'],
                        'source': f"{row['exchange']}_circular",
                        'category': row['category'],
                    })
                conn.close()
            except Exception as e:
                self.logger.error(f"Error fetching circulars: {e}")

        # Fetch from press releases
        if self.press_releases_db.exists():
            try:
                conn = sqlite3.connect(self.press_releases_db)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                sql = """
                    SELECT id, ticker, company_name, title, summary, content,
                           source, release_date, categories
                    FROM press_releases
                    WHERE release_date >= ?
                """
                params = [cutoff_date]

                if ticker:
                    sql += " AND ticker LIKE ?"
                    params.append(f"%{ticker.replace('.NS', '').replace('.BO', '')}%")

                sql += " ORDER BY release_date DESC LIMIT 500"

                cursor.execute(sql, params)
                for row in cursor.fetchall():
                    all_news.append({
                        'id': row['id'],
                        'ticker': row['ticker'],
                        'company_name': row['company_name'],
                        'title': row['title'],
                        'description': row['summary'] or '',
                        'content': row['content'] or '',
                        'date': row['release_date'],
                        'source': f"{row['source']}_press",
                        'category': row['categories'],
                    })
                conn.close()
            except Exception as e:
                self.logger.error(f"Error fetching press releases: {e}")

        return all_news

    def detect_catalysts(self, news_item: Dict) -> List[StockCatalyst]:
        """Detect catalysts in a news item."""
        catalysts = []

        # Combine title, description, and content for analysis
        text = f"{news_item.get('title', '')} {news_item.get('description', '')} {news_item.get('content', '')}"
        text_lower = text.lower()

        for catalyst_type, patterns in self.CATALYST_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_confidence(text_lower, catalyst_type)

                    # Calculate impact score
                    impact = self._calculate_impact(text_lower, catalyst_type)

                    catalyst = StockCatalyst(
                        ticker=news_item.get('ticker', ''),
                        company_name=news_item.get('company_name', ''),
                        catalyst_type=catalyst_type,
                        headline=news_item.get('title', '')[:200],
                        description=news_item.get('description', '')[:500],
                        source=news_item.get('source', ''),
                        date=news_item.get('date', ''),
                        confidence=confidence,
                        impact_score=impact,
                        time_relevance=self.TIME_RELEVANCE.get(catalyst_type, 'medium')
                    )
                    catalysts.append(catalyst)
                    break  # One catalyst per type per news item

        return catalysts

    def _calculate_confidence(self, text: str, catalyst_type: CatalystType) -> float:
        """Calculate confidence score for catalyst detection."""
        confidence = 0.5  # Base confidence

        # Higher confidence for specific monetary values
        if re.search(r'(?:Rs\.?|INR|₹)\s*[\d,]+\s*(?:cr|crore|million|billion)', text):
            confidence += 0.2

        # Higher confidence for official announcements
        if any(kw in text for kw in ['board approved', 'board meeting', 'regulatory filing', 'sebi']):
            confidence += 0.15

        # Higher confidence for multiple pattern matches
        matches = 0
        for pattern in self.CATALYST_PATTERNS.get(catalyst_type, []):
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        if matches > 1:
            confidence += 0.15

        return min(confidence, 1.0)

    def _calculate_impact(self, text: str, catalyst_type: CatalystType) -> float:
        """Calculate potential impact score (0-10)."""
        base_impact = {
            CatalystType.CONTRACT_WIN: 6,
            CatalystType.PRODUCT_LAUNCH: 5,
            CatalystType.EXPANSION: 5,
            CatalystType.PARTNERSHIP: 5,
            CatalystType.ACQUISITION: 7,
            CatalystType.STRONG_RESULTS: 7,
            CatalystType.DIVIDEND: 4,
            CatalystType.BUYBACK: 5,
            CatalystType.ORDER_WIN: 6,
            CatalystType.REGULATORY_APPROVAL: 8,
            CatalystType.RATING_UPGRADE: 5,
            CatalystType.COMMODITY_BENEFIT: 4,
            CatalystType.MANAGEMENT_POSITIVE: 3,
            CatalystType.SECTOR_TAILWIND: 4,
        }

        impact = base_impact.get(catalyst_type, 5)

        # Boost for large monetary values
        value_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:cr|crore)', text)
        if value_match:
            try:
                value = float(value_match.group(1).replace(',', ''))
                if value > 1000:  # > 1000 crore
                    impact += 2
                elif value > 100:  # > 100 crore
                    impact += 1
            except ValueError:
                pass

        # Boost for superlatives
        if any(kw in text for kw in ['largest', 'biggest', 'record', 'first', 'breakthrough', 'revolutionary']):
            impact += 1

        return min(impact, 10)

    def _identify_sector(self, text: str, company_name: str = "") -> str:
        """Identify sector from text."""
        combined = f"{text} {company_name}".lower()

        for sector, keywords in self.SECTOR_KEYWORDS.items():
            if any(kw in combined for kw in keywords):
                return sector

        return "DIVERSIFIED"

    def screen_stocks(
        self,
        time_horizon: TimeHorizon = TimeHorizon.WEEKLY,
        sector: Optional[str] = None,
        min_confidence: float = 0.5,
        limit: int = 10
    ) -> List[StockRecommendation]:
        """
        Screen stocks based on recent news and catalysts.

        Args:
            time_horizon: Investment time frame
            sector: Filter by sector (optional)
            min_confidence: Minimum confidence threshold
            limit: Maximum recommendations to return

        Returns:
            List of StockRecommendation sorted by score
        """
        # Determine lookback period based on time horizon
        days_map = {
            TimeHorizon.INTRADAY: 1,
            TimeHorizon.SHORT: 2,
            TimeHorizon.WEEKLY: 7,
            TimeHorizon.MONTHLY: 30,
            TimeHorizon.QUARTERLY: 90,
            TimeHorizon.LONG: 180,
        }
        days_back = days_map.get(time_horizon, 7)

        # Fetch recent news
        news_items = self._fetch_recent_news(days_back=days_back)

        if not news_items:
            self.logger.warning("No recent news found in database")
            return []

        # Detect catalysts for all news
        stock_catalysts: Dict[str, List[StockCatalyst]] = {}

        for item in news_items:
            catalysts = self.detect_catalysts(item)
            for catalyst in catalysts:
                if catalyst.confidence >= min_confidence:
                    ticker = catalyst.ticker
                    if ticker not in stock_catalysts:
                        stock_catalysts[ticker] = []
                    stock_catalysts[ticker].append(catalyst)

        # Score and rank stocks
        recommendations = []

        for ticker, catalysts in stock_catalysts.items():
            if not ticker:
                continue

            # Filter by time relevance
            relevant_catalysts = self._filter_by_time_relevance(catalysts, time_horizon)

            if not relevant_catalysts:
                continue

            # Calculate aggregate score
            score = self._calculate_stock_score(relevant_catalysts, time_horizon)

            # Get company info from first catalyst
            company_name = relevant_catalysts[0].company_name

            # Identify sector
            all_text = " ".join([c.headline + " " + c.description for c in relevant_catalysts])
            detected_sector = self._identify_sector(all_text, company_name)

            # Filter by sector if specified
            if sector and detected_sector != sector.upper():
                continue

            # Determine risk level
            risk_level = self._assess_risk(relevant_catalysts)

            # Generate reasoning
            reasoning = self._generate_reasoning(relevant_catalysts, time_horizon)

            recommendation = StockRecommendation(
                ticker=self._normalize_ticker(ticker),
                company_name=company_name,
                score=score,
                catalysts=relevant_catalysts,
                time_horizon=time_horizon,
                reasoning=reasoning,
                risk_level=risk_level,
                sector=detected_sector
            )
            recommendations.append(recommendation)

        # Sort by score and return top N
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:limit]

    def _filter_by_time_relevance(
        self,
        catalysts: List[StockCatalyst],
        horizon: TimeHorizon
    ) -> List[StockCatalyst]:
        """Filter catalysts by time relevance."""
        horizon_priority = {
            TimeHorizon.INTRADAY: ['short'],
            TimeHorizon.SHORT: ['short'],
            TimeHorizon.WEEKLY: ['short', 'medium'],
            TimeHorizon.MONTHLY: ['short', 'medium'],
            TimeHorizon.QUARTERLY: ['medium', 'long'],
            TimeHorizon.LONG: ['medium', 'long'],
        }

        allowed = horizon_priority.get(horizon, ['short', 'medium', 'long'])
        return [c for c in catalysts if c.time_relevance in allowed]

    def _calculate_stock_score(
        self,
        catalysts: List[StockCatalyst],
        horizon: TimeHorizon
    ) -> float:
        """Calculate aggregate score for a stock."""
        if not catalysts:
            return 0

        # Base score from impact and confidence
        impact_sum = sum(c.impact_score * c.confidence for c in catalysts)

        # Bonus for multiple catalysts (diversified positive news)
        catalyst_bonus = min(len(catalysts) * 0.5, 3)

        # Recency bonus (more recent = higher score for short-term)
        recency_bonus = 0
        if horizon in [TimeHorizon.INTRADAY, TimeHorizon.SHORT, TimeHorizon.WEEKLY]:
            today = datetime.now().date()
            for c in catalysts:
                try:
                    cat_date = datetime.fromisoformat(c.date.split('T')[0]).date()
                    days_old = (today - cat_date).days
                    if days_old <= 1:
                        recency_bonus += 2
                    elif days_old <= 3:
                        recency_bonus += 1
                except (ValueError, AttributeError):
                    pass

        # Calculate final score
        score = impact_sum + catalyst_bonus + recency_bonus

        # Normalize to 0-100 scale
        return min(score * 5, 100)

    def _assess_risk(self, catalysts: List[StockCatalyst]) -> str:
        """Assess risk level based on catalysts."""
        # More catalysts with high confidence = lower risk
        avg_confidence = sum(c.confidence for c in catalysts) / len(catalysts)

        # Check for high-impact events
        has_high_impact = any(c.impact_score >= 7 for c in catalysts)

        if avg_confidence > 0.7 and has_high_impact:
            return "LOW"
        elif avg_confidence > 0.5:
            return "MEDIUM"
        else:
            return "HIGH"

    def _generate_reasoning(
        self,
        catalysts: List[StockCatalyst],
        horizon: TimeHorizon
    ) -> str:
        """Generate human-readable reasoning."""
        if not catalysts:
            return "No specific catalysts identified."

        # Group by catalyst type
        catalyst_types = {}
        for c in catalysts:
            ct = c.catalyst_type.value.replace('_', ' ').title()
            if ct not in catalyst_types:
                catalyst_types[ct] = []
            catalyst_types[ct].append(c.headline[:100])

        # Build reasoning
        reasons = []
        for ctype, headlines in catalyst_types.items():
            if len(headlines) == 1:
                reasons.append(f"{ctype}: {headlines[0]}")
            else:
                reasons.append(f"{ctype} ({len(headlines)} events)")

        horizon_text = {
            TimeHorizon.INTRADAY: "today",
            TimeHorizon.SHORT: "in the next day",
            TimeHorizon.WEEKLY: "this week",
            TimeHorizon.MONTHLY: "this month",
            TimeHorizon.QUARTERLY: "this quarter",
            TimeHorizon.LONG: "over the long term",
        }

        return f"Potential upside {horizon_text.get(horizon, 'expected')} based on: " + "; ".join(reasons[:3])

    def _normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker to NSE format."""
        ticker = ticker.upper().strip()
        if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
            # Check if it's numeric (BSE code)
            if ticker.isdigit():
                return f"{ticker}.BO"
            return f"{ticker}.NS"
        return ticker

    def get_commodity_plays(
        self,
        commodity: str,
        trend: str = "bullish"
    ) -> List[Dict]:
        """
        Get stocks that benefit from commodity price movements.

        Args:
            commodity: gold, oil, steel, copper, agriculture
            trend: bullish or bearish

        Returns:
            List of related stocks with reasoning
        """
        commodity = commodity.lower()
        if commodity not in self.COMMODITY_SECTORS:
            return []

        keywords = self.COMMODITY_SECTORS[commodity]

        # Search news for commodity mentions
        news_items = self._fetch_recent_news(days_back=30)

        relevant_stocks = []
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('description', '')}".lower()
            if any(kw in text for kw in keywords):
                relevant_stocks.append({
                    'ticker': self._normalize_ticker(item.get('ticker', '')),
                    'company_name': item.get('company_name', ''),
                    'headline': item.get('title', ''),
                    'commodity': commodity,
                    'trend_benefit': trend,
                })

        # Deduplicate by ticker
        seen = set()
        unique = []
        for stock in relevant_stocks:
            if stock['ticker'] and stock['ticker'] not in seen:
                seen.add(stock['ticker'])
                unique.append(stock)

        return unique[:10]

    def get_screening_summary(self) -> Dict:
        """Get summary of available data for screening."""
        summary = {
            'circulars_available': False,
            'press_releases_available': False,
            'total_documents': 0,
            'date_range': None,
        }

        if self.circulars_db.exists():
            try:
                conn = sqlite3.connect(self.circulars_db)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*), MIN(filing_date), MAX(filing_date) FROM circulars")
                count, min_date, max_date = cursor.fetchone()
                if count:
                    summary['circulars_available'] = True
                    summary['total_documents'] += count
                    summary['date_range'] = (min_date, max_date)
                conn.close()
            except Exception:
                pass

        if self.press_releases_db.exists():
            try:
                conn = sqlite3.connect(self.press_releases_db)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM press_releases")
                count = cursor.fetchone()[0]
                if count:
                    summary['press_releases_available'] = True
                    summary['total_documents'] += count
                conn.close()
            except Exception:
                pass

        return summary

"""
News and Sentiment Pipeline
Aggregates news and computes sentiment scores.
"""

import requests
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
import sqlite3
from pathlib import Path
import json
import re

logger = logging.getLogger(__name__)


class SentimentPipeline:
    """
    Fetches news and computes aggregated sentiment.

    Sources: NewsAPI, Finnhub, Marketaux

    Sentiment scoring:
    - Uses keyword-based analysis as primary method
    - Can integrate with external APIs for advanced NLP
    """

    # Source reliability weights
    SOURCE_WEIGHTS = {
        'reuters': 1.0,
        'bloomberg': 1.0,
        'economic times': 0.9,
        'moneycontrol': 0.85,
        'business standard': 0.85,
        'livemint': 0.85,
        'financial express': 0.8,
        'cnbc': 0.8,
        'default': 0.7
    }

    # Sentiment keywords
    POSITIVE_KEYWORDS = [
        'surge', 'soar', 'rally', 'gain', 'profit', 'growth', 'record', 'high',
        'bullish', 'upgrade', 'outperform', 'beat', 'strong', 'robust', 'boost',
        'positive', 'optimistic', 'expand', 'increase', 'rise', 'jump', 'recover',
        'breakthrough', 'success', 'win', 'deal', 'partnership', 'acquisition'
    ]

    NEGATIVE_KEYWORDS = [
        'crash', 'plunge', 'fall', 'drop', 'loss', 'decline', 'low', 'bearish',
        'downgrade', 'underperform', 'miss', 'weak', 'poor', 'cut', 'negative',
        'pessimistic', 'shrink', 'decrease', 'slump', 'slide', 'warning', 'fraud',
        'scandal', 'investigation', 'lawsuit', 'default', 'bankruptcy', 'debt'
    ]

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        finnhub_key: Optional[str] = None,
        db_path: str = "data/cache/sentiment.db"
    ):
        self.newsapi_key = newsapi_key
        self.finnhub_key = finnhub_key
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._init_db()

    def _init_db(self):
        """Initialize database for caching news and sentiment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id TEXT PRIMARY KEY,
                ticker TEXT,
                headline TEXT NOT NULL,
                summary TEXT,
                source TEXT,
                url TEXT,
                published_at TEXT,
                sentiment_score REAL,
                fetched_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_cache (
                ticker TEXT PRIMARY KEY,
                sentiment_score REAL,
                article_count INTEGER,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                computed_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_ticker
            ON news_articles(ticker)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_date
            ON news_articles(published_at)
        """)

        conn.commit()
        conn.close()

    def get_ticker_news(
        self,
        ticker: str,
        days_back: int = 7,
        max_articles: int = 20
    ) -> List[Dict]:
        """
        Fetch recent news for a ticker.

        Args:
            ticker: Stock symbol (e.g., "RELIANCE")
            days_back: Number of days to look back
            max_articles: Maximum articles to return

        Returns:
            List of {headline, source, url, published_at, sentiment_score}
        """
        # Clean ticker symbol
        clean_ticker = ticker.replace('.NS', '').replace('.BO', '').upper()

        articles = []

        # Try NewsAPI if configured
        if self.newsapi_key:
            articles.extend(
                self._fetch_from_newsapi(clean_ticker, days_back, max_articles)
            )

        # Try Finnhub if configured
        if self.finnhub_key and len(articles) < max_articles:
            articles.extend(
                self._fetch_from_finnhub(clean_ticker, days_back, max_articles - len(articles))
            )

        # If no API keys, use cached or return empty
        if not articles:
            articles = self._get_cached_news(clean_ticker, days_back)

        # Calculate sentiment for each article
        for article in articles:
            if article.get('sentiment_score') is None:
                article['sentiment_score'] = self._calculate_article_sentiment(
                    article.get('headline', ''),
                    article.get('summary', '')
                )

        # Cache articles
        self._cache_articles(clean_ticker, articles)

        return articles[:max_articles]

    def _fetch_from_newsapi(
        self,
        ticker: str,
        days_back: int,
        max_articles: int
    ) -> List[Dict]:
        """Fetch news from NewsAPI."""
        articles = []

        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            # Get company name mapping for better search
            company_names = {
                'RELIANCE': 'Reliance Industries',
                'TCS': 'Tata Consultancy Services',
                'INFY': 'Infosys',
                'HDFCBANK': 'HDFC Bank',
                'ICICIBANK': 'ICICI Bank',
                'WIPRO': 'Wipro',
                'BHARTIARTL': 'Bharti Airtel',
                'SBIN': 'State Bank of India',
                'KOTAKBANK': 'Kotak Mahindra Bank',
                'HINDUNILVR': 'Hindustan Unilever',
            }

            query = company_names.get(ticker, ticker)

            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{query}" stock',
                'from': from_date,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': max_articles,
                'apiKey': self.newsapi_key
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                for item in data.get('articles', []):
                    articles.append({
                        'id': self._generate_article_id(item.get('url', '')),
                        'ticker': ticker,
                        'headline': item.get('title', ''),
                        'summary': item.get('description', ''),
                        'source': item.get('source', {}).get('name', 'Unknown'),
                        'url': item.get('url', ''),
                        'published_at': item.get('publishedAt', ''),
                        'sentiment_score': None  # Calculate later
                    })

        except Exception as e:
            self.logger.error(f"Error fetching from NewsAPI: {e}")

        return articles

    def _fetch_from_finnhub(
        self,
        ticker: str,
        days_back: int,
        max_articles: int
    ) -> List[Dict]:
        """Fetch news from Finnhub."""
        articles = []

        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')

            # Finnhub uses different ticker format
            symbol = f"{ticker}.NS"

            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': from_date,
                'to': to_date,
                'token': self.finnhub_key
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                for item in data[:max_articles]:
                    articles.append({
                        'id': self._generate_article_id(item.get('url', '')),
                        'ticker': ticker,
                        'headline': item.get('headline', ''),
                        'summary': item.get('summary', ''),
                        'source': item.get('source', 'Unknown'),
                        'url': item.get('url', ''),
                        'published_at': datetime.fromtimestamp(
                            item.get('datetime', 0)
                        ).isoformat(),
                        'sentiment_score': None
                    })

        except Exception as e:
            self.logger.error(f"Error fetching from Finnhub: {e}")

        return articles

    def _generate_article_id(self, url: str) -> str:
        """Generate unique article ID from URL."""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()[:16]

    def _calculate_article_sentiment(
        self,
        headline: str,
        summary: str = ""
    ) -> float:
        """
        Calculate sentiment score for an article.

        Returns:
            Score between -1 (very negative) and 1 (very positive)
        """
        text = f"{headline} {summary}".lower()

        positive_count = sum(
            1 for word in self.POSITIVE_KEYWORDS
            if word in text
        )
        negative_count = sum(
            1 for word in self.NEGATIVE_KEYWORDS
            if word in text
        )

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        # Calculate score between -1 and 1
        score = (positive_count - negative_count) / total

        return round(score, 3)

    def _get_cached_news(
        self,
        ticker: str,
        days_back: int
    ) -> List[Dict]:
        """Get cached news articles."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()

        cursor.execute("""
            SELECT * FROM news_articles
            WHERE ticker = ? AND published_at >= ?
            ORDER BY published_at DESC
        """, (ticker, cutoff))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def _cache_articles(self, ticker: str, articles: List[Dict]):
        """Cache news articles to database."""
        if not articles:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for article in articles:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO news_articles
                    (id, ticker, headline, summary, source, url,
                     published_at, sentiment_score, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article.get('id'),
                    ticker,
                    article.get('headline'),
                    article.get('summary'),
                    article.get('source'),
                    article.get('url'),
                    article.get('published_at'),
                    article.get('sentiment_score'),
                    datetime.now().isoformat()
                ))
            except Exception as e:
                self.logger.warning(f"Error caching article: {e}")

        conn.commit()
        conn.close()

    def get_sector_sentiment(self, sector: str) -> Dict:
        """
        Aggregate sentiment for a sector.

        Args:
            sector: Sector name (e.g., "IT", "Banking", "Pharma")

        Returns:
            {
                "sector": str,
                "sentiment_score": float,
                "article_count": int,
                "top_headlines": List[str],
                "computed_at": str
            }
        """
        # Sector to tickers mapping
        sector_tickers = {
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM'],
            'BANKING': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
            'PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON'],
            'AUTO': ['MARUTI', 'TATAMOTORS', 'BAJAJ-AUTO', 'M&M', 'HEROMOTOCO'],
            'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR'],
            'ENERGY': ['RELIANCE', 'ONGC', 'NTPC', 'POWERGRID', 'ADANIGREEN'],
            'METALS': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL', 'COALINDIA'],
            'REALTY': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'BRIGADE'],
        }

        tickers = sector_tickers.get(sector.upper(), [])
        if not tickers:
            return {
                "sector": sector,
                "sentiment_score": 0.0,
                "article_count": 0,
                "error": "Unknown sector",
                "computed_at": datetime.now().isoformat()
            }

        all_articles = []
        for ticker in tickers:
            articles = self.get_ticker_news(ticker, days_back=7, max_articles=5)
            all_articles.extend(articles)

        if not all_articles:
            return {
                "sector": sector,
                "sentiment_score": 0.0,
                "article_count": 0,
                "top_headlines": [],
                "computed_at": datetime.now().isoformat()
            }

        # Calculate aggregate sentiment
        sentiment_score = self.aggregate_sentiment(all_articles)

        # Get top headlines
        sorted_articles = sorted(
            all_articles,
            key=lambda x: abs(x.get('sentiment_score', 0)),
            reverse=True
        )
        top_headlines = [a['headline'] for a in sorted_articles[:5]]

        return {
            "sector": sector,
            "sentiment_score": sentiment_score,
            "article_count": len(all_articles),
            "top_headlines": top_headlines,
            "tickers_analyzed": tickers,
            "computed_at": datetime.now().isoformat()
        }

    def aggregate_sentiment(self, articles: List[Dict]) -> float:
        """
        Compute weighted sentiment score (0-1).

        Weights by recency and source reliability.

        Args:
            articles: List of articles with sentiment_score and published_at

        Returns:
            Aggregated sentiment score (-1 to 1)
        """
        if not articles:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0

        now = datetime.now()

        for article in articles:
            sentiment = article.get('sentiment_score', 0)
            if sentiment is None:
                continue

            # Source weight
            source = article.get('source', '').lower()
            source_weight = self.SOURCE_WEIGHTS.get(
                source,
                self.SOURCE_WEIGHTS['default']
            )

            # Recency weight (exponential decay)
            try:
                pub_date = datetime.fromisoformat(
                    article.get('published_at', '').replace('Z', '+00:00')
                )
                days_old = (now - pub_date.replace(tzinfo=None)).days
                recency_weight = 0.9 ** days_old  # 10% decay per day
            except (ValueError, TypeError):
                recency_weight = 0.5

            # Combined weight
            weight = source_weight * recency_weight

            weighted_sum += sentiment * weight
            weight_total += weight

        if weight_total == 0:
            return 0.0

        return round(weighted_sum / weight_total, 3)

    def get_market_sentiment(self) -> Dict:
        """
        Get overall market sentiment.

        Analyzes sentiment for major indices and top stocks.
        """
        sectors = ['IT', 'BANKING', 'PHARMA', 'AUTO', 'FMCG', 'ENERGY']

        sector_sentiments = {}
        total_score = 0.0
        total_articles = 0

        for sector in sectors:
            sector_data = self.get_sector_sentiment(sector)
            sector_sentiments[sector] = {
                'score': sector_data['sentiment_score'],
                'articles': sector_data['article_count']
            }
            total_score += sector_data['sentiment_score'] * sector_data['article_count']
            total_articles += sector_data['article_count']

        overall_score = total_score / total_articles if total_articles > 0 else 0.0

        # Determine market mood
        if overall_score > 0.3:
            mood = "BULLISH"
        elif overall_score > 0.1:
            mood = "SLIGHTLY_BULLISH"
        elif overall_score < -0.3:
            mood = "BEARISH"
        elif overall_score < -0.1:
            mood = "SLIGHTLY_BEARISH"
        else:
            mood = "NEUTRAL"

        return {
            "overall_sentiment": round(overall_score, 3),
            "mood": mood,
            "sectors": sector_sentiments,
            "total_articles_analyzed": total_articles,
            "computed_at": datetime.now().isoformat()
        }

    def get_sentiment_summary(
        self,
        ticker: str,
        days_back: int = 7
    ) -> Dict:
        """
        Get comprehensive sentiment summary for a ticker.

        Returns:
            {
                "ticker": str,
                "overall_sentiment": float,
                "sentiment_trend": str,
                "positive_count": int,
                "negative_count": int,
                "neutral_count": int,
                "key_themes": List[str],
                "recent_headlines": List[Dict]
            }
        """
        articles = self.get_ticker_news(ticker, days_back=days_back)

        if not articles:
            return {
                "ticker": ticker,
                "overall_sentiment": 0.0,
                "sentiment_trend": "UNKNOWN",
                "article_count": 0,
                "message": "No recent news found"
            }

        # Count sentiment categories
        positive = sum(1 for a in articles if a.get('sentiment_score', 0) > 0.2)
        negative = sum(1 for a in articles if a.get('sentiment_score', 0) < -0.2)
        neutral = len(articles) - positive - negative

        # Calculate aggregate
        overall = self.aggregate_sentiment(articles)

        # Determine trend
        if overall > 0.3:
            trend = "STRONGLY_POSITIVE"
        elif overall > 0.1:
            trend = "POSITIVE"
        elif overall < -0.3:
            trend = "STRONGLY_NEGATIVE"
        elif overall < -0.1:
            trend = "NEGATIVE"
        else:
            trend = "NEUTRAL"

        return {
            "ticker": ticker,
            "overall_sentiment": overall,
            "sentiment_trend": trend,
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "article_count": len(articles),
            "recent_headlines": [
                {
                    "headline": a['headline'],
                    "source": a['source'],
                    "sentiment": a.get('sentiment_score', 0),
                    "date": a.get('published_at')
                }
                for a in articles[:5]
            ],
            "computed_at": datetime.now().isoformat()
        }

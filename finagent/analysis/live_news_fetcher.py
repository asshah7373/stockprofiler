"""
Live News Fetcher Module

Fetches real-time news for Indian stocks from multiple sources:
1. Yahoo Finance news API (via yfinance)
2. Google News RSS feeds
3. NSE India announcements

No pre-ingestion required - fetches on demand during scan.
"""

import logging
import re
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """Represents a news item."""
    title: str
    description: str
    source: str
    published_date: Optional[datetime] = None
    url: str = ""
    ticker: str = ""

    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'description': self.description,
            'source': self.source,
            'published_date': self.published_date.isoformat() if self.published_date else None,
            'url': self.url,
            'ticker': self.ticker,
        }


class LiveNewsFetcher:
    """
    Fetches live news from multiple sources.

    Sources:
    1. Yahoo Finance - stock-specific news via yfinance
    2. Google News RSS - search-based news
    3. NSE India - corporate announcements

    Usage:
        fetcher = LiveNewsFetcher()
        news = fetcher.fetch_news("RELIANCE.NS", days_back=7)
    """

    # Cache to avoid repeated fetches
    _cache: Dict[str, List[NewsItem]] = {}
    _cache_time: Dict[str, datetime] = {}
    _cache_ttl = timedelta(minutes=30)  # Cache for 30 minutes

    # Company name mappings for better search
    COMPANY_NAMES = {
        "RELIANCE": "Reliance Industries",
        "TCS": "Tata Consultancy Services",
        "HDFCBANK": "HDFC Bank",
        "INFY": "Infosys",
        "ICICIBANK": "ICICI Bank",
        "HINDUNILVR": "Hindustan Unilever",
        "SBIN": "State Bank of India",
        "BHARTIARTL": "Bharti Airtel",
        "KOTAKBANK": "Kotak Mahindra Bank",
        "ITC": "ITC Limited",
        "LT": "Larsen & Toubro",
        "AXISBANK": "Axis Bank",
        "ASIANPAINT": "Asian Paints",
        "MARUTI": "Maruti Suzuki",
        "HCLTECH": "HCL Technologies",
        "SUNPHARMA": "Sun Pharmaceutical",
        "BAJFINANCE": "Bajaj Finance",
        "TITAN": "Titan Company",
        "WIPRO": "Wipro",
        "TATASTEEL": "Tata Steel",
        "TATAMOTORS": "Tata Motors",
        "ADANIENT": "Adani Enterprises",
        "ADANIPORTS": "Adani Ports",
        "NTPC": "NTPC Limited",
        "POWERGRID": "Power Grid Corporation",
        "ONGC": "Oil and Natural Gas Corporation",
        "COALINDIA": "Coal India",
        "JSWSTEEL": "JSW Steel",
        "HINDALCO": "Hindalco Industries",
        "ULTRACEMCO": "UltraTech Cement",
        "GRASIM": "Grasim Industries",
        "TECHM": "Tech Mahindra",
        "DRREDDY": "Dr. Reddy's Laboratories",
        "CIPLA": "Cipla",
        "BRITANNIA": "Britannia Industries",
        "NESTLEIND": "Nestle India",
        "DIVISLAB": "Divi's Laboratories",
        "APOLLOHOSP": "Apollo Hospitals",
        "EICHERMOT": "Eicher Motors",
        "BAJAJ-AUTO": "Bajaj Auto",
        "HEROMOTOCO": "Hero MotoCorp",
        "INDUSINDBK": "IndusInd Bank",
        "BPCL": "Bharat Petroleum",
        "HDFCLIFE": "HDFC Life Insurance",
        "SBILIFE": "SBI Life Insurance",
        "TATACONSUM": "Tata Consumer Products",
        "UPL": "UPL Limited",
    }

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.logger = logging.getLogger(__name__)

    def _get_cache_key(self, ticker: str, days_back: int) -> str:
        """Generate cache key."""
        return f"{ticker}_{days_back}"

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache is still valid."""
        if not self.use_cache:
            return False
        if key not in self._cache_time:
            return False
        return datetime.now() - self._cache_time[key] < self._cache_ttl

    def _normalize_ticker(self, ticker: str) -> str:
        """Extract base ticker from full symbol."""
        return ticker.replace('.NS', '').replace('.BO', '').upper()

    def _get_company_name(self, ticker: str) -> str:
        """Get company name for search queries."""
        base_ticker = self._normalize_ticker(ticker)
        return self.COMPANY_NAMES.get(base_ticker, base_ticker)

    def fetch_news(self, ticker: str, days_back: int = 7) -> List[NewsItem]:
        """
        Fetch news for a ticker from all sources.

        Args:
            ticker: Stock ticker (e.g., "RELIANCE.NS")
            days_back: How many days back to fetch news

        Returns:
            List of NewsItem objects
        """
        cache_key = self._get_cache_key(ticker, days_back)

        # Check cache
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        all_news = []

        # Fetch from Yahoo Finance
        yahoo_news = self._fetch_yahoo_news(ticker)
        all_news.extend(yahoo_news)

        # Fetch from Google News RSS
        google_news = self._fetch_google_news(ticker, days_back)
        all_news.extend(google_news)

        # Deduplicate by title similarity
        unique_news = self._deduplicate_news(all_news)

        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        filtered_news = [
            n for n in unique_news
            if n.published_date is None or n.published_date >= cutoff_date
        ]

        # Sort by date (newest first)
        filtered_news.sort(
            key=lambda n: n.published_date or datetime.min,
            reverse=True
        )

        # Cache results
        self._cache[cache_key] = filtered_news
        self._cache_time[cache_key] = datetime.now()

        return filtered_news

    def _fetch_yahoo_news(self, ticker: str) -> List[NewsItem]:
        """Fetch news from Yahoo Finance via yfinance."""
        news_items = []

        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)

            # Get news from yfinance
            if hasattr(stock, 'news') and stock.news:
                for item in stock.news[:10]:  # Limit to 10 items
                    try:
                        # Parse publish time
                        pub_time = None
                        if 'providerPublishTime' in item:
                            pub_time = datetime.fromtimestamp(item['providerPublishTime'])

                        news_items.append(NewsItem(
                            title=item.get('title', ''),
                            description=item.get('summary', item.get('title', '')),
                            source=f"Yahoo/{item.get('publisher', 'Unknown')}",
                            published_date=pub_time,
                            url=item.get('link', ''),
                            ticker=ticker,
                        ))
                    except Exception as e:
                        self.logger.debug(f"Error parsing Yahoo news item: {e}")
                        continue

        except Exception as e:
            self.logger.debug(f"Error fetching Yahoo news for {ticker}: {e}")

        return news_items

    def _fetch_google_news(self, ticker: str, days_back: int = 7) -> List[NewsItem]:
        """Fetch news from Google News RSS."""
        news_items = []

        try:
            import urllib.request

            company_name = self._get_company_name(ticker)
            base_ticker = self._normalize_ticker(ticker)

            # Search query: company name + stock
            query = f"{company_name} stock NSE"
            encoded_query = quote_plus(query)

            # Google News RSS URL
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

            # Fetch RSS
            req = urllib.request.Request(
                rss_url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()

            # Parse XML
            root = ET.fromstring(xml_data)

            # Find all items
            for item in root.findall('.//item')[:10]:  # Limit to 10 items
                try:
                    title = item.find('title')
                    title_text = title.text if title is not None else ''

                    description = item.find('description')
                    desc_text = description.text if description is not None else ''

                    # Clean HTML from description
                    desc_text = re.sub(r'<[^>]+>', '', desc_text)

                    pub_date = item.find('pubDate')
                    pub_time = None
                    if pub_date is not None and pub_date.text:
                        try:
                            # Parse RFC 822 date format
                            from email.utils import parsedate_to_datetime
                            pub_time = parsedate_to_datetime(pub_date.text)
                            pub_time = pub_time.replace(tzinfo=None)  # Remove timezone
                        except Exception:
                            pass

                    link = item.find('link')
                    link_text = link.text if link is not None else ''

                    source = item.find('source')
                    source_text = source.text if source is not None else 'Google News'

                    news_items.append(NewsItem(
                        title=title_text,
                        description=desc_text[:500],  # Truncate
                        source=f"Google/{source_text}",
                        published_date=pub_time,
                        url=link_text,
                        ticker=ticker,
                    ))
                except Exception as e:
                    self.logger.debug(f"Error parsing Google news item: {e}")
                    continue

        except Exception as e:
            self.logger.debug(f"Error fetching Google news for {ticker}: {e}")

        return news_items

    def _deduplicate_news(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate news items based on title similarity."""
        unique = []
        seen_hashes = set()

        for item in news_items:
            # Create hash from normalized title
            normalized = re.sub(r'[^a-z0-9]', '', item.title.lower())
            title_hash = hashlib.md5(normalized[:50].encode()).hexdigest()

            if title_hash not in seen_hashes:
                seen_hashes.add(title_hash)
                unique.append(item)

        return unique

    def fetch_batch(self, tickers: List[str], days_back: int = 7) -> Dict[str, List[NewsItem]]:
        """Fetch news for multiple tickers."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.fetch_news(ticker, days_back)
        return results

    def clear_cache(self):
        """Clear the news cache."""
        self._cache.clear()
        self._cache_time.clear()


# Singleton instance for reuse
_fetcher_instance = None


def get_live_news_fetcher() -> LiveNewsFetcher:
    """Get singleton LiveNewsFetcher instance."""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = LiveNewsFetcher()
    return _fetcher_instance


def fetch_live_news(ticker: str, days_back: int = 7) -> List[NewsItem]:
    """Convenience function to fetch live news."""
    fetcher = get_live_news_fetcher()
    return fetcher.fetch_news(ticker, days_back)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    fetcher = LiveNewsFetcher()

    # Test with a few tickers
    test_tickers = ["RELIANCE.NS", "INFY.NS", "TCS.NS"]

    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"News for {ticker}")
        print('='*60)

        news = fetcher.fetch_news(ticker, days_back=7)

        if not news:
            print("  No news found")
        else:
            for i, item in enumerate(news[:5], 1):
                print(f"\n{i}. [{item.source}] {item.title[:80]}...")
                if item.published_date:
                    print(f"   Date: {item.published_date.strftime('%Y-%m-%d %H:%M')}")

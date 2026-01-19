"""
Live News Fetcher Module

Fetches real-time news for Indian stocks from multiple sources:
1. Yahoo Finance news API (via yfinance)
2. Google News RSS feeds
3. NSE India corporate announcements
4. BSE India corporate announcements

No pre-ingestion required - fetches on demand during scan.
"""

import logging
import re
import hashlib
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urlencode
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
    3. NSE India - corporate announcements/circulars
    4. BSE India - corporate announcements/circulars

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
            self.logger.debug(f"Using cached news for {ticker}")
            return self._cache[cache_key]

        all_news = []
        errors = []

        # Fetch from NSE India corporate announcements (highest priority for Indian stocks)
        try:
            nse_news = self._fetch_nse_announcements(ticker, days_back)
            all_news.extend(nse_news)
            self.logger.debug(f"NSE returned {len(nse_news)} announcements for {ticker}")
        except Exception as e:
            errors.append(f"NSE: {e}")

        # Fetch from BSE India corporate announcements
        try:
            bse_news = self._fetch_bse_announcements(ticker, days_back)
            all_news.extend(bse_news)
            self.logger.debug(f"BSE returned {len(bse_news)} announcements for {ticker}")
        except Exception as e:
            errors.append(f"BSE: {e}")

        # Fetch from Yahoo Finance
        try:
            yahoo_news = self._fetch_yahoo_news(ticker)
            all_news.extend(yahoo_news)
            self.logger.debug(f"Yahoo Finance returned {len(yahoo_news)} items for {ticker}")
        except Exception as e:
            errors.append(f"Yahoo: {e}")

        # Fetch from Google News RSS
        try:
            google_news = self._fetch_google_news(ticker, days_back)
            all_news.extend(google_news)
            self.logger.debug(f"Google News returned {len(google_news)} items for {ticker}")
        except Exception as e:
            errors.append(f"Google: {e}")

        # Log any errors at warning level so they're visible
        if errors and not all_news:
            self.logger.warning(f"News fetch errors for {ticker}: {'; '.join(errors)}")

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

        self.logger.debug(f"Total {len(filtered_news)} news items for {ticker}")
        return filtered_news

    def _fetch_yahoo_news(self, ticker: str) -> List[NewsItem]:
        """Fetch news from Yahoo Finance via yfinance."""
        news_items = []

        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)

            # Get news from yfinance - try both 'news' attribute and get_news method
            news_data = None
            if hasattr(stock, 'news'):
                news_data = stock.news
            elif hasattr(stock, 'get_news'):
                try:
                    news_data = stock.get_news()
                except Exception:
                    pass

            if news_data:
                self.logger.debug(f"Yahoo Finance returned {len(news_data)} raw news items for {ticker}")
                for item in news_data[:10]:  # Limit to 10 items
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
            else:
                self.logger.debug(f"No news data from Yahoo Finance for {ticker}")

        except ImportError:
            self.logger.warning("yfinance not installed - Yahoo Finance news disabled")
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

    def _fetch_nse_announcements(self, ticker: str, days_back: int = 7) -> List[NewsItem]:
        """
        Fetch corporate announcements from NSE India.

        NSE provides official corporate announcements including:
        - Quarterly/Annual results
        - Board meeting outcomes
        - Corporate actions (dividends, bonus, splits)
        - Shareholding patterns
        - Press releases
        """
        news_items = []

        try:
            import urllib.request
            import ssl

            base_ticker = self._normalize_ticker(ticker)

            # NSE API endpoint for corporate announcements
            # Using the symbol to fetch announcements
            nse_url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={base_ticker}"

            # NSE requires specific headers to work
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.nseindia.com/',
                'Connection': 'keep-alive',
            }

            # Create SSL context that doesn't verify (NSE has certificate issues sometimes)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(nse_url, headers=headers)

            with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
                data = json.loads(response.read().decode('utf-8'))

            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Process announcements
            if isinstance(data, list):
                for item in data[:15]:  # Limit to 15 items
                    try:
                        # Parse date
                        pub_time = None
                        date_str = item.get('an_dt') or item.get('date')
                        if date_str:
                            try:
                                # Try different date formats
                                for fmt in ['%d-%b-%Y', '%d-%m-%Y', '%Y-%m-%d']:
                                    try:
                                        pub_time = datetime.strptime(date_str, fmt)
                                        break
                                    except ValueError:
                                        continue
                            except Exception:
                                pass

                        # Skip if older than cutoff
                        if pub_time and pub_time < cutoff_date:
                            continue

                        subject = item.get('desc') or item.get('subject') or ''
                        attchmnt = item.get('attchmntText') or item.get('attachment') or ''
                        category = item.get('smIndustry') or item.get('category') or 'Corporate'

                        news_items.append(NewsItem(
                            title=subject[:200],
                            description=f"{category}: {attchmnt[:300]}" if attchmnt else subject,
                            source=f"NSE/{category}",
                            published_date=pub_time,
                            url=f"https://www.nseindia.com/companies-listing/corporate-filings-announcements",
                            ticker=ticker,
                        ))
                    except Exception as e:
                        self.logger.debug(f"Error parsing NSE item: {e}")
                        continue

        except Exception as e:
            self.logger.debug(f"Error fetching NSE announcements for {ticker}: {e}")

        return news_items

    def _fetch_bse_announcements(self, ticker: str, days_back: int = 7) -> List[NewsItem]:
        """
        Fetch corporate announcements from BSE India.

        BSE provides official corporate announcements including:
        - Financial results
        - Board meetings
        - Corporate actions
        - Press releases
        - Shareholding disclosures
        """
        news_items = []

        try:
            import urllib.request
            import ssl

            base_ticker = self._normalize_ticker(ticker)
            company_name = self._get_company_name(ticker)

            # BSE API for announcements - search by company name
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
            to_date = datetime.now().strftime('%Y%m%d')

            # BSE uses scrip codes, but we can search by name
            bse_url = f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?strCat=-1&strPrevDate={from_date}&strScrip=&strSearch=P&strToDate={to_date}&strType=C"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.bseindia.com/',
                'Origin': 'https://www.bseindia.com',
            }

            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(bse_url, headers=headers)

            with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
                data = json.loads(response.read().decode('utf-8'))

            # Filter by company name/ticker
            if isinstance(data, dict) and 'Table' in data:
                announcements = data['Table']
            elif isinstance(data, list):
                announcements = data
            else:
                announcements = []

            cutoff_date = datetime.now() - timedelta(days=days_back)

            for item in announcements[:50]:  # Check more items since we need to filter
                try:
                    # Check if this is for our company
                    scrip_name = item.get('SLONGNAME', '') or item.get('SCRIP_NAME', '')
                    news_sub = item.get('NEWSSUB', '') or item.get('NEWS_SUBJECT', '')

                    # Match by ticker or company name
                    if not (base_ticker.lower() in scrip_name.lower() or
                            company_name.lower() in scrip_name.lower()):
                        continue

                    # Parse date
                    pub_time = None
                    date_str = item.get('NEWS_DT') or item.get('DisssemDT')
                    if date_str:
                        try:
                            # BSE typically uses DD-Mon-YYYY or YYYY-MM-DD
                            for fmt in ['%d %b %Y', '%d-%b-%Y', '%Y-%m-%d', '%d/%m/%Y']:
                                try:
                                    pub_time = datetime.strptime(date_str.strip()[:11], fmt)
                                    break
                                except ValueError:
                                    continue
                        except Exception:
                            pass

                    if pub_time and pub_time < cutoff_date:
                        continue

                    category = item.get('CATEGORYNAME', 'Corporate')
                    headline = item.get('HEADLINE', '') or news_sub

                    news_items.append(NewsItem(
                        title=headline[:200],
                        description=f"{category}: {news_sub[:300]}",
                        source=f"BSE/{category}",
                        published_date=pub_time,
                        url="https://www.bseindia.com/corporates/ann.html",
                        ticker=ticker,
                    ))

                    if len(news_items) >= 10:  # Limit results
                        break

                except Exception as e:
                    self.logger.debug(f"Error parsing BSE item: {e}")
                    continue

        except Exception as e:
            self.logger.debug(f"Error fetching BSE announcements for {ticker}: {e}")

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

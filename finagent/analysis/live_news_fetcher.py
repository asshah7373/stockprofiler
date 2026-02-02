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

    # Company name mappings for better search (expanded list)
    COMPANY_NAMES = {
        # Nifty 50
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
        "LT": "Larsen Toubro",
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
        "NTPC": "NTPC Power",
        "POWERGRID": "Power Grid Corporation",
        "ONGC": "ONGC Oil Gas",
        "COALINDIA": "Coal India",
        "JSWSTEEL": "JSW Steel",
        "HINDALCO": "Hindalco Industries",
        "ULTRACEMCO": "UltraTech Cement",
        "GRASIM": "Grasim Industries",
        "TECHM": "Tech Mahindra",
        "DRREDDY": "Dr Reddy Laboratories",
        "CIPLA": "Cipla Pharma",
        "BRITANNIA": "Britannia Industries",
        "NESTLEIND": "Nestle India",
        "DIVISLAB": "Divi Laboratories",
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
        # Nifty Next 50 and Mid-cap
        "DMART": "Avenue Supermarts DMart",
        "ADANIGREEN": "Adani Green Energy",
        "AMBUJACEM": "Ambuja Cements",
        "BANKBARODA": "Bank of Baroda",
        "CANBK": "Canara Bank",
        "CHOLAFIN": "Cholamandalam Finance",
        "COLPAL": "Colgate Palmolive India",
        "DABUR": "Dabur India",
        "DLF": "DLF Real Estate",
        "GAIL": "GAIL India Gas",
        "GODREJCP": "Godrej Consumer Products",
        "HAVELLS": "Havells India",
        "INDIGO": "IndiGo Airlines InterGlobe",
        "JINDALSTEL": "Jindal Steel Power",
        "LICI": "LIC India Insurance",
        "MARICO": "Marico India",
        "MUTHOOTFIN": "Muthoot Finance",
        "NAUKRI": "Naukri InfoEdge",
        "PIDILITIND": "Pidilite Industries",
        "PNB": "Punjab National Bank",
        "SIEMENS": "Siemens India",
        "SRF": "SRF Limited",
        "TORNTPHARM": "Torrent Pharmaceuticals",
        "TRENT": "Trent Westside Zudio",
        "VEDL": "Vedanta Limited",
        "ETERNAL": "Zomato Eternal",
        "ZYDUSLIFE": "Zydus Lifesciences",
        # Popular Mid-caps
        "CARERATING": "CARE Ratings",
        "FEDERALBNK": "Federal Bank",
        "IDFCFIRSTB": "IDFC First Bank",
        "IRCTC": "IRCTC Railways",
        "HAL": "Hindustan Aeronautics HAL",
        "BEL": "Bharat Electronics BEL",
        "DIXON": "Dixon Technologies",
        "POLYCAB": "Polycab India",
        "PERSISTENT": "Persistent Systems",
        "COFORGE": "Coforge IT",
        "MPHASIS": "Mphasis IT",
        "LTIM": "LTIMindtree",
        "LTTS": "L&T Technology Services",
        "HAPPSTMNDS": "Happiest Minds Technologies",
        "KPITTECH": "KPIT Technologies",
        "TATAELXSI": "Tata Elxsi",
        "MINDACORP": "Minda Corporation",
        "EXIDEIND": "Exide Industries",
        "ESCORTS": "Escorts Kubota",
        "ASHOKLEY": "Ashok Leyland",
        "TVSMOTORS": "TVS Motor Company",
        "BIOCON": "Biocon Pharma",
        "LUPIN": "Lupin Pharma",
        "AUROPHARMA": "Aurobindo Pharma",
        "ALKEM": "Alkem Laboratories",
        "GLENMARK": "Glenmark Pharma",
        "GRANULES": "Granules India Pharma",
        "LALPATHLAB": "Dr Lal PathLabs",
        "METROPOLIS": "Metropolis Healthcare",
        "MAXHEALTH": "Max Healthcare",
        "FORTIS": "Fortis Healthcare",
        "JUBLFOOD": "Jubilant FoodWorks Dominos",
        "ZEEL": "Zee Entertainment",
        "PVRINOX": "PVR INOX Cinema",
        "TATAPOWER": "Tata Power",
        "ADANIPOWER": "Adani Power",
        "NHPC": "NHPC Hydro Power",
        "RECLTD": "REC Limited Power Finance",
        "PFC": "Power Finance Corporation",
        "IRFC": "Indian Railway Finance",
        "HUDCO": "HUDCO Housing",
        "SAIL": "Steel Authority of India SAIL",
        "NMDC": "NMDC Mining",
        "NATIONALUM": "National Aluminium NALCO",
        "HINDCOPPER": "Hindustan Copper",
        "COALINDIA": "Coal India",
        "RVNL": "Rail Vikas Nigam",
        "IRCON": "IRCON International Railways",
        "NBCC": "NBCC Construction",
        "SJVN": "SJVN Power",
        "MASTEK": "Mastek IT",
        "ECLERX": "eClerx Services",
        "CYIENT": "Cyient Engineering",
        "INFY": "Infosys",
        "GALLANTT": "Gallantt Ispat Steel",
        "INDOWIND": "Indo Wind Energy",
        "EMKAY": "Emkay Global Financial",
        "GRPLTD": "GRP Limited",
        "HCC": "Hindustan Construction Company",
        "HMVL": "HT Media",
        "INDIAMART": "IndiaMART InterMESH",
        "GLOBUSSPR": "Globus Spirits",
        "KAJARIACER": "Kajaria Ceramics",
        "HERCULES": "Hercules Hoists",
        "CAMLINFINE": "Camlin Fine Sciences",
        "AARTISURF": "Aarti Surfactants",
        "ASIANHOTNR": "Asian Hotels North",
        "GRSE": "Garden Reach Shipbuilders GRSE",
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
        if base_ticker in self.COMPANY_NAMES:
            return self.COMPANY_NAMES[base_ticker]

        # For unknown tickers, create a search-friendly name
        # Convert CAMELCASE or abbreviations to readable format
        # E.g., "CARERATING" -> "Care Rating", "FEDERALBNK" -> "Federal Bank"
        name = base_ticker

        # Common suffixes to expand
        suffixes = {
            'BNK': ' Bank', 'BANK': ' Bank', 'FIN': ' Finance',
            'TECH': ' Technologies', 'PHARMA': ' Pharma', 'IND': ' Industries',
            'LTD': '', 'AUTO': ' Auto', 'INFRA': ' Infrastructure',
            'POWER': ' Power', 'CEMENT': ' Cement', 'STEEL': ' Steel'
        }

        for suffix, replacement in suffixes.items():
            if name.endswith(suffix):
                name = name[:-len(suffix)] + replacement
                break

        # Add "India stock" to improve search relevance
        return f"{name} India"

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


# ============================================================================
# MACRO/POLICY NEWS AND SECTOR IMPACT ANALYSIS
# ============================================================================

# Sector-Policy Mappings: Keywords in news that benefit specific sectors
POLICY_SECTOR_MAPPINGS = {
    # Infrastructure & Construction
    "infrastructure": ["infra", "construction", "cement", "steel"],
    "highway": ["infra", "construction", "cement"],
    "smart city": ["infra", "realestate", "it"],
    "metro": ["infra", "construction", "capital_goods"],
    "railway": ["infra", "railway", "capital_goods"],
    "port": ["infra", "logistics", "shipping"],
    "airport": ["infra", "aviation"],

    # Energy & Power
    "solar": ["renewable", "power", "capital_goods"],
    "wind energy": ["renewable", "power"],
    "renewable energy": ["renewable", "power"],
    "green hydrogen": ["renewable", "power", "chemicals"],
    "electric vehicle": ["auto", "ev", "power"],
    "ev policy": ["auto", "ev", "battery"],
    "battery": ["ev", "chemicals", "power"],
    "power grid": ["power", "capital_goods"],

    # Technology & Digital
    "data center": ["it", "power", "realestate"],
    "digital india": ["it", "telecom", "fintech"],
    "semiconductor": ["it", "electronics", "capital_goods"],
    "chip manufacturing": ["it", "electronics", "capital_goods"],
    "5g": ["telecom", "it"],
    "ai artificial intelligence": ["it"],
    "cloud computing": ["it"],
    "cybersecurity": ["it"],

    # Manufacturing & PLI
    "pli scheme": ["manufacturing", "electronics", "pharma", "auto"],
    "make in india": ["manufacturing", "capital_goods"],
    "electronics manufacturing": ["electronics", "it"],
    "mobile manufacturing": ["electronics", "telecom"],
    "defence manufacturing": ["defence", "capital_goods"],
    "textile pli": ["textile"],
    "pharma pli": ["pharma"],
    "auto pli": ["auto"],

    # Commodities & Mining
    "rare earth": ["metals", "mining"],
    "lithium": ["metals", "ev", "battery"],
    "copper": ["metals", "power"],
    "steel duty": ["metals", "steel"],
    "iron ore": ["metals", "mining", "steel"],
    "coal": ["power", "mining", "metals"],
    "mining policy": ["metals", "mining"],
    "import duty metal": ["metals", "steel"],
    "export duty": ["metals", "chemicals"],

    # Financial & Banking
    "rbi policy": ["banking", "nbfc", "fintech"],
    "interest rate": ["banking", "nbfc", "realestate"],
    "repo rate": ["banking", "nbfc", "realestate"],
    "banking reform": ["banking", "nbfc"],
    "insurance": ["insurance"],
    "pension": ["insurance", "amc"],
    "gst": ["fmcg", "auto", "cement"],
    "tax cut": ["all"],
    "fiscal stimulus": ["infra", "banking", "fmcg"],

    # Agriculture & FMCG
    "msp": ["agri", "fmcg", "fertilizer"],
    "fertilizer subsidy": ["fertilizer", "agri", "chemicals"],
    "food processing": ["fmcg", "agri"],
    "rural development": ["agri", "fmcg", "tractor"],
    "irrigation": ["agri", "infra", "capital_goods"],
    "cold chain": ["agri", "fmcg", "logistics"],

    # Healthcare & Pharma
    "healthcare": ["pharma", "hospitals"],
    "ayushman bharat": ["pharma", "hospitals", "insurance"],
    "medical device": ["pharma", "healthcare"],
    "bulk drug": ["pharma", "chemicals"],
    "api manufacturing": ["pharma", "chemicals"],

    # Real Estate & Housing
    "housing for all": ["realestate", "cement", "steel"],
    "affordable housing": ["realestate", "cement", "nbfc"],
    "rera": ["realestate"],
    "stamp duty": ["realestate"],
    "real estate": ["realestate", "cement", "steel", "nbfc"],

    # Defence & Aerospace
    "defence budget": ["defence", "capital_goods"],
    "defence order": ["defence"],
    "atmanirbhar defence": ["defence", "capital_goods"],
    "aircraft": ["defence", "aviation"],
    "navy order": ["defence", "shipping"],

    # Environment & ESG
    "carbon credit": ["renewable", "power"],
    "pollution control": ["chemicals", "capital_goods"],
    "esg": ["all"],
    "green bond": ["renewable", "banking"],
}

# Sector to Stock Mappings (key stocks in each sector)
SECTOR_STOCKS = {
    "infra": ["LT.NS", "LTIM.NS", "KEC.NS", "KNRCON.NS", "IRB.NS", "NBCC.NS", "NCC.NS", "PNCINFRA.NS"],
    "construction": ["LT.NS", "NBCC.NS", "NCC.NS", "PNCINFRA.NS", "ASHOKA.NS", "CAPACITE.NS"],
    "cement": ["ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS", "RAMCOCEM.NS", "DALBHARAT.NS", "JKCEMENT.NS"],
    "steel": ["TATASTEEL.NS", "JSWSTEEL.NS", "JINDALSTEL.NS", "SAIL.NS", "NMDC.NS"],

    "power": ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "NHPC.NS", "SJVN.NS", "CESC.NS"],
    "renewable": ["ADANIGREEN.NS", "TATAPOWER.NS", "NHPC.NS", "SJVN.NS", "SWSOLAR.NS", "INOXGREEN.NS"],

    "it": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS", "COFORGE.NS", "PERSISTENT.NS", "MPHASIS.NS"],
    "telecom": ["BHARTIARTL.NS", "IDEA.NS", "INDUSTOWER.NS"],
    "electronics": ["DIXON.NS", "AMBER.NS", "KAYNES.NS", "TATAELXSI.NS", "DATAMATICS.NS"],

    "auto": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "ASHOKLEY.NS", "TVSMOTOR.NS"],
    "ev": ["TATAMOTORS.NS", "M&M.NS", "OLECTRA.NS", "TATAPOWER.NS", "EXIDEIND.NS"],
    "tractor": ["M&M.NS", "ESCORTS.NS"],

    "pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS", "AUROPHARMA.NS", "BIOCON.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS"],
    "hospitals": ["APOLLOHOSP.NS", "FORTIS.NS", "MAXHEALTH.NS", "MEDANTA.NS"],

    "banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS"],
    "nbfc": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "MANAPPURAM.NS", "LICHSGFIN.NS", "POONAWALLA.NS"],
    "fintech": ["PAYTM.NS", "POLICYBZR.NS"],
    "insurance": ["HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS", "ICICIGI.NS", "STARHEALTH.NS"],
    "amc": ["HDFCAMC.NS", "NAM-INDIA.NS", "UTIAMC.NS"],

    "fmcg": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS", "COLPAL.NS", "GODREJCP.NS", "TATACONSUM.NS", "VBL.NS"],

    "metals": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "JINDALSTEL.NS", "NMDC.NS", "NATIONALUM.NS", "HINDCOPPER.NS", "COALINDIA.NS"],
    "mining": ["NMDC.NS", "COALINDIA.NS", "VEDL.NS", "MOIL.NS", "HINDCOPPER.NS"],

    "chemicals": ["PIDILITIND.NS", "SRF.NS", "DEEPAKNTR.NS", "ATUL.NS", "NAVINFLUOR.NS", "ALKYLAMINE.NS", "CLEAN.NS", "AARTI.NS"],
    "fertilizer": ["CHAMBLFERT.NS", "COROMANDEL.NS", "GNFC.NS", "GSFC.NS", "DEEPAKFERT.NS", "FACT.NS"],

    "realestate": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "BRIGADE.NS", "SOBHA.NS", "PHOENIXLTD.NS", "LODHA.NS"],

    "defence": ["HAL.NS", "BEL.NS", "BDL.NS", "MAZAGON.NS", "GRSE.NS", "COCHINSHIP.NS", "MIDHANI.NS"],

    "capital_goods": ["LT.NS", "SIEMENS.NS", "ABB.NS", "BHEL.NS", "THERMAX.NS", "CGPOWER.NS", "CUMMINSIND.NS", "KEC.NS"],

    "logistics": ["CONCOR.NS", "DELHIVERY.NS", "MAHLOG.NS", "ALLCARGO.NS", "GATEWAY.NS", "VRLLOG.NS"],
    "shipping": ["COCHINSHIP.NS", "GRSE.NS", "GESHIP.NS"],
    "aviation": ["INDIGO.NS", "SPICEJET.NS"],
    "railway": ["IRCTC.NS", "IRFC.NS", "IRCON.NS", "RVNL.NS", "RAILTEL.NS", "TITAGARH.NS"],

    "textile": ["RAYMOND.NS", "ARVIND.NS", "WELSPUNIND.NS", "TRIDENT.NS", "KPRMILL.NS"],

    "agri": ["UPL.NS", "PIIND.NS", "BAYER.NS", "RALLIS.NS"],
    "battery": ["EXIDEIND.NS", "AMARAJABAT.NS"],
}


class MacroPolicyAnalyzer:
    """
    Analyzes macro/policy news and its impact on stocks.

    Fetches:
    - Government policy announcements
    - Budget news
    - RBI policy updates
    - Ministry announcements
    - Global macro events affecting Indian markets
    """

    # Cache for macro news
    _macro_cache: Dict[str, List[NewsItem]] = {}
    _macro_cache_time: Optional[datetime] = None
    _macro_cache_ttl = timedelta(hours=1)  # Cache macro news for 1 hour

    # Search queries for macro news
    MACRO_NEWS_QUERIES = [
        "India government policy economy",
        "India budget announcement",
        "RBI monetary policy India",
        "India ministry scheme announcement",
        "PLI scheme India",
        "India infrastructure investment",
        "India tax policy changes",
        "FII DII investment India",
    ]

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def fetch_macro_news(self, days_back: int = 7) -> List[NewsItem]:
        """Fetch macro/policy news from Google News RSS."""
        # Check cache
        if (self._macro_cache_time and
            datetime.now() - self._macro_cache_time < self._macro_cache_ttl and
            "macro" in self._macro_cache):
            return self._macro_cache["macro"]

        all_news = []

        try:
            import httpx
        except ImportError:
            self.logger.warning("httpx not installed, macro news disabled")
            return []

        for query in self.MACRO_NEWS_QUERIES[:4]:  # Limit queries to avoid rate limiting
            try:
                encoded_query = quote_plus(query)
                url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

                with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                    response = client.get(url)

                    if response.status_code == 200:
                        root = ET.fromstring(response.text)
                        items = root.findall('.//item')

                        cutoff_date = datetime.now() - timedelta(days=days_back)

                        for item in items[:10]:  # Take top 10 per query
                            title = item.find('title')
                            description = item.find('description')
                            pub_date = item.find('pubDate')
                            link = item.find('link')

                            if title is not None:
                                pub_datetime = None
                                if pub_date is not None and pub_date.text:
                                    try:
                                        pub_datetime = datetime.strptime(
                                            pub_date.text,
                                            "%a, %d %b %Y %H:%M:%S %Z"
                                        )
                                    except ValueError:
                                        pass

                                # Filter by date
                                if pub_datetime and pub_datetime < cutoff_date:
                                    continue

                                news_item = NewsItem(
                                    title=title.text or "",
                                    description=description.text if description is not None else "",
                                    source="Google News (Macro)",
                                    published_date=pub_datetime,
                                    url=link.text if link is not None else "",
                                    ticker="MACRO"
                                )
                                all_news.append(news_item)

                # Small delay between queries
                import time
                time.sleep(0.5)

            except Exception as e:
                self.logger.debug(f"Error fetching macro news for '{query}': {e}")

        # Deduplicate
        seen_titles = set()
        unique_news = []
        for item in all_news:
            title_key = item.title.lower()[:50]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(item)

        # Cache results
        self._macro_cache["macro"] = unique_news
        self._macro_cache_time = datetime.now()

        return unique_news

    def get_policy_impact(self, ticker: str, macro_news: List[NewsItem] = None) -> Dict[str, Any]:
        """
        Analyze if recent policy news benefits a specific stock.

        Returns:
            Dict with:
            - has_policy_boost: bool
            - policy_score: float (-50 to +50)
            - relevant_policies: List of policy headlines
            - affected_sectors: List of sectors
        """
        result = {
            "has_policy_boost": False,
            "policy_score": 0.0,
            "relevant_policies": [],
            "affected_sectors": [],
        }

        # Get macro news if not provided
        if macro_news is None:
            macro_news = self.fetch_macro_news()

        if not macro_news:
            return result

        # Normalize ticker
        base_ticker = ticker.replace(".NS", "").replace(".BO", "").upper()

        # Find which sectors this stock belongs to
        stock_sectors = []
        for sector, stocks in SECTOR_STOCKS.items():
            normalized_stocks = [s.replace(".NS", "").replace(".BO", "").upper() for s in stocks]
            if base_ticker in normalized_stocks:
                stock_sectors.append(sector)

        if not stock_sectors:
            return result

        result["affected_sectors"] = stock_sectors

        # Analyze each news item for policy keywords
        policy_score = 0.0
        relevant_policies = []

        for news in macro_news:
            news_text = (news.title + " " + news.description).lower()

            # Check each policy keyword
            for keyword, affected_sectors in POLICY_SECTOR_MAPPINGS.items():
                if keyword.lower() in news_text:
                    # Check if any of our stock's sectors are affected
                    matching_sectors = set(stock_sectors) & set(affected_sectors)
                    if matching_sectors or "all" in affected_sectors:
                        # Determine if positive or negative
                        positive_words = ["boost", "increase", "rise", "benefit", "growth",
                                        "approve", "launch", "invest", "expand", "cut tax",
                                        "reduce duty", "support", "promote", "incentive"]
                        negative_words = ["cut", "reduce", "decline", "restrict", "ban",
                                        "hike duty", "increase tax", "impose", "limit"]

                        is_positive = any(pw in news_text for pw in positive_words)
                        is_negative = any(nw in news_text for nw in negative_words)

                        if is_positive and not is_negative:
                            policy_score += 10.0
                            relevant_policies.append({
                                "headline": news.title[:100],
                                "keyword": keyword,
                                "sectors": list(matching_sectors) if matching_sectors else ["all"],
                                "sentiment": "POSITIVE"
                            })
                        elif is_negative and not is_positive:
                            policy_score -= 10.0
                            relevant_policies.append({
                                "headline": news.title[:100],
                                "keyword": keyword,
                                "sectors": list(matching_sectors) if matching_sectors else ["all"],
                                "sentiment": "NEGATIVE"
                            })

        # Cap the score
        result["policy_score"] = max(-50.0, min(50.0, policy_score))
        result["has_policy_boost"] = policy_score > 0
        result["relevant_policies"] = relevant_policies[:5]  # Top 5

        return result

    def get_fii_dii_sentiment(self) -> Dict[str, Any]:
        """Get overall FII/DII sentiment from news."""
        result = {
            "fii_sentiment": "NEUTRAL",
            "dii_sentiment": "NEUTRAL",
            "fii_score": 0.0,
            "dii_score": 0.0,
            "headlines": []
        }

        # Fetch FII/DII specific news
        try:
            import httpx

            query = "FII DII India stock market investment"
            encoded_query = quote_plus(query)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.get(url)

                if response.status_code == 200:
                    root = ET.fromstring(response.text)
                    items = root.findall('.//item')

                    fii_score = 0.0
                    dii_score = 0.0

                    for item in items[:15]:
                        title = item.find('title')
                        if title is not None and title.text:
                            text = title.text.lower()

                            # FII analysis
                            if "fii" in text or "foreign" in text:
                                if any(w in text for w in ["buy", "bought", "inflow", "bullish", "positive"]):
                                    fii_score += 10
                                    result["headlines"].append(f"FII+: {title.text[:80]}")
                                elif any(w in text for w in ["sell", "sold", "outflow", "bearish", "negative"]):
                                    fii_score -= 10
                                    result["headlines"].append(f"FII-: {title.text[:80]}")

                            # DII analysis
                            if "dii" in text or "domestic" in text:
                                if any(w in text for w in ["buy", "bought", "inflow", "bullish", "positive"]):
                                    dii_score += 10
                                    result["headlines"].append(f"DII+: {title.text[:80]}")
                                elif any(w in text for w in ["sell", "sold", "outflow", "bearish", "negative"]):
                                    dii_score -= 10
                                    result["headlines"].append(f"DII-: {title.text[:80]}")

                    # Determine sentiment
                    result["fii_score"] = max(-50, min(50, fii_score))
                    result["dii_score"] = max(-50, min(50, dii_score))

                    if fii_score > 10:
                        result["fii_sentiment"] = "BULLISH"
                    elif fii_score < -10:
                        result["fii_sentiment"] = "BEARISH"

                    if dii_score > 10:
                        result["dii_sentiment"] = "BULLISH"
                    elif dii_score < -10:
                        result["dii_sentiment"] = "BEARISH"

                    result["headlines"] = result["headlines"][:5]

        except Exception as e:
            self.logger.debug(f"Error fetching FII/DII news: {e}")

        return result


# Singleton instance
_macro_analyzer_instance = None


def get_macro_analyzer() -> MacroPolicyAnalyzer:
    """Get singleton MacroPolicyAnalyzer instance."""
    global _macro_analyzer_instance
    if _macro_analyzer_instance is None:
        _macro_analyzer_instance = MacroPolicyAnalyzer()
    return _macro_analyzer_instance


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

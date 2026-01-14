"""
Press Release Ingestion Pipeline
Fetches and parses press releases from company websites, BSE, and NSE.

Press releases provide vital information about:
- Corporate developments
- Business updates
- Strategic announcements
- Management commentary
- Investor communications
"""

import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
import json
import hashlib
from datetime import datetime, timedelta
import sqlite3
import re
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


@dataclass
class PressRelease:
    """Represents a press release document."""
    id: str
    ticker: str
    company_name: str
    title: str
    url: str
    source: str  # 'BSE', 'NSE', 'Company', 'PRNewswire', etc.
    release_date: str
    content: str
    summary: str
    categories: List[str]
    key_topics: List[str]
    sentiment_keywords: List[str]
    local_path: Optional[str] = None
    is_parsed: bool = False

    def to_dict(self) -> Dict:
        result = asdict(self)
        # Convert lists to JSON strings for SQLite
        result['categories'] = json.dumps(result['categories'])
        result['key_topics'] = json.dumps(result['key_topics'])
        result['sentiment_keywords'] = json.dumps(result['sentiment_keywords'])
        return result


class PressReleasePipeline:
    """
    Fetches and processes press releases from multiple sources.

    Sources:
    - BSE Press Release announcements
    - NSE Press Release announcements
    - PRNewswire India
    - Business Wire India
    - Company investor relations pages

    Features:
    - Content extraction from HTML and PDF
    - Key topic identification
    - Sentiment keyword extraction
    - RAG-ready chunking
    """

    # BSE Press Release endpoint
    BSE_PRESS_RELEASE_API = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"

    # NSE endpoints
    NSE_BASE_URL = "https://www.nseindia.com"
    NSE_ANNOUNCEMENTS_API = "https://www.nseindia.com/api/corporate-announcements"

    # News wire services
    PRNEWSWIRE_INDIA = "https://www.prnewswire.co.in"
    BUSINESSWIRE_INDIA = "https://www.businesswireindia.com"

    # Common company IR page patterns
    IR_PAGE_PATTERNS = [
        '/investors',
        '/investor-relations',
        '/ir',
        '/press-releases',
        '/media',
        '/news',
        '/announcements'
    ]

    # Keywords for categorization
    CATEGORY_KEYWORDS = {
        'financial_results': ['quarterly', 'annual', 'results', 'earnings', 'revenue', 'profit', 'loss'],
        'acquisition': ['acquisition', 'acquire', 'merger', 'takeover', 'buyout'],
        'partnership': ['partnership', 'collaboration', 'alliance', 'joint venture', 'mou'],
        'product_launch': ['launch', 'introduce', 'unveil', 'announce', 'new product'],
        'expansion': ['expansion', 'new plant', 'new facility', 'capacity', 'investment'],
        'management': ['appoint', 'resignation', 'ceo', 'cfo', 'director', 'management'],
        'dividend': ['dividend', 'payout', 'distribution'],
        'regulatory': ['sebi', 'rbi', 'regulatory', 'compliance', 'approval'],
        'sustainability': ['esg', 'sustainability', 'carbon', 'renewable', 'green'],
        'awards': ['award', 'recognition', 'ranked', 'best', 'top'],
    }

    # Sentiment keywords
    POSITIVE_KEYWORDS = [
        'growth', 'profit', 'increase', 'success', 'record', 'strong',
        'innovative', 'expand', 'milestone', 'award', 'leading', 'breakthrough'
    ]

    NEGATIVE_KEYWORDS = [
        'loss', 'decline', 'decrease', 'challenge', 'issue', 'concern',
        'delay', 'suspend', 'terminate', 'lawsuit', 'investigation'
    ]

    # Headers for BSE API
    BSE_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.bseindia.com/',
        'Origin': 'https://www.bseindia.com',
    }

    # Headers for NSE API
    # Note: Don't use 'br' (brotli) encoding - requires extra library
    NSE_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Referer': 'https://www.nseindia.com/',
        'Origin': 'https://www.nseindia.com',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
    }

    # General headers for web scraping
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    def __init__(
        self,
        data_dir: str = "data/press_releases",
        db_path: str = "data/cache/press_releases.db"
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.rate_limit = 1.0
        self.last_request_time = 0
        self.logger = logging.getLogger(__name__)
        self._nse_session = None
        self._init_db()

    def _init_db(self):
        """Initialize database for storing press releases."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS press_releases (
                id TEXT PRIMARY KEY,
                ticker TEXT,
                company_name TEXT,
                title TEXT NOT NULL,
                url TEXT,
                source TEXT,
                release_date TEXT,
                content TEXT,
                summary TEXT,
                categories TEXT,
                key_topics TEXT,
                sentiment_keywords TEXT,
                local_path TEXT,
                is_parsed INTEGER DEFAULT 0,
                fetched_at TEXT NOT NULL
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pr_ticker ON press_releases(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pr_date ON press_releases(release_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pr_source ON press_releases(source)")

        conn.commit()
        conn.close()

    def _rate_limit_wait(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _generate_id(self, url: str, title: str, date: str = "") -> str:
        """Generate unique ID for press release."""
        content = f"{url}:{title}:{date}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _get_nse_session(self) -> requests.Session:
        """Get NSE session with cookies."""
        if self._nse_session is None:
            self._nse_session = requests.Session()
            self._nse_session.headers.update(self.NSE_HEADERS)
            try:
                self._rate_limit_wait()
                response = self._nse_session.get(self.NSE_BASE_URL, timeout=10)
                self.logger.debug(f"NSE session initialized, status: {response.status_code}, cookies: {len(self._nse_session.cookies)}")

                # Sometimes need additional request to get cookies
                if len(self._nse_session.cookies) == 0:
                    self._rate_limit_wait()
                    self._nse_session.get("https://www.nseindia.com/api/marketStatus", timeout=10)
            except Exception as e:
                self.logger.warning(f"NSE session init error: {e}")
        return self._nse_session

    def _reset_nse_session(self):
        """Reset NSE session."""
        self._nse_session = None

    # =========================================================================
    # BSE Press Releases
    # =========================================================================

    def fetch_bse_press_releases(
        self,
        ticker: Optional[str] = None,
        days_back: int = 30
    ) -> List[PressRelease]:
        """
        Fetch press releases from BSE.

        BSE categorizes press releases under specific announcement types.
        """
        releases = []

        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
        to_date = datetime.now().strftime('%Y%m%d')

        params = {
            'strCat': 'Press Release',  # Filter for press releases
            'strPrevDate': from_date,
            'strScrip': ticker.replace('.BO', '').upper() if ticker else '',
            'strSearch': 'P',
            'strToDate': to_date,
            'strType': 'C'
        }

        try:
            self._rate_limit_wait()

            response = requests.get(
                self.BSE_PRESS_RELEASE_API,
                headers=self.BSE_HEADERS,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"BSE response not JSON. First 200 chars: {response.text[:200]}")
                    return releases

                for item in data.get('Table', []):
                    try:
                        news_dt = item.get('NEWS_DT', '')
                        if news_dt:
                            try:
                                release_date = datetime.strptime(
                                    news_dt.split('T')[0], '%Y-%m-%d'
                                ).isoformat()
                            except ValueError:
                                release_date = news_dt

                        # Build URL
                        attachment = item.get('ATTACHMENTNAME', '')
                        if attachment and not attachment.startswith('http'):
                            url = f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attachment}"
                        else:
                            url = attachment

                        title = item.get('HEADLINE', '')

                        pr = PressRelease(
                            id=self._generate_id(url, title, news_dt),
                            ticker=item.get('SCRIP_CD', ''),
                            company_name=item.get('SLONGNAME', ''),
                            title=title,
                            url=url,
                            source='BSE',
                            release_date=release_date,
                            content='',  # To be fetched
                            summary=item.get('MORE', ''),
                            categories=self._categorize_content(title),
                            key_topics=self._extract_key_topics(title),
                            sentiment_keywords=self._extract_sentiment_keywords(title)
                        )
                        releases.append(pr)

                    except Exception as e:
                        self.logger.warning(f"Error parsing BSE press release: {e}")

                self.logger.info(f"Fetched {len(releases)} BSE press releases")

        except requests.RequestException as e:
            self.logger.error(f"Error fetching BSE press releases: {e}")

        return releases

    # =========================================================================
    # NSE Press Releases
    # =========================================================================

    def fetch_nse_press_releases(
        self,
        ticker: Optional[str] = None,
        days_back: int = 30
    ) -> List[PressRelease]:
        """
        Fetch press releases from NSE corporate announcements.
        """
        releases = []
        session = self._get_nse_session()

        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%d-%m-%Y')
        to_date = datetime.now().strftime('%d-%m-%Y')

        params = {
            'index': 'equities',
            'from_date': from_date,
            'to_date': to_date
        }

        if ticker:
            params['symbol'] = ticker.replace('.NS', '').upper()

        try:
            self._rate_limit_wait()

            response = session.get(
                self.NSE_ANNOUNCEMENTS_API,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"NSE response not JSON. First 200 chars: {response.text[:200]}")
                    self._reset_nse_session()
                    return releases

                # Handle non-list responses
                if not isinstance(data, list):
                    self.logger.warning(f"NSE returned unexpected data type: {type(data)}")
                    return releases

                for item in data:
                    desc = (item.get('desc', '') or item.get('subject', '')).lower()

                    # Filter for press releases
                    if not any(kw in desc for kw in ['press release', 'media release', 'press statement']):
                        continue

                    try:
                        broadcast_dt = item.get('an_dt', '') or item.get('sort_date', '')
                        if broadcast_dt:
                            try:
                                release_date = datetime.strptime(
                                    broadcast_dt.split(' ')[0], '%d-%b-%Y'
                                ).isoformat()
                            except ValueError:
                                release_date = broadcast_dt

                        attachment = item.get('attchmntFile', '')
                        if attachment and not attachment.startswith('http'):
                            url = f"https://www.nseindia.com/api/corporate-announcements/download?fileName={attachment}"
                        else:
                            url = attachment

                        title = item.get('desc', '') or item.get('subject', '')

                        pr = PressRelease(
                            id=self._generate_id(url, title, broadcast_dt),
                            ticker=item.get('symbol', ''),
                            company_name=item.get('sm_name', ''),
                            title=title,
                            url=url,
                            source='NSE',
                            release_date=release_date,
                            content='',
                            summary=item.get('details', ''),
                            categories=self._categorize_content(title),
                            key_topics=self._extract_key_topics(title),
                            sentiment_keywords=self._extract_sentiment_keywords(title)
                        )
                        releases.append(pr)

                    except Exception as e:
                        self.logger.warning(f"Error parsing NSE press release: {e}")

                self.logger.info(f"Fetched {len(releases)} NSE press releases")

        except requests.RequestException as e:
            self.logger.error(f"Error fetching NSE press releases: {e}")

        return releases

    # =========================================================================
    # Company Website Scraping
    # =========================================================================

    def fetch_company_press_releases(
        self,
        company_website: str,
        ticker: str,
        company_name: str,
        max_releases: int = 20
    ) -> List[PressRelease]:
        """
        Scrape press releases from a company's investor relations page.

        This is a best-effort scraper that works with common IR page structures.
        """
        releases = []

        # Try to find the press release page
        press_release_url = self._find_press_release_page(company_website)
        if not press_release_url:
            self.logger.warning(f"Could not find press release page for {company_website}")
            return releases

        try:
            self._rate_limit_wait()

            response = requests.get(
                press_release_url,
                headers=self.HEADERS,
                timeout=15
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for common press release list patterns
                pr_links = self._find_press_release_links(soup, press_release_url)

                for link_info in pr_links[:max_releases]:
                    try:
                        pr = PressRelease(
                            id=self._generate_id(link_info['url'], link_info['title']),
                            ticker=ticker,
                            company_name=company_name,
                            title=link_info['title'],
                            url=link_info['url'],
                            source='Company',
                            release_date=link_info.get('date', ''),
                            content='',
                            summary='',
                            categories=self._categorize_content(link_info['title']),
                            key_topics=self._extract_key_topics(link_info['title']),
                            sentiment_keywords=self._extract_sentiment_keywords(link_info['title'])
                        )
                        releases.append(pr)

                    except Exception as e:
                        self.logger.warning(f"Error creating press release: {e}")

                self.logger.info(f"Found {len(releases)} press releases from {company_website}")

        except requests.RequestException as e:
            self.logger.error(f"Error fetching from {press_release_url}: {e}")

        return releases

    def _find_press_release_page(self, base_url: str) -> Optional[str]:
        """Find the press release or news page on a company website."""
        parsed = urlparse(base_url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        for pattern in self.IR_PAGE_PATTERNS:
            url = urljoin(base, pattern)
            try:
                self._rate_limit_wait()
                response = requests.head(url, headers=self.HEADERS, timeout=5, allow_redirects=True)
                if response.status_code == 200:
                    return url
            except requests.RequestException:
                continue

        return None

    def _find_press_release_links(
        self,
        soup: BeautifulSoup,
        base_url: str
    ) -> List[Dict]:
        """Extract press release links from a page."""
        links = []

        # Common selectors for press release lists
        selectors = [
            'article a',
            '.press-release a',
            '.news-item a',
            '.announcement a',
            'ul.news li a',
            'div.press a',
            '.media-release a',
        ]

        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                for elem in elements:
                    href = elem.get('href', '')
                    title = elem.get_text(strip=True)

                    if href and title and len(title) > 10:
                        full_url = urljoin(base_url, href)
                        links.append({
                            'url': full_url,
                            'title': title,
                            'date': self._extract_date_from_context(elem)
                        })

        # If no specific selector worked, try generic link finding
        if not links:
            for a_tag in soup.find_all('a', href=True):
                text = a_tag.get_text(strip=True)
                href = a_tag['href']

                # Filter for likely press release links
                if any(kw in text.lower() for kw in ['press', 'release', 'announcement', 'news']):
                    if len(text) > 20:
                        full_url = urljoin(base_url, href)
                        links.append({
                            'url': full_url,
                            'title': text,
                            'date': ''
                        })

        return links

    def _extract_date_from_context(self, element) -> str:
        """Try to extract date from nearby elements."""
        # Check parent and siblings for date-like content
        parent = element.parent
        if parent:
            text = parent.get_text()
            # Simple date pattern matching
            date_patterns = [
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',
            ]
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group()
        return ''

    # =========================================================================
    # Content Processing
    # =========================================================================

    def fetch_press_release_content(self, pr: PressRelease) -> PressRelease:
        """Fetch and parse the full content of a press release."""
        if not pr.url:
            return pr

        try:
            self._rate_limit_wait()

            # Check if it's a PDF
            if pr.url.lower().endswith('.pdf') or 'pdf' in pr.url.lower():
                content = self._fetch_pdf_content(pr.url)
            else:
                content = self._fetch_html_content(pr.url)

            if content:
                pr.content = content
                pr.summary = self._generate_summary(content)
                pr.categories = self._categorize_content(content)
                pr.key_topics = self._extract_key_topics(content)
                pr.sentiment_keywords = self._extract_sentiment_keywords(content)
                pr.is_parsed = True

        except Exception as e:
            self.logger.error(f"Error fetching press release content: {e}")

        return pr

    def _fetch_html_content(self, url: str) -> str:
        """Fetch and extract content from HTML page."""
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=15)
            if response.status_code != 200:
                return ''

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()

            # Try to find main content
            content_selectors = [
                'article',
                '.content',
                '.main-content',
                '.press-release-content',
                '.news-content',
                'main',
                '#content',
            ]

            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    text = content_elem.get_text(separator='\n', strip=True)
                    if len(text) > 200:
                        return self._clean_text(text)

            # Fallback to body
            body = soup.body
            if body:
                return self._clean_text(body.get_text(separator='\n', strip=True))

            return ''

        except Exception as e:
            self.logger.error(f"Error fetching HTML: {e}")
            return ''

    def _fetch_pdf_content(self, url: str) -> str:
        """Download and extract text from PDF."""
        try:
            # Download PDF
            response = requests.get(url, headers=self.HEADERS, timeout=30, stream=True)
            if response.status_code != 200:
                return ''

            # Save temporarily
            temp_path = self.data_dir / f"temp_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract text
            try:
                from pypdf import PdfReader
                reader = PdfReader(temp_path)
                text_parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                content = '\n'.join(text_parts)
            except ImportError:
                content = ''
            finally:
                # Clean up
                temp_path.unlink(missing_ok=True)

            return self._clean_text(content)

        except Exception as e:
            self.logger.error(f"Error fetching PDF: {e}")
            return ''

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        # Remove very short lines (likely navigation/headers)
        lines = [line for line in text.split('\n') if len(line.strip()) > 20 or line.strip() == '']
        return '\n'.join(lines).strip()

    def _generate_summary(self, content: str, max_length: int = 500) -> str:
        """Generate a summary from content."""
        if not content:
            return ''

        # Take first few sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        summary = ''
        for sentence in sentences:
            if len(summary) + len(sentence) < max_length:
                summary += sentence + ' '
            else:
                break

        return summary.strip()

    def _categorize_content(self, text: str) -> List[str]:
        """Categorize content based on keywords."""
        categories = []
        text_lower = text.lower()

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                categories.append(category)

        return categories if categories else ['general']

    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from text."""
        topics = []
        text_lower = text.lower()

        # Extract mentioned numbers/figures
        money_pattern = r'(?:Rs\.?|INR|USD|\$)\s*[\d,]+(?:\.\d+)?(?:\s*(?:cr|crore|lakh|million|billion))?'
        money_matches = re.findall(money_pattern, text, re.IGNORECASE)
        if money_matches:
            topics.extend([f"Amount: {m}" for m in money_matches[:3]])

        # Extract percentage changes
        pct_pattern = r'(\d+(?:\.\d+)?)\s*(?:%|percent)'
        pct_matches = re.findall(pct_pattern, text)
        if pct_matches:
            topics.append(f"Percentages mentioned: {', '.join(pct_matches[:3])}")

        # Extract company names mentioned (capitalized words)
        company_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        companies = re.findall(company_pattern, text)
        if companies:
            unique_companies = list(set(companies))[:3]
            topics.extend([f"Entity: {c}" for c in unique_companies])

        return topics

    def _extract_sentiment_keywords(self, text: str) -> List[str]:
        """Extract sentiment-indicating keywords."""
        text_lower = text.lower()
        found = []

        for kw in self.POSITIVE_KEYWORDS:
            if kw in text_lower:
                found.append(f"+{kw}")

        for kw in self.NEGATIVE_KEYWORDS:
            if kw in text_lower:
                found.append(f"-{kw}")

        return found

    # =========================================================================
    # Combined Operations
    # =========================================================================

    def fetch_all_press_releases(
        self,
        ticker: Optional[str] = None,
        days_back: int = 30
    ) -> List[PressRelease]:
        """
        Fetch press releases from all sources.
        """
        all_releases = []

        # BSE
        bse_releases = self.fetch_bse_press_releases(ticker, days_back)
        all_releases.extend(bse_releases)

        # NSE
        nse_releases = self.fetch_nse_press_releases(ticker, days_back)
        all_releases.extend(nse_releases)

        # Deduplicate
        seen_ids = set()
        unique_releases = []
        for pr in all_releases:
            if pr.id not in seen_ids:
                seen_ids.add(pr.id)
                unique_releases.append(pr)

        # Sort by date
        unique_releases.sort(
            key=lambda x: x.release_date if x.release_date else '',
            reverse=True
        )

        return unique_releases

    def process_press_release(
        self,
        pr: PressRelease,
        fetch_content: bool = True
    ) -> Tuple[PressRelease, List[Dict]]:
        """
        Process a press release:
        1. Fetch full content (if requested)
        2. Generate chunks for RAG
        3. Save to database
        """
        if fetch_content and pr.url:
            pr = self.fetch_press_release_content(pr)

        # Generate chunks
        chunks = self._chunk_for_rag(pr)

        # Save to database
        self.save_press_release(pr)

        return pr, chunks

    def _chunk_for_rag(
        self,
        pr: PressRelease,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Dict]:
        """Chunk press release content for RAG."""
        chunks = []
        text = pr.content or pr.summary or pr.title

        if not text:
            return chunks

        # Clean and split
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        chunk_index = 0

        metadata = {
            "doc_id": pr.id,
            "ticker": pr.ticker,
            "company_name": pr.company_name,
            "source": pr.source,
            "release_date": pr.release_date,
            "categories": pr.categories,
            "doc_type": "press_release"
        }

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": {**metadata, "chunk_index": chunk_index},
                        "chunk_index": chunk_index
                    })
                    chunk_index += 1

                if overlap > 0 and chunks:
                    overlap_text = chunks[-1]["text"][-overlap:]
                    current_chunk = overlap_text + " " + sentence + " "
                else:
                    current_chunk = sentence + " "

        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": {**metadata, "chunk_index": chunk_index},
                "chunk_index": chunk_index
            })

        return chunks

    def save_press_release(self, pr: PressRelease):
        """Save press release to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        pr_dict = pr.to_dict()

        cursor.execute("""
            INSERT OR REPLACE INTO press_releases
            (id, ticker, company_name, title, url, source, release_date,
             content, summary, categories, key_topics, sentiment_keywords,
             local_path, is_parsed, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pr_dict['id'],
            pr_dict['ticker'],
            pr_dict['company_name'],
            pr_dict['title'],
            pr_dict['url'],
            pr_dict['source'],
            pr_dict['release_date'],
            pr_dict['content'],
            pr_dict['summary'],
            pr_dict['categories'],
            pr_dict['key_topics'],
            pr_dict['sentiment_keywords'],
            pr_dict['local_path'],
            1 if pr_dict['is_parsed'] else 0,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def search_press_releases(
        self,
        query: Optional[str] = None,
        ticker: Optional[str] = None,
        source: Optional[str] = None,
        days_back: int = 30
    ) -> List[Dict]:
        """Search stored press releases."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        sql = "SELECT * FROM press_releases WHERE 1=1"
        params = []

        if ticker:
            sql += " AND ticker LIKE ?"
            params.append(f"%{ticker.replace('.NS', '').replace('.BO', '')}%")

        if source:
            sql += " AND source = ?"
            params.append(source)

        if days_back:
            cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
            sql += " AND release_date >= ?"
            params.append(cutoff)

        if query:
            sql += " AND (title LIKE ? OR content LIKE ? OR summary LIKE ?)"
            params.extend([f"%{query}%"] * 3)

        sql += " ORDER BY release_date DESC LIMIT 100"

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def document_exists(self, doc_id: str) -> bool:
        """Check if a press release already exists in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM press_releases WHERE id = ?", (doc_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def get_existing_doc_ids(self) -> set:
        """Get set of all existing press release IDs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM press_releases")
        ids = {row[0] for row in cursor.fetchall()}
        conn.close()
        return ids

    def filter_new_documents(
        self,
        releases: List[PressRelease]
    ) -> List[PressRelease]:
        """Filter out press releases that already exist in the database."""
        existing_ids = self.get_existing_doc_ids()
        new_releases = [pr for pr in releases if pr.id not in existing_ids]
        skipped = len(releases) - len(new_releases)
        if skipped > 0:
            self.logger.info(f"Skipped {skipped} already-ingested press releases")
        return new_releases

    def get_stats(self) -> Dict:
        """Get press release statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM press_releases")
        stats['total'] = cursor.fetchone()[0]

        cursor.execute("SELECT source, COUNT(*) FROM press_releases GROUP BY source")
        stats['by_source'] = dict(cursor.fetchall())

        cursor.execute("SELECT is_parsed, COUNT(*) FROM press_releases GROUP BY is_parsed")
        parsed = dict(cursor.fetchall())
        stats['parsed'] = parsed.get(1, 0)
        stats['unparsed'] = parsed.get(0, 0)

        conn.close()

        return stats

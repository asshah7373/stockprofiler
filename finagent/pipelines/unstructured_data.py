"""
Unstructured Data Pipeline
Fetches and parses BSE/NSE circulars, corporate filings, and announcements.

This module provides comprehensive ingestion of:
- BSE Corporate Announcements
- NSE Corporate Filings
- Exchange Circulars
- Quarterly Results
- Board Meeting Outcomes
- Shareholding Patterns
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
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Types of documents from exchanges."""
    CIRCULAR = "circular"
    QUARTERLY_RESULTS = "quarterly_results"
    ANNUAL_REPORT = "annual_report"
    BOARD_MEETING = "board_meeting"
    SHAREHOLDING = "shareholding_pattern"
    CORPORATE_ACTION = "corporate_action"
    ANNOUNCEMENT = "announcement"
    AGM_EGM = "agm_egm"
    INVESTOR_PRESENTATION = "investor_presentation"
    CREDIT_RATING = "credit_rating"
    INSIDER_TRADING = "insider_trading"
    PRESS_RELEASE = "press_release"


@dataclass
class CircularDocument:
    """Represents a circular/filing document."""
    id: str
    ticker: str
    company_name: str
    title: str
    url: str
    doc_type: str
    exchange: str
    filing_date: str
    category: str
    subcategory: str
    description: str
    attachment_name: str
    file_size: Optional[int] = None
    local_path: Optional[str] = None
    parsed: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


class CircularPipeline:
    """
    Scrapes, downloads, and parses regulatory circulars.

    Sources:
    - BSE Corporate Announcements API
    - NSE Corporate Filings API
    - Exchange Circulars

    Rate Limiting:
    - Respects 1 request per second limit
    - Uses session cookies for NSE
    """

    # BSE API Endpoints
    BSE_ANNOUNCEMENTS_API = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
    BSE_CORPORATE_API = "https://api.bseindia.com/BseIndiaAPI/api/CorporateAction/w"
    BSE_RESULT_API = "https://api.bseindia.com/BseIndiaAPI/api/FinancialResults/w"

    # NSE API Endpoints
    NSE_BASE_URL = "https://www.nseindia.com"
    NSE_ANNOUNCEMENTS_API = "https://www.nseindia.com/api/corporate-announcements"
    NSE_BOARD_MEETINGS_API = "https://www.nseindia.com/api/corporate-board-meetings"
    NSE_FINANCIAL_RESULTS_API = "https://www.nseindia.com/api/corporates-financial-results"
    NSE_ACTIONS_API = "https://www.nseindia.com/api/corporates-corporateActions"
    NSE_SHAREHOLDING_API = "https://www.nseindia.com/api/corporate-shareholding"

    # Document category mappings
    BSE_CATEGORIES = {
        'Result': DocumentType.QUARTERLY_RESULTS,
        'AGM/EGM': DocumentType.AGM_EGM,
        'Board Meeting': DocumentType.BOARD_MEETING,
        'Acquisition': DocumentType.CORPORATE_ACTION,
        'Dividend': DocumentType.CORPORATE_ACTION,
        'Bonus': DocumentType.CORPORATE_ACTION,
        'Split': DocumentType.CORPORATE_ACTION,
        'Rights': DocumentType.CORPORATE_ACTION,
        'Insider Trading': DocumentType.INSIDER_TRADING,
        'Credit Rating': DocumentType.CREDIT_RATING,
        'Shareholding': DocumentType.SHAREHOLDING,
        'Press Release': DocumentType.PRESS_RELEASE,
        'Investor Presentation': DocumentType.INVESTOR_PRESENTATION,
    }

    # Headers to mimic browser request for NSE
    # Note: Don't use 'br' (brotli) encoding - requires extra library
    NSE_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Referer': 'https://www.nseindia.com/',
        'Origin': 'https://www.nseindia.com',
        'Connection': 'keep-alive',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
    }

    # Headers for BSE API
    BSE_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.bseindia.com/',
        'Origin': 'https://www.bseindia.com',
        'Connection': 'keep-alive',
    }

    # Alias for backwards compatibility
    HEADERS = NSE_HEADERS

    def __init__(
        self,
        data_dir: str = "data/circulars",
        db_path: str = "data/cache/circulars.db"
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.rate_limit = 1.0  # seconds between requests
        self.last_request_time = 0
        self.logger = logging.getLogger(__name__)
        self._nse_session = None
        self._init_db()

    def _init_db(self):
        """Initialize database for storing circular metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS circulars (
                id TEXT PRIMARY KEY,
                ticker TEXT,
                company_name TEXT,
                title TEXT NOT NULL,
                url TEXT,
                doc_type TEXT,
                exchange TEXT,
                filing_date TEXT,
                category TEXT,
                subcategory TEXT,
                description TEXT,
                attachment_name TEXT,
                file_size INTEGER,
                local_path TEXT,
                parsed_text TEXT,
                tables_json TEXT,
                key_figures_json TEXT,
                embedding_id TEXT,
                fetched_at TEXT NOT NULL,
                parsed_at TEXT,
                is_processed INTEGER DEFAULT 0
            )
        """)

        # Index for efficient queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_circulars_ticker ON circulars(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_circulars_date ON circulars(filing_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_circulars_type ON circulars(doc_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_circulars_exchange ON circulars(exchange)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_circulars_category ON circulars(category)")

        conn.commit()
        conn.close()

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _generate_doc_id(self, url: str, title: str, date: str = "") -> str:
        """Generate unique document ID."""
        content = f"{url}:{title}:{date}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _get_nse_session(self) -> requests.Session:
        """Get or create NSE session with required cookies."""
        if self._nse_session is None:
            self._nse_session = requests.Session()
            self._nse_session.headers.update(self.NSE_HEADERS)

            try:
                # Visit main page to get cookies - NSE requires this
                self._rate_limit_wait()
                response = self._nse_session.get(
                    self.NSE_BASE_URL,
                    timeout=10
                )
                self.logger.debug(f"NSE session initialized, status: {response.status_code}, cookies: {len(self._nse_session.cookies)}")

                # Sometimes need to visit an API endpoint to fully initialize
                if len(self._nse_session.cookies) == 0:
                    self._rate_limit_wait()
                    self._nse_session.get(
                        "https://www.nseindia.com/api/marketStatus",
                        timeout=10
                    )
                    self.logger.debug(f"NSE session retry, cookies: {len(self._nse_session.cookies)}")

            except Exception as e:
                self.logger.warning(f"Error initializing NSE session: {e}")

        return self._nse_session

    def _reset_nse_session(self):
        """Reset NSE session if it becomes invalid."""
        self._nse_session = None

    # =========================================================================
    # BSE Data Fetching
    # =========================================================================

    def fetch_bse_announcements(
        self,
        ticker: Optional[str] = None,
        days_back: int = 7,
        category: Optional[str] = None
    ) -> List[CircularDocument]:
        """
        Fetch announcements from BSE API.

        Args:
            ticker: Stock code (without .BO suffix)
            days_back: Number of days to look back
            category: Filter by category (e.g., 'Result', 'Board Meeting')

        Returns:
            List of CircularDocument objects
        """
        documents = []

        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
        to_date = datetime.now().strftime('%Y%m%d')

        params = {
            'strCat': category if category else '-1',
            'strPrevDate': from_date,
            'strScrip': ticker.replace('.BO', '').upper() if ticker else '',
            'strSearch': 'P',
            'strToDate': to_date,
            'strType': 'C'
        }

        try:
            self._rate_limit_wait()

            response = requests.get(
                self.BSE_ANNOUNCEMENTS_API,
                headers=self.BSE_HEADERS,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                # Check if response is actually JSON
                content_type = response.headers.get('Content-Type', '')
                if 'json' not in content_type.lower():
                    self.logger.warning(f"BSE returned non-JSON response: {content_type}")
                    # Try parsing anyway in case content-type header is wrong
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        self.logger.error(f"BSE response not JSON. First 200 chars: {response.text[:200]}")
                        return documents
                else:
                    data = response.json()

                for item in data.get('Table', []):
                    try:
                        # Parse date
                        news_dt = item.get('NEWS_DT', '')
                        filing_date = ''
                        if news_dt:
                            try:
                                filing_date = datetime.strptime(
                                    news_dt.split('T')[0], '%Y-%m-%d'
                                ).isoformat()
                            except ValueError:
                                filing_date = news_dt

                        # Determine document type - handle None
                        cat_name = item.get('CATEGORYNAME') or ''
                        doc_type = self._categorize_bse_document(cat_name)

                        # Build attachment URL
                        attachment = item.get('ATTACHMENTNAME') or ''
                        if attachment and not attachment.startswith('http'):
                            attachment = f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attachment}"

                        # SCRIP_CD is an integer, convert to string
                        scrip_cd = item.get('SCRIP_CD')
                        ticker_str = str(scrip_cd) if scrip_cd is not None else ''

                        doc = CircularDocument(
                            id=self._generate_doc_id(
                                attachment,
                                item.get('HEADLINE') or '',
                                news_dt
                            ),
                            ticker=ticker_str,
                            company_name=item.get('SLONGNAME') or item.get('NSURL') or '',
                            title=item.get('HEADLINE') or '',
                            url=attachment,
                            doc_type=doc_type.value,
                            exchange='BSE',
                            filing_date=filing_date,
                            category=cat_name,
                            subcategory=item.get('SUBCATNAME') or '',
                            description=item.get('MORE') or '',
                            attachment_name=item.get('ATTACHMENTNAME') or ''
                        )
                        documents.append(doc)

                    except Exception as e:
                        self.logger.warning(f"Error parsing BSE announcement: {e}")
                        continue

                self.logger.info(f"Fetched {len(documents)} BSE announcements")

            else:
                self.logger.error(f"BSE API error: {response.status_code}")

        except requests.RequestException as e:
            self.logger.error(f"Error fetching BSE announcements: {e}")

        return documents

    def fetch_bse_financial_results(
        self,
        ticker: Optional[str] = None,
        days_back: int = 90
    ) -> List[CircularDocument]:
        """Fetch quarterly/annual financial results from BSE."""
        return self.fetch_bse_announcements(
            ticker=ticker,
            days_back=days_back,
            category='Result'
        )

    def fetch_bse_board_meetings(
        self,
        ticker: Optional[str] = None,
        days_back: int = 30
    ) -> List[CircularDocument]:
        """Fetch board meeting outcomes from BSE."""
        return self.fetch_bse_announcements(
            ticker=ticker,
            days_back=days_back,
            category='Board Meeting'
        )

    def _categorize_bse_document(self, category_name: str) -> DocumentType:
        """Map BSE category to DocumentType."""
        if not category_name:
            return DocumentType.ANNOUNCEMENT
        category_name = category_name.strip()

        for key, doc_type in self.BSE_CATEGORIES.items():
            if key.lower() in category_name.lower():
                return doc_type

        return DocumentType.ANNOUNCEMENT

    # =========================================================================
    # NSE Data Fetching
    # =========================================================================

    def fetch_nse_announcements(
        self,
        ticker: Optional[str] = None,
        days_back: int = 7,
        index: str = "equities"
    ) -> List[CircularDocument]:
        """
        Fetch corporate announcements from NSE API.

        Args:
            ticker: Stock symbol (without .NS suffix)
            days_back: Number of days to look back
            index: Index type ('equities', 'sme', 'debt')

        Returns:
            List of CircularDocument objects
        """
        documents = []
        session = self._get_nse_session()

        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%d-%m-%Y')
        to_date = datetime.now().strftime('%d-%m-%Y')

        params = {
            'index': index,
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
                # Check if response is actually JSON
                content_type = response.headers.get('Content-Type', '')
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"NSE response not JSON. First 200 chars: {response.text[:200]}")
                    # Reset session and retry once
                    self._reset_nse_session()
                    return documents

                # Handle case where data is a dict with error message
                if isinstance(data, dict) and 'error' in data:
                    self.logger.error(f"NSE API error: {data.get('error')}")
                    return documents

                # Ensure data is a list
                if not isinstance(data, list):
                    self.logger.warning(f"NSE returned unexpected data type: {type(data)}")
                    return documents

                for item in data:
                    try:
                        # Parse date
                        broadcast_dt = item.get('an_dt', '') or item.get('sort_date', '')
                        if broadcast_dt:
                            try:
                                filing_date = datetime.strptime(
                                    broadcast_dt.split(' ')[0], '%d-%b-%Y'
                                ).isoformat()
                            except ValueError:
                                filing_date = broadcast_dt

                        # Get attachment URL
                        attachment = item.get('attchmntFile', '') or item.get('attachment', '')
                        if attachment and not attachment.startswith('http'):
                            attachment = f"https://www.nseindia.com/api/corporate-announcements/download?fileName={attachment}"

                        # Determine document type
                        subject = item.get('desc', '') or item.get('subject', '')
                        doc_type = self._categorize_nse_document(subject)

                        doc = CircularDocument(
                            id=self._generate_doc_id(
                                attachment,
                                subject,
                                broadcast_dt
                            ),
                            ticker=item.get('symbol', ''),
                            company_name=item.get('sm_name', '') or item.get('companyName', ''),
                            title=subject,
                            url=attachment,
                            doc_type=doc_type.value,
                            exchange='NSE',
                            filing_date=filing_date,
                            category=item.get('category', ''),
                            subcategory=item.get('subCategory', ''),
                            description=item.get('details', ''),
                            attachment_name=item.get('attchmntFile', '')
                        )
                        documents.append(doc)

                    except Exception as e:
                        self.logger.warning(f"Error parsing NSE announcement: {e}")
                        continue

                self.logger.info(f"Fetched {len(documents)} NSE announcements")

            elif response.status_code == 401:
                self.logger.warning("NSE session expired, resetting...")
                self._reset_nse_session()

            else:
                self.logger.error(f"NSE API error: {response.status_code}")

        except requests.RequestException as e:
            self.logger.error(f"Error fetching NSE announcements: {e}")

        return documents

    def fetch_nse_financial_results(
        self,
        ticker: Optional[str] = None,
        period: str = "Quarterly"
    ) -> List[CircularDocument]:
        """
        Fetch financial results from NSE.

        Args:
            ticker: Stock symbol
            period: 'Quarterly', 'Half-Yearly', 'Annual'
        """
        documents = []
        session = self._get_nse_session()

        params = {
            'index': 'equities',
            'period': period
        }

        if ticker:
            params['symbol'] = ticker.replace('.NS', '').upper()

        try:
            self._rate_limit_wait()

            response = session.get(
                self.NSE_FINANCIAL_RESULTS_API,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()

                for item in data:
                    try:
                        # Parse filing date
                        result_date = item.get('re_broadcast_date', '') or item.get('relatingTo', '')

                        # Get XBRL/PDF link
                        xbrl_link = item.get('xbrl', '')
                        pdf_link = item.get('re_attachment', '')
                        attachment = pdf_link or xbrl_link

                        if attachment and not attachment.startswith('http'):
                            attachment = f"https://www.nseindia.com{attachment}"

                        doc = CircularDocument(
                            id=self._generate_doc_id(
                                attachment,
                                f"{item.get('symbol', '')} {period} Results",
                                result_date
                            ),
                            ticker=item.get('symbol', ''),
                            company_name=item.get('companyName', ''),
                            title=f"{item.get('symbol', '')} - {period} Financial Results",
                            url=attachment,
                            doc_type=DocumentType.QUARTERLY_RESULTS.value,
                            exchange='NSE',
                            filing_date=result_date,
                            category='Financial Results',
                            subcategory=period,
                            description=f"Financial results for period: {item.get('relatingTo', '')}",
                            attachment_name=item.get('re_attachment', '')
                        )
                        documents.append(doc)

                    except Exception as e:
                        self.logger.warning(f"Error parsing NSE result: {e}")
                        continue

            elif response.status_code == 401:
                self._reset_nse_session()

        except requests.RequestException as e:
            self.logger.error(f"Error fetching NSE financial results: {e}")

        return documents

    def fetch_nse_board_meetings(
        self,
        ticker: Optional[str] = None,
        days_back: int = 30
    ) -> List[CircularDocument]:
        """Fetch board meeting information from NSE."""
        documents = []
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
                self.NSE_BOARD_MEETINGS_API,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()

                for item in data:
                    try:
                        meeting_date = item.get('bm_date', '')

                        doc = CircularDocument(
                            id=self._generate_doc_id(
                                item.get('symbol', ''),
                                item.get('bm_purpose', ''),
                                meeting_date
                            ),
                            ticker=item.get('symbol', ''),
                            company_name=item.get('sm_name', ''),
                            title=f"Board Meeting - {item.get('bm_purpose', '')}",
                            url='',  # Board meetings may not have attachments
                            doc_type=DocumentType.BOARD_MEETING.value,
                            exchange='NSE',
                            filing_date=meeting_date,
                            category='Board Meeting',
                            subcategory=item.get('bm_purpose', ''),
                            description=item.get('bm_desc', ''),
                            attachment_name=''
                        )
                        documents.append(doc)

                    except Exception as e:
                        self.logger.warning(f"Error parsing NSE board meeting: {e}")
                        continue

        except requests.RequestException as e:
            self.logger.error(f"Error fetching NSE board meetings: {e}")

        return documents

    def fetch_nse_shareholding(
        self,
        ticker: str
    ) -> List[CircularDocument]:
        """Fetch shareholding pattern from NSE."""
        documents = []
        session = self._get_nse_session()

        symbol = ticker.replace('.NS', '').upper()

        try:
            self._rate_limit_wait()

            response = session.get(
                f"{self.NSE_SHAREHOLDING_API}?symbol={symbol}",
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()

                # Get historical shareholding
                for item in data.get('data', []):
                    try:
                        doc = CircularDocument(
                            id=self._generate_doc_id(
                                symbol,
                                'Shareholding Pattern',
                                item.get('date', '')
                            ),
                            ticker=symbol,
                            company_name=data.get('companyName', ''),
                            title=f"Shareholding Pattern - {item.get('date', '')}",
                            url=item.get('link', ''),
                            doc_type=DocumentType.SHAREHOLDING.value,
                            exchange='NSE',
                            filing_date=item.get('date', ''),
                            category='Shareholding Pattern',
                            subcategory=item.get('period', ''),
                            description=f"Shareholding pattern for {item.get('period', '')}",
                            attachment_name=''
                        )
                        documents.append(doc)

                    except Exception as e:
                        self.logger.warning(f"Error parsing shareholding: {e}")
                        continue

        except requests.RequestException as e:
            self.logger.error(f"Error fetching shareholding: {e}")

        return documents

    def _categorize_nse_document(self, subject: str) -> DocumentType:
        """Categorize NSE document based on subject."""
        subject_lower = subject.lower()

        if any(kw in subject_lower for kw in ['result', 'financial', 'earnings']):
            return DocumentType.QUARTERLY_RESULTS
        elif any(kw in subject_lower for kw in ['board meeting', 'meeting of board']):
            return DocumentType.BOARD_MEETING
        elif any(kw in subject_lower for kw in ['agm', 'egm', 'general meeting']):
            return DocumentType.AGM_EGM
        elif any(kw in subject_lower for kw in ['dividend', 'bonus', 'split', 'rights']):
            return DocumentType.CORPORATE_ACTION
        elif any(kw in subject_lower for kw in ['shareholding', 'holding pattern']):
            return DocumentType.SHAREHOLDING
        elif any(kw in subject_lower for kw in ['credit rating', 'rating']):
            return DocumentType.CREDIT_RATING
        elif any(kw in subject_lower for kw in ['insider', 'trading']):
            return DocumentType.INSIDER_TRADING
        elif any(kw in subject_lower for kw in ['press release', 'media']):
            return DocumentType.PRESS_RELEASE
        elif any(kw in subject_lower for kw in ['investor', 'presentation', 'analyst']):
            return DocumentType.INVESTOR_PRESENTATION
        else:
            return DocumentType.ANNOUNCEMENT

    # =========================================================================
    # Combined Fetching
    # =========================================================================

    def fetch_recent_circulars(
        self,
        exchange: str = "BOTH",
        ticker: Optional[str] = None,
        days_back: int = 7,
        doc_types: Optional[List[str]] = None
    ) -> List[CircularDocument]:
        """
        Fetch recent circulars from specified exchange(s).

        Args:
            exchange: 'BSE', 'NSE', or 'BOTH'
            ticker: Optional stock symbol
            days_back: Number of days to look back
            doc_types: Filter by document types

        Returns:
            List of CircularDocument objects
        """
        documents = []

        if exchange.upper() in ['BSE', 'BOTH']:
            bse_docs = self.fetch_bse_announcements(ticker, days_back)
            documents.extend(bse_docs)

        if exchange.upper() in ['NSE', 'BOTH']:
            nse_docs = self.fetch_nse_announcements(ticker, days_back)
            documents.extend(nse_docs)

        # Filter by document types if specified
        if doc_types:
            documents = [
                d for d in documents
                if d.doc_type in doc_types
            ]

        # Remove duplicates (same company, same date, similar title)
        documents = self._deduplicate_documents(documents)

        # Sort by date (newest first)
        documents.sort(
            key=lambda x: x.filing_date if x.filing_date else '',
            reverse=True
        )

        return documents

    def fetch_company_filings(
        self,
        ticker: str,
        days_back: int = 365
    ) -> Dict[str, List[CircularDocument]]:
        """
        Fetch all types of filings for a specific company.

        Returns documents organized by type.
        """
        results = {
            'quarterly_results': [],
            'board_meetings': [],
            'announcements': [],
            'shareholding': [],
            'all': []
        }

        # Fetch from both exchanges
        all_docs = self.fetch_recent_circulars(
            exchange='BOTH',
            ticker=ticker,
            days_back=days_back
        )

        results['all'] = all_docs

        # Categorize
        for doc in all_docs:
            if doc.doc_type == DocumentType.QUARTERLY_RESULTS.value:
                results['quarterly_results'].append(doc)
            elif doc.doc_type == DocumentType.BOARD_MEETING.value:
                results['board_meetings'].append(doc)
            elif doc.doc_type == DocumentType.SHAREHOLDING.value:
                results['shareholding'].append(doc)
            else:
                results['announcements'].append(doc)

        # Also fetch specific data
        nse_shareholding = self.fetch_nse_shareholding(ticker)
        results['shareholding'].extend(nse_shareholding)

        return results

    def _deduplicate_documents(
        self,
        documents: List[CircularDocument]
    ) -> List[CircularDocument]:
        """Remove duplicate documents based on content similarity."""
        seen = set()
        unique = []

        for doc in documents:
            # Create a key based on ticker, date, and title prefix
            # Ensure ticker is string (BSE returns integers)
            ticker_str = str(doc.ticker).upper() if doc.ticker else ''
            key = (
                ticker_str,
                doc.filing_date[:10] if doc.filing_date else '',
                doc.title[:50].lower() if doc.title else ''
            )

            if key not in seen:
                seen.add(key)
                unique.append(doc)

        return unique

    # =========================================================================
    # Document Processing
    # =========================================================================

    def download_document(self, doc: CircularDocument) -> Optional[Path]:
        """
        Download a document to local storage.

        Args:
            doc: CircularDocument object

        Returns:
            Path to downloaded file or None if failed
        """
        if not doc.url:
            return None

        try:
            self._rate_limit_wait()

            # Use appropriate session for NSE
            if doc.exchange == 'NSE':
                session = self._get_nse_session()
            else:
                session = requests.Session()
                session.headers.update(self.HEADERS)

            response = session.get(
                doc.url,
                timeout=30,
                stream=True
            )

            if response.status_code == 200:
                # Determine file extension
                content_type = response.headers.get('Content-Type', '').lower()
                content_disp = response.headers.get('Content-Disposition', '')

                if 'pdf' in content_type or '.pdf' in doc.url.lower():
                    ext = '.pdf'
                elif 'xml' in content_type or '.xml' in doc.url.lower():
                    ext = '.xml'
                elif 'zip' in content_type:
                    ext = '.zip'
                elif 'excel' in content_type or 'spreadsheet' in content_type:
                    ext = '.xlsx'
                else:
                    ext = '.pdf'  # Default

                # Create filename
                safe_ticker = re.sub(r'[^\w\-]', '', doc.ticker)
                filename = f"{safe_ticker}_{doc.id}{ext}"
                filepath = self.data_dir / doc.doc_type / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                self.logger.info(f"Downloaded: {filepath}")
                return filepath

            else:
                self.logger.warning(f"Failed to download {doc.url}: {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Error downloading {doc.url}: {e}")
            return None

    def parse_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Parse PDF into structured content.

        Uses pypdf for text extraction.
        For complex PDFs with tables, uses camelot if available.

        Returns:
            {
                "text": "Full extracted text",
                "tables": [{"headers": [...], "data": [[...]]}],
                "key_figures": {...},
                "metadata": {"pages": N, "parsed_at": "..."}
            }
        """
        result = {
            "text": "",
            "tables": [],
            "key_figures": {},
            "metadata": {
                "pages": 0,
                "parsed_at": datetime.now().isoformat(),
                "parser": "pypdf",
                "file_path": str(pdf_path)
            }
        }

        if not pdf_path.exists():
            self.logger.error(f"PDF not found: {pdf_path}")
            return result

        try:
            # Try pypdf first
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            result["metadata"]["pages"] = len(reader.pages)

            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            result["text"] = "\n\n".join(text_parts)

            # Extract key figures (financial numbers)
            result["key_figures"] = self._extract_key_figures(result["text"])

            # Try to extract tables using camelot if available
            try:
                import camelot
                tables = camelot.read_pdf(str(pdf_path), pages='all')
                for table in tables:
                    df = table.df
                    result["tables"].append({
                        "headers": df.iloc[0].tolist() if len(df) > 0 else [],
                        "data": df.iloc[1:].values.tolist() if len(df) > 1 else [],
                        "accuracy": table.accuracy
                    })
                result["metadata"]["parser"] = "pypdf+camelot"
            except ImportError:
                self.logger.debug("Camelot not available for table extraction")
            except Exception as e:
                self.logger.debug(f"Table extraction failed: {e}")

        except ImportError:
            self.logger.error("pypdf not installed")
        except Exception as e:
            self.logger.error(f"Error parsing PDF {pdf_path}: {e}")

        return result

    def _extract_key_figures(self, text: str) -> Dict[str, Any]:
        """Extract key financial figures from text."""
        figures = {}

        # Common patterns for financial figures
        patterns = {
            'revenue': r'(?:revenue|total\s+income|net\s+sales)[\s:]+(?:Rs\.?\s*)?([0-9,]+(?:\.[0-9]+)?)\s*(?:cr|crore|lakh|million)?',
            'net_profit': r'(?:net\s+profit|profit\s+after\s+tax|PAT)[\s:]+(?:Rs\.?\s*)?([0-9,]+(?:\.[0-9]+)?)\s*(?:cr|crore|lakh|million)?',
            'eps': r'(?:EPS|earnings\s+per\s+share)[\s:]+(?:Rs\.?\s*)?([0-9]+(?:\.[0-9]+)?)',
            'dividend': r'(?:dividend)[\s:]+(?:Rs\.?\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+share|%)?',
        }

        text_lower = text.lower()

        for key, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                try:
                    value = match.group(1).replace(',', '')
                    figures[key] = float(value)
                except (ValueError, IndexError):
                    pass

        return figures

    def chunk_for_rag(
        self,
        content: Dict[str, Any],
        doc_id: str,
        metadata: Optional[Dict] = None,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Dict]:
        """
        Chunk document for vector storage.

        Args:
            content: Parsed document content
            doc_id: Document ID for reference
            metadata: Additional metadata to include
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of {text, metadata, chunk_index}
        """
        chunks = []
        text = content.get("text", "")

        if not text:
            return chunks

        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        chunk_index = 0

        base_metadata = {
            "doc_id": doc_id,
            "total_pages": content.get("metadata", {}).get("pages", 0),
            "has_tables": len(content.get("tables", [])) > 0,
            "key_figures": content.get("key_figures", {})
        }
        if metadata:
            base_metadata.update(metadata)

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": {
                            **base_metadata,
                            "chunk_index": chunk_index
                        },
                        "chunk_index": chunk_index
                    })
                    chunk_index += 1

                # Start new chunk with overlap
                if overlap > 0 and chunks:
                    overlap_text = chunks[-1]["text"][-overlap:]
                    current_chunk = overlap_text + " " + sentence + " "
                else:
                    current_chunk = sentence + " "

        # Add remaining text
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": {
                    **base_metadata,
                    "chunk_index": chunk_index
                },
                "chunk_index": chunk_index
            })

        return chunks

    def save_document(
        self,
        doc: CircularDocument,
        parsed_content: Optional[Dict] = None
    ):
        """Save document metadata and content to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO circulars
            (id, ticker, company_name, title, url, doc_type, exchange, filing_date,
             category, subcategory, description, attachment_name, file_size,
             local_path, parsed_text, tables_json, key_figures_json,
             fetched_at, parsed_at, is_processed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc.id,
            doc.ticker,
            doc.company_name,
            doc.title,
            doc.url,
            doc.doc_type,
            doc.exchange,
            doc.filing_date,
            doc.category,
            doc.subcategory,
            doc.description,
            doc.attachment_name,
            doc.file_size,
            doc.local_path,
            parsed_content.get('text') if parsed_content else None,
            json.dumps(parsed_content.get('tables', [])) if parsed_content else None,
            json.dumps(parsed_content.get('key_figures', {})) if parsed_content else None,
            datetime.now().isoformat(),
            datetime.now().isoformat() if parsed_content else None,
            1 if parsed_content else 0
        ))

        conn.commit()
        conn.close()

    def process_document(
        self,
        doc: CircularDocument,
        download: bool = True,
        parse: bool = True
    ) -> Tuple[CircularDocument, Optional[Dict], List[Dict]]:
        """
        Full processing pipeline for a document:
        1. Download file (if URL present)
        2. Parse content (if PDF)
        3. Chunk for RAG
        4. Save to database

        Returns:
            (document, parsed_content, chunks)
        """
        parsed_content = None
        chunks = []

        # Download
        if download and doc.url:
            local_path = self.download_document(doc)
            if local_path:
                doc.local_path = str(local_path)

        # Parse
        if parse and doc.local_path:
            local_path = Path(doc.local_path)
            if local_path.exists() and local_path.suffix.lower() == '.pdf':
                parsed_content = self.parse_pdf(local_path)
                doc.parsed = True

                # Chunk for RAG
                chunks = self.chunk_for_rag(
                    parsed_content,
                    doc.id,
                    metadata={
                        "ticker": doc.ticker,
                        "company_name": doc.company_name,
                        "doc_type": doc.doc_type,
                        "exchange": doc.exchange,
                        "filing_date": doc.filing_date,
                        "category": doc.category
                    }
                )

        # Save
        self.save_document(doc, parsed_content)

        return doc, parsed_content, chunks

    def search_circulars(
        self,
        query: Optional[str] = None,
        ticker: Optional[str] = None,
        doc_type: Optional[str] = None,
        exchange: Optional[str] = None,
        days_back: int = 30
    ) -> List[Dict]:
        """Search stored circulars."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        sql = "SELECT * FROM circulars WHERE 1=1"
        params = []

        if ticker:
            sql += " AND ticker LIKE ?"
            params.append(f"%{ticker.replace('.NS', '').replace('.BO', '')}%")

        if doc_type:
            sql += " AND doc_type = ?"
            params.append(doc_type)

        if exchange:
            sql += " AND exchange = ?"
            params.append(exchange.upper())

        if days_back:
            cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
            sql += " AND filing_date >= ?"
            params.append(cutoff)

        if query:
            sql += " AND (title LIKE ? OR parsed_text LIKE ? OR description LIKE ?)"
            params.extend([f"%{query}%"] * 3)

        sql += " ORDER BY filing_date DESC LIMIT 100"

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Get a specific document by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM circulars WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document already exists in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM circulars WHERE id = ?", (doc_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def is_document_processed(self, doc_id: str) -> bool:
        """Check if a document exists and has been processed (downloaded + parsed)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT is_processed FROM circulars WHERE id = ? AND is_processed = 1",
            (doc_id,)
        )
        processed = cursor.fetchone() is not None
        conn.close()
        return processed

    def get_existing_doc_ids(self) -> set:
        """Get set of all existing document IDs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM circulars")
        ids = {row[0] for row in cursor.fetchall()}
        conn.close()
        return ids

    def filter_new_documents(
        self,
        documents: List[CircularDocument]
    ) -> List[CircularDocument]:
        """Filter out documents that already exist in the database."""
        existing_ids = self.get_existing_doc_ids()
        new_docs = [doc for doc in documents if doc.id not in existing_ids]
        skipped = len(documents) - len(new_docs)
        if skipped > 0:
            self.logger.info(f"Skipped {skipped} already-ingested documents")
        return new_docs

    def get_unprocessed_documents(self, limit: int = 50) -> List[Dict]:
        """Get documents that haven't been processed yet."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM circulars
            WHERE is_processed = 0 AND url IS NOT NULL AND url != ''
            ORDER BY filing_date DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_ingestion_stats(self) -> Dict:
        """Get statistics about ingested documents."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Total documents
        cursor.execute("SELECT COUNT(*) FROM circulars")
        stats['total_documents'] = cursor.fetchone()[0]

        # By exchange
        cursor.execute("SELECT exchange, COUNT(*) FROM circulars GROUP BY exchange")
        stats['by_exchange'] = dict(cursor.fetchall())

        # By type
        cursor.execute("SELECT doc_type, COUNT(*) FROM circulars GROUP BY doc_type")
        stats['by_type'] = dict(cursor.fetchall())

        # Processed vs unprocessed
        cursor.execute("SELECT is_processed, COUNT(*) FROM circulars GROUP BY is_processed")
        processed = dict(cursor.fetchall())
        stats['processed'] = processed.get(1, 0)
        stats['unprocessed'] = processed.get(0, 0)

        # Recent (last 7 days)
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute("SELECT COUNT(*) FROM circulars WHERE fetched_at >= ?", (cutoff,))
        stats['last_7_days'] = cursor.fetchone()[0]

        conn.close()

        return stats

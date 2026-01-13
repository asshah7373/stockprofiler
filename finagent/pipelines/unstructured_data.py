"""
Unstructured Data Pipeline
Fetches and parses BSE/NSE circulars and corporate filings.
"""

import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import json
import hashlib
from datetime import datetime, timedelta
import sqlite3
import re

logger = logging.getLogger(__name__)


class CircularPipeline:
    """
    Scrapes, downloads, and parses regulatory circulars.

    Sources:
    - BSE Corporate Announcements
    - NSE Corporate Filings
    - Exchange Circulars
    """

    BSE_ANNOUNCEMENTS_URL = "https://www.bseindia.com/corporates/ann.html"
    NSE_FILINGS_URL = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"

    # Headers to mimic browser request
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

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
        self._init_db()

    def _init_db(self):
        """Initialize database for storing circular metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS circulars (
                id TEXT PRIMARY KEY,
                ticker TEXT,
                title TEXT NOT NULL,
                url TEXT,
                doc_type TEXT,
                exchange TEXT,
                filing_date TEXT,
                local_path TEXT,
                parsed_text TEXT,
                tables_json TEXT,
                embedding_id TEXT,
                fetched_at TEXT NOT NULL,
                parsed_at TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_circulars_ticker
            ON circulars(ticker)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_circulars_date
            ON circulars(filing_date)
        """)

        conn.commit()
        conn.close()

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _generate_doc_id(self, url: str, title: str) -> str:
        """Generate unique document ID."""
        content = f"{url}:{title}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def fetch_recent_circulars(
        self,
        exchange: str = "NSE",
        ticker: Optional[str] = None,
        days_back: int = 7
    ) -> List[Dict]:
        """
        Fetch list of recent circulars.

        Note: Due to website structure complexity and anti-scraping measures,
        this method may need adjustment based on actual website structure.

        Returns:
            List of {id, title, url, date, ticker, type, exchange}
        """
        circulars = []

        try:
            self._rate_limit_wait()

            if exchange.upper() == "NSE":
                circulars = self._fetch_nse_circulars(ticker, days_back)
            elif exchange.upper() == "BSE":
                circulars = self._fetch_bse_circulars(ticker, days_back)
            else:
                self.logger.warning(f"Unknown exchange: {exchange}")

        except Exception as e:
            self.logger.error(f"Error fetching circulars: {e}")

        return circulars

    def _fetch_nse_circulars(
        self,
        ticker: Optional[str],
        days_back: int
    ) -> List[Dict]:
        """Fetch circulars from NSE website."""
        circulars = []

        # NSE API endpoint for corporate filings
        api_url = "https://www.nseindia.com/api/corporates-corporateActions"

        try:
            # Create a session to handle cookies
            session = requests.Session()

            # First visit main page to get cookies
            session.get(
                "https://www.nseindia.com",
                headers=self.HEADERS,
                timeout=10
            )

            self._rate_limit_wait()

            # Prepare params
            params = {
                'index': 'equities'
            }
            if ticker:
                params['symbol'] = ticker.replace('.NS', '').upper()

            response = session.get(
                api_url,
                headers=self.HEADERS,
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                cutoff_date = datetime.now() - timedelta(days=days_back)

                for item in data.get('data', []):
                    try:
                        filing_date = datetime.strptime(
                            item.get('recordDt', ''),
                            '%d-%b-%Y'
                        )

                        if filing_date >= cutoff_date:
                            circular = {
                                'id': self._generate_doc_id(
                                    str(item),
                                    item.get('subject', '')
                                ),
                                'ticker': item.get('symbol', ''),
                                'title': item.get('subject', ''),
                                'url': item.get('attachment', ''),
                                'doc_type': item.get('series', 'corporate_action'),
                                'exchange': 'NSE',
                                'filing_date': filing_date.isoformat(),
                            }
                            circulars.append(circular)
                    except ValueError:
                        continue

        except requests.RequestException as e:
            self.logger.error(f"Error fetching NSE circulars: {e}")

        return circulars

    def _fetch_bse_circulars(
        self,
        ticker: Optional[str],
        days_back: int
    ) -> List[Dict]:
        """Fetch circulars from BSE website."""
        circulars = []

        # BSE API endpoint
        api_url = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"

        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
            to_date = datetime.now().strftime('%Y%m%d')

            params = {
                'strCat': '-1',
                'strPrevDate': from_date,
                'strScrip': ticker.replace('.BO', '') if ticker else '',
                'strSearch': 'P',
                'strToDate': to_date,
                'strType': 'C'
            }

            self._rate_limit_wait()

            response = requests.get(
                api_url,
                headers=self.HEADERS,
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                for item in data.get('Table', []):
                    circular = {
                        'id': self._generate_doc_id(
                            item.get('ATTACHMENTNAME', ''),
                            item.get('HEADLINE', '')
                        ),
                        'ticker': item.get('SCRIP_CD', ''),
                        'title': item.get('HEADLINE', ''),
                        'url': item.get('ATTACHMENTNAME', ''),
                        'doc_type': item.get('CATEGORYNAME', 'announcement'),
                        'exchange': 'BSE',
                        'filing_date': item.get('NEWS_DT', ''),
                    }
                    circulars.append(circular)

        except requests.RequestException as e:
            self.logger.error(f"Error fetching BSE circulars: {e}")

        return circulars

    def download_pdf(self, url: str, doc_id: str) -> Optional[Path]:
        """
        Download PDF circular to local storage.

        Args:
            url: URL of the PDF
            doc_id: Document ID for naming

        Returns:
            Path to downloaded file or None if failed
        """
        if not url:
            return None

        try:
            self._rate_limit_wait()

            response = requests.get(
                url,
                headers=self.HEADERS,
                timeout=30,
                stream=True
            )

            if response.status_code == 200:
                # Determine file extension
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' in content_type.lower():
                    ext = '.pdf'
                elif 'xml' in content_type.lower():
                    ext = '.xml'
                else:
                    ext = '.pdf'  # Default

                filename = f"{doc_id}{ext}"
                filepath = self.data_dir / filename

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                self.logger.info(f"Downloaded: {filepath}")
                return filepath
            else:
                self.logger.warning(f"Failed to download {url}: {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Error downloading {url}: {e}")
            return None

    def parse_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Parse PDF into structured content.

        Uses pypdf for text extraction.
        For complex PDFs with tables, consider using camelot or llama-parse.

        Returns:
            {
                "text": "Full extracted text",
                "tables": [{"headers": [...], "data": [[...]]}],
                "metadata": {"pages": N, "parsed_at": "..."}
            }
        """
        result = {
            "text": "",
            "tables": [],
            "metadata": {
                "pages": 0,
                "parsed_at": datetime.now().isoformat(),
                "parser": "pypdf"
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

            # Try to extract tables using camelot if available
            try:
                import camelot
                tables = camelot.read_pdf(str(pdf_path), pages='all')
                for table in tables:
                    df = table.df
                    result["tables"].append({
                        "headers": df.iloc[0].tolist() if len(df) > 0 else [],
                        "data": df.iloc[1:].values.tolist() if len(df) > 1 else []
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

    def chunk_for_rag(
        self,
        content: Dict[str, Any],
        doc_id: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Dict]:
        """
        Chunk document for vector storage.

        Args:
            content: Parsed document content
            doc_id: Document ID for reference
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

        # Split into sentences (roughly)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        chunk_index = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": {
                            "doc_id": doc_id,
                            "chunk_index": chunk_index,
                            "total_pages": content.get("metadata", {}).get("pages", 0)
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
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "total_pages": content.get("metadata", {}).get("pages", 0)
                },
                "chunk_index": chunk_index
            })

        return chunks

    def save_circular(self, circular: Dict, parsed_content: Optional[Dict] = None):
        """Save circular metadata and content to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO circulars
            (id, ticker, title, url, doc_type, exchange, filing_date,
             local_path, parsed_text, tables_json, fetched_at, parsed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            circular.get('id'),
            circular.get('ticker'),
            circular.get('title'),
            circular.get('url'),
            circular.get('doc_type'),
            circular.get('exchange'),
            circular.get('filing_date'),
            str(circular.get('local_path', '')),
            parsed_content.get('text') if parsed_content else None,
            json.dumps(parsed_content.get('tables', [])) if parsed_content else None,
            datetime.now().isoformat(),
            datetime.now().isoformat() if parsed_content else None
        ))

        conn.commit()
        conn.close()

    def search_circulars(
        self,
        query: Optional[str] = None,
        ticker: Optional[str] = None,
        doc_type: Optional[str] = None,
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
            params.append(f"%{ticker}%")

        if doc_type:
            sql += " AND doc_type = ?"
            params.append(doc_type)

        if days_back:
            cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
            sql += " AND filing_date >= ?"
            params.append(cutoff)

        if query:
            sql += " AND (title LIKE ? OR parsed_text LIKE ?)"
            params.extend([f"%{query}%", f"%{query}%"])

        sql += " ORDER BY filing_date DESC LIMIT 100"

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_circular_by_id(self, doc_id: str) -> Optional[Dict]:
        """Get a specific circular by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM circulars WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def process_circular(self, circular: Dict) -> Dict:
        """
        Full processing pipeline for a circular:
        1. Download PDF
        2. Parse content
        3. Chunk for RAG
        4. Save to database

        Returns processed circular with chunks
        """
        doc_id = circular.get('id')
        url = circular.get('url')

        # Download
        local_path = self.download_pdf(url, doc_id)
        circular['local_path'] = local_path

        parsed_content = None
        chunks = []

        if local_path and local_path.exists():
            # Parse
            parsed_content = self.parse_pdf(local_path)

            # Chunk
            chunks = self.chunk_for_rag(parsed_content, doc_id)

        # Save
        self.save_circular(circular, parsed_content)

        return {
            **circular,
            'parsed_content': parsed_content,
            'chunks': chunks
        }

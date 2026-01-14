"""
Ingestion Orchestrator
Coordinates batch ingestion of circulars, press releases, and other documents.

This module provides:
- Scheduled/batch ingestion from BSE/NSE
- Document processing pipeline
- Vector store integration
- Progress tracking and reporting
"""

import logging
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

from .unstructured_data import CircularPipeline, CircularDocument, DocumentType
from .press_releases import PressReleasePipeline, PressRelease

logger = logging.getLogger(__name__)


class IngestionStatus(Enum):
    """Status of ingestion job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class IngestionJob:
    """Represents an ingestion job."""
    job_id: str
    job_type: str  # 'circulars', 'press_releases', 'all'
    exchange: str  # 'BSE', 'NSE', 'BOTH'
    ticker: Optional[str]
    days_back: int
    status: IngestionStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    documents_found: int = 0
    documents_processed: int = 0
    documents_failed: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['status'] = self.status.value
        result['errors'] = json.dumps(self.errors)
        return result


@dataclass
class IngestionResult:
    """Result of an ingestion run."""
    job: IngestionJob
    circulars: List[CircularDocument]
    press_releases: List[PressRelease]
    chunks_generated: int
    errors: List[str]
    duration_seconds: float


class IngestionOrchestrator:
    """
    Orchestrates ingestion of financial documents.

    Features:
    - Batch processing of BSE/NSE documents
    - Parallel document download and parsing
    - Progress tracking
    - Error handling and retry
    - Integration with RAG vector store
    """

    def __init__(
        self,
        data_dir: str = "data",
        db_path: str = "data/cache/ingestion.db",
        max_workers: int = 4
    ):
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

        # Initialize pipelines
        self.circular_pipeline = CircularPipeline(
            data_dir=str(self.data_dir / "circulars"),
            db_path=str(self.data_dir / "cache" / "circulars.db")
        )
        self.press_pipeline = PressReleasePipeline(
            data_dir=str(self.data_dir / "press_releases"),
            db_path=str(self.data_dir / "cache" / "press_releases.db")
        )

        self._init_db()

    def _init_db(self):
        """Initialize ingestion job tracking database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                exchange TEXT,
                ticker TEXT,
                days_back INTEGER,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                documents_found INTEGER DEFAULT 0,
                documents_processed INTEGER DEFAULT 0,
                documents_failed INTEGER DEFAULT 0,
                errors TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status
            ON ingestion_jobs(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_created
            ON ingestion_jobs(created_at)
        """)

        conn.commit()
        conn.close()

    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        import uuid
        return str(uuid.uuid4())[:12]

    def _save_job(self, job: IngestionJob):
        """Save job to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        job_dict = job.to_dict()

        cursor.execute("""
            INSERT OR REPLACE INTO ingestion_jobs
            (job_id, job_type, exchange, ticker, days_back, status,
             created_at, started_at, completed_at, documents_found,
             documents_processed, documents_failed, errors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_dict['job_id'],
            job_dict['job_type'],
            job_dict['exchange'],
            job_dict['ticker'],
            job_dict['days_back'],
            job_dict['status'],
            job_dict['created_at'],
            job_dict['started_at'],
            job_dict['completed_at'],
            job_dict['documents_found'],
            job_dict['documents_processed'],
            job_dict['documents_failed'],
            job_dict['errors']
        ))

        conn.commit()
        conn.close()

    # =========================================================================
    # Main Ingestion Methods
    # =========================================================================

    def ingest_circulars(
        self,
        exchange: str = "BOTH",
        ticker: Optional[str] = None,
        days_back: int = 7,
        download: bool = True,
        parse: bool = True,
        skip_existing: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> IngestionResult:
        """
        Ingest circulars from BSE/NSE.

        Args:
            exchange: 'BSE', 'NSE', or 'BOTH'
            ticker: Optional ticker to filter
            days_back: Number of days to look back
            download: Whether to download document files
            parse: Whether to parse document content
            skip_existing: Skip documents already in database (default: True)
            progress_callback: Callback(current, total, message)

        Returns:
            IngestionResult with details
        """
        start_time = time.time()

        # Create job
        job = IngestionJob(
            job_id=self._generate_job_id(),
            job_type='circulars',
            exchange=exchange,
            ticker=ticker,
            days_back=days_back,
            status=IngestionStatus.RUNNING,
            created_at=datetime.now().isoformat(),
            started_at=datetime.now().isoformat()
        )
        self._save_job(job)

        errors = []
        processed_circulars = []
        all_chunks = []

        try:
            # Fetch documents
            if progress_callback:
                progress_callback(0, 0, f"Fetching {exchange} circulars...")

            circulars = self.circular_pipeline.fetch_recent_circulars(
                exchange=exchange,
                ticker=ticker,
                days_back=days_back
            )

            total_found = len(circulars)

            # Filter out already-ingested documents
            if skip_existing:
                circulars = self.circular_pipeline.filter_new_documents(circulars)
                skipped = total_found - len(circulars)
                if skipped > 0:
                    self.logger.info(f"Skipped {skipped} already-ingested circulars")

            job.documents_found = len(circulars)
            self._save_job(job)

            self.logger.info(f"Found {len(circulars)} new circulars to process (total: {total_found})")

            if progress_callback:
                progress_callback(0, len(circulars), f"Processing {len(circulars)} documents...")

            # Process documents
            for i, doc in enumerate(circulars):
                try:
                    processed_doc, content, chunks = self.circular_pipeline.process_document(
                        doc,
                        download=download,
                        parse=parse
                    )
                    processed_circulars.append(processed_doc)
                    all_chunks.extend(chunks)
                    job.documents_processed += 1

                    if progress_callback:
                        progress_callback(
                            i + 1,
                            len(circulars),
                            f"Processed: {doc.ticker} - {doc.title[:50]}..."
                        )

                except Exception as e:
                    error_msg = f"Error processing {doc.id}: {str(e)}"
                    errors.append(error_msg)
                    job.documents_failed += 1
                    self.logger.error(error_msg)

                self._save_job(job)

            job.status = IngestionStatus.COMPLETED if not errors else IngestionStatus.PARTIAL
            job.completed_at = datetime.now().isoformat()
            job.errors = errors

        except Exception as e:
            job.status = IngestionStatus.FAILED
            job.completed_at = datetime.now().isoformat()
            job.errors = [str(e)]
            errors.append(str(e))
            self.logger.error(f"Ingestion job failed: {e}")

        self._save_job(job)

        return IngestionResult(
            job=job,
            circulars=processed_circulars,
            press_releases=[],
            chunks_generated=len(all_chunks),
            errors=errors,
            duration_seconds=time.time() - start_time
        )

    def ingest_press_releases(
        self,
        ticker: Optional[str] = None,
        days_back: int = 30,
        fetch_content: bool = True,
        skip_existing: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> IngestionResult:
        """
        Ingest press releases from BSE/NSE.

        Args:
            ticker: Optional ticker to filter
            days_back: Number of days to look back
            fetch_content: Whether to fetch full content
            skip_existing: Skip press releases already in database (default: True)
            progress_callback: Callback(current, total, message)

        Returns:
            IngestionResult with details
        """
        start_time = time.time()

        job = IngestionJob(
            job_id=self._generate_job_id(),
            job_type='press_releases',
            exchange='BOTH',
            ticker=ticker,
            days_back=days_back,
            status=IngestionStatus.RUNNING,
            created_at=datetime.now().isoformat(),
            started_at=datetime.now().isoformat()
        )
        self._save_job(job)

        errors = []
        processed_releases = []
        all_chunks = []

        try:
            if progress_callback:
                progress_callback(0, 0, "Fetching press releases...")

            releases = self.press_pipeline.fetch_all_press_releases(
                ticker=ticker,
                days_back=days_back
            )

            total_found = len(releases)

            # Filter out already-ingested press releases
            if skip_existing:
                releases = self.press_pipeline.filter_new_documents(releases)
                skipped = total_found - len(releases)
                if skipped > 0:
                    self.logger.info(f"Skipped {skipped} already-ingested press releases")

            job.documents_found = len(releases)
            self._save_job(job)

            self.logger.info(f"Found {len(releases)} new press releases to process (total: {total_found})")

            if progress_callback:
                progress_callback(0, len(releases), f"Processing {len(releases)} press releases...")

            for i, pr in enumerate(releases):
                try:
                    processed_pr, chunks = self.press_pipeline.process_press_release(
                        pr,
                        fetch_content=fetch_content
                    )
                    processed_releases.append(processed_pr)
                    all_chunks.extend(chunks)
                    job.documents_processed += 1

                    if progress_callback:
                        progress_callback(
                            i + 1,
                            len(releases),
                            f"Processed: {pr.ticker} - {pr.title[:50]}..."
                        )

                except Exception as e:
                    error_msg = f"Error processing press release {pr.id}: {str(e)}"
                    errors.append(error_msg)
                    job.documents_failed += 1
                    self.logger.error(error_msg)

                self._save_job(job)

            job.status = IngestionStatus.COMPLETED if not errors else IngestionStatus.PARTIAL
            job.completed_at = datetime.now().isoformat()
            job.errors = errors

        except Exception as e:
            job.status = IngestionStatus.FAILED
            job.completed_at = datetime.now().isoformat()
            job.errors = [str(e)]
            errors.append(str(e))
            self.logger.error(f"Press release ingestion failed: {e}")

        self._save_job(job)

        return IngestionResult(
            job=job,
            circulars=[],
            press_releases=processed_releases,
            chunks_generated=len(all_chunks),
            errors=errors,
            duration_seconds=time.time() - start_time
        )

    def ingest_all(
        self,
        exchange: str = "BOTH",
        ticker: Optional[str] = None,
        days_back: int = 7,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, IngestionResult]:
        """
        Ingest all document types.

        Returns:
            Dictionary with results for each document type
        """
        results = {}

        # Circulars
        self.logger.info("Starting circular ingestion...")
        results['circulars'] = self.ingest_circulars(
            exchange=exchange,
            ticker=ticker,
            days_back=days_back,
            progress_callback=progress_callback
        )

        # Press releases
        self.logger.info("Starting press release ingestion...")
        results['press_releases'] = self.ingest_press_releases(
            ticker=ticker,
            days_back=days_back,
            progress_callback=progress_callback
        )

        return results

    def ingest_company(
        self,
        ticker: str,
        days_back: int = 365,
        include_results: bool = True,
        include_announcements: bool = True,
        include_press_releases: bool = True,
        skip_existing: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive ingestion for a specific company.

        Fetches:
        - Quarterly results
        - Board meeting outcomes
        - Shareholding patterns
        - Corporate announcements
        - Press releases

        Args:
            skip_existing: Skip documents already in database (default: True)
        """
        results = {
            'ticker': ticker,
            'quarterly_results': [],
            'board_meetings': [],
            'shareholding': [],
            'announcements': [],
            'press_releases': [],
            'total_documents': 0,
            'total_chunks': 0,
            'errors': []
        }

        try:
            # Fetch all filings
            if progress_callback:
                progress_callback(0, 0, f"Fetching all filings for {ticker}...")

            filings = self.circular_pipeline.fetch_company_filings(
                ticker=ticker,
                days_back=days_back
            )

            # Filter out already-ingested documents if skip_existing is True
            if skip_existing:
                existing_circular_ids = self.circular_pipeline.get_existing_doc_ids()
                existing_press_ids = self.press_pipeline.get_existing_doc_ids()

                for key in ['quarterly_results', 'announcements', 'board_meetings', 'shareholding']:
                    if key in filings:
                        original_count = len(filings[key])
                        filings[key] = [doc for doc in filings[key] if doc.id not in existing_circular_ids]
                        skipped = original_count - len(filings[key])
                        if skipped > 0:
                            self.logger.info(f"Skipped {skipped} already-ingested {key}")

            # Process by type
            if include_results:
                for doc in filings.get('quarterly_results', []):
                    try:
                        processed, _, chunks = self.circular_pipeline.process_document(doc)
                        results['quarterly_results'].append(processed.to_dict())
                        results['total_chunks'] += len(chunks)
                        results['total_documents'] += 1
                    except Exception as e:
                        results['errors'].append(f"Results error: {e}")

            if include_announcements:
                for doc in filings.get('announcements', []):
                    try:
                        processed, _, chunks = self.circular_pipeline.process_document(doc)
                        results['announcements'].append(processed.to_dict())
                        results['total_chunks'] += len(chunks)
                        results['total_documents'] += 1
                    except Exception as e:
                        results['errors'].append(f"Announcement error: {e}")

            # Board meetings
            for doc in filings.get('board_meetings', []):
                results['board_meetings'].append(doc.to_dict())
                results['total_documents'] += 1

            # Shareholding
            for doc in filings.get('shareholding', []):
                results['shareholding'].append(doc.to_dict())
                results['total_documents'] += 1

            # Press releases
            if include_press_releases:
                if progress_callback:
                    progress_callback(0, 0, f"Fetching press releases for {ticker}...")

                press_releases = self.press_pipeline.fetch_all_press_releases(
                    ticker=ticker,
                    days_back=days_back
                )

                # Filter out already-ingested press releases
                if skip_existing:
                    original_count = len(press_releases)
                    # Reuse existing_press_ids from above if available, otherwise fetch
                    if 'existing_press_ids' not in dir():
                        existing_press_ids = self.press_pipeline.get_existing_doc_ids()
                    press_releases = [pr for pr in press_releases if pr.id not in existing_press_ids]
                    skipped = original_count - len(press_releases)
                    if skipped > 0:
                        self.logger.info(f"Skipped {skipped} already-ingested press releases")

                for pr in press_releases:
                    try:
                        processed, chunks = self.press_pipeline.process_press_release(pr)
                        results['press_releases'].append(processed.to_dict())
                        results['total_chunks'] += len(chunks)
                        results['total_documents'] += 1
                    except Exception as e:
                        results['errors'].append(f"Press release error: {e}")

        except Exception as e:
            results['errors'].append(f"Company ingestion error: {e}")
            self.logger.error(f"Error ingesting company {ticker}: {e}")

        return results

    # =========================================================================
    # Index Building
    # =========================================================================

    def build_rag_index(
        self,
        vector_store,
        ticker: Optional[str] = None,
        days_back: int = 90,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, int]:
        """
        Build/update RAG index from ingested documents.

        Args:
            vector_store: VectorStore instance
            ticker: Optional ticker filter
            days_back: Lookback period
            progress_callback: Progress callback

        Returns:
            Statistics about indexed documents
        """
        stats = {
            'circulars_indexed': 0,
            'press_releases_indexed': 0,
            'total_chunks': 0,
            'errors': 0
        }

        # Get unprocessed circulars
        circulars = self.circular_pipeline.search_circulars(
            ticker=ticker,
            days_back=days_back
        )

        if progress_callback:
            progress_callback(0, len(circulars), "Indexing circulars...")

        for i, doc_dict in enumerate(circulars):
            try:
                # Check if parsed
                if doc_dict.get('parsed_text'):
                    content = {
                        'text': doc_dict['parsed_text'],
                        'tables': json.loads(doc_dict.get('tables_json', '[]')),
                        'key_figures': json.loads(doc_dict.get('key_figures_json', '{}')),
                        'metadata': {
                            'pages': 0
                        }
                    }

                    chunks = self.circular_pipeline.chunk_for_rag(
                        content,
                        doc_dict['id'],
                        metadata={
                            'ticker': doc_dict.get('ticker'),
                            'company_name': doc_dict.get('company_name'),
                            'doc_type': doc_dict.get('doc_type'),
                            'exchange': doc_dict.get('exchange'),
                            'filing_date': doc_dict.get('filing_date'),
                            'category': doc_dict.get('category')
                        }
                    )

                    if chunks:
                        vector_store.add_document(
                            collection='circulars',
                            doc_id=doc_dict['id'],
                            chunks=chunks,
                            metadata={
                                'ticker': doc_dict.get('ticker'),
                                'filing_date': doc_dict.get('filing_date'),
                                'doc_type': doc_dict.get('doc_type')
                            }
                        )
                        stats['circulars_indexed'] += 1
                        stats['total_chunks'] += len(chunks)

                if progress_callback:
                    progress_callback(i + 1, len(circulars), f"Indexed {doc_dict.get('ticker')}")

            except Exception as e:
                stats['errors'] += 1
                self.logger.error(f"Error indexing circular: {e}")

        # Index press releases
        press_releases = self.press_pipeline.search_press_releases(
            ticker=ticker,
            days_back=days_back
        )

        if progress_callback:
            progress_callback(0, len(press_releases), "Indexing press releases...")

        for i, pr_dict in enumerate(press_releases):
            try:
                if pr_dict.get('content') or pr_dict.get('summary'):
                    # Create fake PressRelease for chunking
                    text = pr_dict.get('content') or pr_dict.get('summary') or pr_dict.get('title')

                    chunks = [{
                        'text': text[:500],  # Simple chunking
                        'metadata': {
                            'doc_id': pr_dict['id'],
                            'ticker': pr_dict.get('ticker'),
                            'source': pr_dict.get('source'),
                            'release_date': pr_dict.get('release_date'),
                            'doc_type': 'press_release'
                        }
                    }]

                    vector_store.add_document(
                        collection='press_releases',
                        doc_id=pr_dict['id'],
                        chunks=chunks,
                        metadata={
                            'ticker': pr_dict.get('ticker'),
                            'release_date': pr_dict.get('release_date')
                        }
                    )
                    stats['press_releases_indexed'] += 1
                    stats['total_chunks'] += len(chunks)

                if progress_callback:
                    progress_callback(i + 1, len(press_releases), f"Indexed {pr_dict.get('ticker')}")

            except Exception as e:
                stats['errors'] += 1
                self.logger.error(f"Error indexing press release: {e}")

        return stats

    # =========================================================================
    # Job Management
    # =========================================================================

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job details."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def get_recent_jobs(self, limit: int = 10) -> List[Dict]:
        """Get recent ingestion jobs."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM ingestion_jobs
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive ingestion statistics."""
        stats = {
            'jobs': {},
            'circulars': {},
            'press_releases': {},
            'last_updated': datetime.now().isoformat()
        }

        # Job stats
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM ingestion_jobs")
        stats['jobs']['total'] = cursor.fetchone()[0]

        cursor.execute("SELECT status, COUNT(*) FROM ingestion_jobs GROUP BY status")
        stats['jobs']['by_status'] = dict(cursor.fetchall())

        cursor.execute("""
            SELECT SUM(documents_processed), SUM(documents_failed)
            FROM ingestion_jobs
        """)
        row = cursor.fetchone()
        stats['jobs']['total_processed'] = row[0] or 0
        stats['jobs']['total_failed'] = row[1] or 0

        conn.close()

        # Circular stats
        stats['circulars'] = self.circular_pipeline.get_ingestion_stats()

        # Press release stats
        stats['press_releases'] = self.press_pipeline.get_stats()

        return stats

    def generate_report(self) -> str:
        """Generate a text report of ingestion status."""
        stats = self.get_stats()

        report = f"""
{'='*60}
INGESTION STATUS REPORT
Generated: {stats['last_updated']}
{'='*60}

JOBS SUMMARY
{'─'*60}
Total Jobs: {stats['jobs']['total']}
By Status: {json.dumps(stats['jobs']['by_status'], indent=2)}
Total Documents Processed: {stats['jobs']['total_processed']}
Total Documents Failed: {stats['jobs']['total_failed']}

CIRCULARS
{'─'*60}
Total Documents: {stats['circulars'].get('total_documents', 0)}
By Exchange: {json.dumps(stats['circulars'].get('by_exchange', {}), indent=2)}
By Type: {json.dumps(stats['circulars'].get('by_type', {}), indent=2)}
Processed: {stats['circulars'].get('processed', 0)}
Pending: {stats['circulars'].get('unprocessed', 0)}
Last 7 Days: {stats['circulars'].get('last_7_days', 0)}

PRESS RELEASES
{'─'*60}
Total: {stats['press_releases'].get('total', 0)}
By Source: {json.dumps(stats['press_releases'].get('by_source', {}), indent=2)}
Parsed: {stats['press_releases'].get('parsed', 0)}
Pending: {stats['press_releases'].get('unparsed', 0)}

{'='*60}
"""
        return report

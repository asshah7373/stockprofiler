"""
Institutional Data Pipeline
Fetches FII/DII activity, Bulk/Block deals, and SAST (insider trading) data.

Sources:
- NSE: FII/DII daily activity, Bulk/Block deals
- BSE: Bulk deals, Block deals
- SEBI: SAST filings (insider trading)

These signals have high alpha for Indian markets as they indicate
institutional conviction and insider confidence.
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
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)


class DealType(Enum):
    """Types of large deals."""
    BULK = "bulk"       # > 0.5% of total shares
    BLOCK = "block"     # > 5 lakh shares or Rs 10 crore
    FII_BUY = "fii_buy"
    FII_SELL = "fii_sell"
    DII_BUY = "dii_buy"
    DII_SELL = "dii_sell"


class InstitutionalType(Enum):
    """Types of institutional investors."""
    FII = "fii"    # Foreign Institutional Investors
    DII = "dii"    # Domestic Institutional Investors
    MF = "mf"      # Mutual Funds
    FPI = "fpi"    # Foreign Portfolio Investors


@dataclass
class InstitutionalActivity:
    """Represents daily FII/DII activity."""
    date: str
    investor_type: str  # FII, DII
    buy_value: float    # in crores
    sell_value: float   # in crores
    net_value: float    # in crores (buy - sell)
    category: str       # Cash, Derivatives, Debt
    exchange: str       # NSE, BSE, Combined

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def is_bullish(self) -> bool:
        return self.net_value > 0


@dataclass
class BulkBlockDeal:
    """Represents a bulk or block deal."""
    id: str
    date: str
    ticker: str
    company_name: str
    client_name: str
    deal_type: str      # bulk, block
    trade_type: str     # buy, sell
    quantity: int
    price: float
    value: float        # in crores
    exchange: str
    remarks: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SASTFiling:
    """Represents SAST (Substantial Acquisition of Shares) filing - insider trading."""
    id: str
    date: str
    ticker: str
    company_name: str
    acquirer_name: str
    acquirer_type: str  # promoter, director, kmp, etc.
    trade_type: str     # buy, sell
    shares_before: int
    shares_acquired: int
    shares_after: int
    percent_before: float
    percent_after: float
    value: float        # in lakhs
    mode: str           # market, off-market, ipo, etc.
    exchange: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def is_bullish(self) -> bool:
        """Insider buying is generally bullish signal."""
        return self.trade_type.lower() == 'buy'


class InstitutionalDataPipeline:
    """
    Pipeline for fetching institutional investment data.

    Provides:
    - FII/DII daily activity from NSE
    - Bulk deals from NSE/BSE
    - Block deals from NSE/BSE
    - SAST filings (insider trading)

    These are high-alpha signals for Indian markets.
    """

    # NSE Endpoints
    NSE_BASE_URL = "https://www.nseindia.com"
    NSE_FII_DII_API = "https://www.nseindia.com/api/fiidiiTradeReact"
    NSE_BULK_DEALS_API = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
    NSE_BLOCK_DEALS_API = "https://www.nseindia.com/api/block-deal"

    # BSE Endpoints
    BSE_BASE_URL = "https://www.bseindia.com"
    BSE_BULK_DEALS_API = "https://api.bseindia.com/BseIndiaAPI/api/BulkDeal/w"
    BSE_BLOCK_DEALS_API = "https://api.bseindia.com/BseIndiaAPI/api/BlockDeal/w"

    # SEBI SAST endpoint (insider trading disclosures)
    SEBI_SAST_URL = "https://www.sebi.gov.in/sebiweb/other/OtherAction.do"

    # Headers for NSE (no brotli)
    NSE_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Referer': 'https://www.nseindia.com/',
        'Origin': 'https://www.nseindia.com',
        'Connection': 'keep-alive',
    }

    # Headers for BSE
    BSE_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.bseindia.com/',
        'Origin': 'https://www.bseindia.com',
    }

    def __init__(
        self,
        db_path: str = "data/cache/institutional.db"
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.rate_limit = 1.0
        self.last_request_time = 0
        self.logger = logging.getLogger(__name__)
        self._nse_session = None
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for caching institutional data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # FII/DII activity table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fii_dii_activity (
                id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                investor_type TEXT NOT NULL,
                buy_value REAL,
                sell_value REAL,
                net_value REAL,
                category TEXT,
                exchange TEXT,
                fetched_at TEXT NOT NULL
            )
        """)

        # Bulk/Block deals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bulk_block_deals (
                id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                ticker TEXT,
                company_name TEXT,
                client_name TEXT,
                deal_type TEXT,
                trade_type TEXT,
                quantity INTEGER,
                price REAL,
                value REAL,
                exchange TEXT,
                remarks TEXT,
                fetched_at TEXT NOT NULL
            )
        """)

        # SAST filings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sast_filings (
                id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                ticker TEXT,
                company_name TEXT,
                acquirer_name TEXT,
                acquirer_type TEXT,
                trade_type TEXT,
                shares_before INTEGER,
                shares_acquired INTEGER,
                shares_after INTEGER,
                percent_before REAL,
                percent_after REAL,
                value REAL,
                mode TEXT,
                exchange TEXT,
                fetched_at TEXT NOT NULL
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fii_dii_date ON fii_dii_activity(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bulk_block_date ON bulk_block_deals(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bulk_block_ticker ON bulk_block_deals(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sast_date ON sast_filings(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sast_ticker ON sast_filings(ticker)")

        conn.commit()
        conn.close()

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _generate_id(self, *args) -> str:
        """Generate unique ID from arguments."""
        content = ":".join(str(a) for a in args)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _get_nse_session(self) -> requests.Session:
        """Get or create NSE session with required cookies."""
        if self._nse_session is None:
            self._nse_session = requests.Session()
            self._nse_session.headers.update(self.NSE_HEADERS)

            try:
                self._rate_limit_wait()
                response = self._nse_session.get(
                    self.NSE_BASE_URL,
                    timeout=10
                )
                self.logger.debug(f"NSE session initialized, cookies: {len(self._nse_session.cookies)}")

                # Sometimes need to visit API endpoint to fully initialize
                if len(self._nse_session.cookies) == 0:
                    self._rate_limit_wait()
                    self._nse_session.get(
                        "https://www.nseindia.com/api/marketStatus",
                        timeout=10
                    )
            except Exception as e:
                self.logger.warning(f"Error initializing NSE session: {e}")

        return self._nse_session

    def _reset_nse_session(self):
        """Reset NSE session if it becomes invalid."""
        self._nse_session = None

    # =========================================================================
    # FII/DII Activity
    # =========================================================================

    def fetch_fii_dii_activity(
        self,
        days_back: int = 30
    ) -> List[InstitutionalActivity]:
        """
        Fetch FII/DII daily activity from NSE.

        Returns daily buy/sell values for FII and DII in cash segment.
        High positive net indicates bullish sentiment, negative indicates bearish.
        """
        activities = []
        session = self._get_nse_session()

        try:
            self._rate_limit_wait()

            response = session.get(
                self.NSE_FII_DII_API,
                timeout=15
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"NSE FII/DII not JSON: {response.text[:200]}")
                    self._reset_nse_session()
                    return activities

                # Process FII data
                for item in data:
                    try:
                        category = item.get('category', '')
                        date_str = item.get('date', '')

                        # Parse date
                        if date_str:
                            try:
                                parsed_date = datetime.strptime(date_str, '%d-%b-%Y')
                                date_iso = parsed_date.isoformat()[:10]
                            except ValueError:
                                date_iso = date_str
                        else:
                            date_iso = datetime.now().isoformat()[:10]

                        # FII data
                        fii_buy = self._parse_crore_value(item.get('fii_BuyValue', 0))
                        fii_sell = self._parse_crore_value(item.get('fii_SellValue', 0))
                        fii_net = self._parse_crore_value(item.get('fii_NetValue', 0))

                        if fii_buy or fii_sell:
                            fii_activity = InstitutionalActivity(
                                date=date_iso,
                                investor_type='FII',
                                buy_value=fii_buy,
                                sell_value=fii_sell,
                                net_value=fii_net,
                                category=category,
                                exchange='NSE'
                            )
                            activities.append(fii_activity)

                        # DII data
                        dii_buy = self._parse_crore_value(item.get('dii_BuyValue', 0))
                        dii_sell = self._parse_crore_value(item.get('dii_SellValue', 0))
                        dii_net = self._parse_crore_value(item.get('dii_NetValue', 0))

                        if dii_buy or dii_sell:
                            dii_activity = InstitutionalActivity(
                                date=date_iso,
                                investor_type='DII',
                                buy_value=dii_buy,
                                sell_value=dii_sell,
                                net_value=dii_net,
                                category=category,
                                exchange='NSE'
                            )
                            activities.append(dii_activity)

                    except Exception as e:
                        self.logger.warning(f"Error parsing FII/DII item: {e}")
                        continue

                self.logger.info(f"Fetched {len(activities)} FII/DII activity records")

            elif response.status_code == 401:
                self._reset_nse_session()
            else:
                self.logger.error(f"NSE FII/DII error: {response.status_code}")

        except requests.RequestException as e:
            self.logger.error(f"Error fetching FII/DII: {e}")

        return activities

    def _parse_crore_value(self, value: Any) -> float:
        """Parse value to float, handling string formats."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove commas and convert
            clean = value.replace(',', '').replace(' ', '')
            if clean and clean != '-':
                try:
                    return float(clean)
                except ValueError:
                    return 0.0
        return 0.0

    def get_fii_dii_trend(
        self,
        days: int = 5
    ) -> Dict[str, Any]:
        """
        Get FII/DII trend analysis for recent days.

        Returns sentiment indicators based on net flows.
        """
        activities = self.fetch_fii_dii_activity(days_back=days)

        # Aggregate by investor type
        fii_net_total = sum(a.net_value for a in activities if a.investor_type == 'FII')
        dii_net_total = sum(a.net_value for a in activities if a.investor_type == 'DII')

        # Count positive days
        fii_positive_days = sum(1 for a in activities if a.investor_type == 'FII' and a.is_bullish)
        dii_positive_days = sum(1 for a in activities if a.investor_type == 'DII' and a.is_bullish)

        fii_days = sum(1 for a in activities if a.investor_type == 'FII')
        dii_days = sum(1 for a in activities if a.investor_type == 'DII')

        return {
            'fii': {
                'net_total_cr': fii_net_total,
                'positive_days': fii_positive_days,
                'total_days': fii_days,
                'sentiment': 'bullish' if fii_net_total > 500 else ('bearish' if fii_net_total < -500 else 'neutral')
            },
            'dii': {
                'net_total_cr': dii_net_total,
                'positive_days': dii_positive_days,
                'total_days': dii_days,
                'sentiment': 'bullish' if dii_net_total > 500 else ('bearish' if dii_net_total < -500 else 'neutral')
            },
            'combined_sentiment': self._get_combined_sentiment(fii_net_total, dii_net_total),
            'analysis_period_days': days
        }

    def _get_combined_sentiment(self, fii_net: float, dii_net: float) -> str:
        """Determine combined institutional sentiment."""
        total_net = fii_net + dii_net

        if total_net > 1000:
            return 'strongly_bullish'
        elif total_net > 500:
            return 'bullish'
        elif total_net < -1000:
            return 'strongly_bearish'
        elif total_net < -500:
            return 'bearish'
        elif fii_net > 500 and dii_net < -500:
            return 'mixed_fii_bullish'
        elif fii_net < -500 and dii_net > 500:
            return 'mixed_dii_bullish'
        else:
            return 'neutral'

    # =========================================================================
    # Bulk Deals
    # =========================================================================

    def fetch_nse_bulk_deals(
        self,
        days_back: int = 7
    ) -> List[BulkBlockDeal]:
        """
        Fetch bulk deals from NSE.

        Bulk deals: Transactions > 0.5% of total shares.
        High alpha signal when promoters or institutions are buying.
        """
        deals = []
        session = self._get_nse_session()

        try:
            self._rate_limit_wait()

            response = session.get(
                self.NSE_BULK_DEALS_API,
                timeout=15
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"NSE bulk deals not JSON: {response.text[:200]}")
                    self._reset_nse_session()
                    return deals

                # NSE returns data in BLOCK_DEALS_DATA or BULK_DEALS_DATA
                bulk_data = data.get('BULK_DEALS_DATA', []) or data.get('data', [])

                for item in bulk_data:
                    try:
                        # Parse date
                        date_str = item.get('BD_DT_DATE', '') or item.get('mTIMESTAMP', '')
                        if date_str:
                            try:
                                parsed_date = datetime.strptime(date_str.split('T')[0], '%d-%b-%Y')
                                date_iso = parsed_date.isoformat()[:10]
                            except ValueError:
                                try:
                                    parsed_date = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
                                    date_iso = parsed_date.isoformat()[:10]
                                except ValueError:
                                    date_iso = date_str
                        else:
                            date_iso = datetime.now().isoformat()[:10]

                        quantity = int(item.get('BD_QTY_TRD', 0) or item.get('quantity', 0))
                        price = float(item.get('BD_TP_WATP', 0) or item.get('price', 0))

                        # Calculate value in crores
                        value = (quantity * price) / 10000000 if quantity and price else 0

                        # Determine trade type from remarks or client type
                        remarks = item.get('BD_REMARKS', '') or ''
                        trade_type = 'buy' if 'buy' in remarks.lower() else ('sell' if 'sell' in remarks.lower() else 'unknown')

                        deal = BulkBlockDeal(
                            id=self._generate_id(
                                item.get('BD_SYMBOL', ''),
                                item.get('BD_CLIENT_NAME', ''),
                                date_str,
                                quantity
                            ),
                            date=date_iso,
                            ticker=item.get('BD_SYMBOL', '') or item.get('symbol', ''),
                            company_name=item.get('BD_SCRIP_NAME', '') or item.get('company', ''),
                            client_name=item.get('BD_CLIENT_NAME', '') or item.get('clientName', ''),
                            deal_type='bulk',
                            trade_type=trade_type,
                            quantity=quantity,
                            price=price,
                            value=value,
                            exchange='NSE',
                            remarks=remarks
                        )
                        deals.append(deal)

                    except Exception as e:
                        self.logger.warning(f"Error parsing NSE bulk deal: {e}")
                        continue

                self.logger.info(f"Fetched {len(deals)} NSE bulk deals")

            elif response.status_code == 401:
                self._reset_nse_session()
            else:
                self.logger.error(f"NSE bulk deals error: {response.status_code}")

        except requests.RequestException as e:
            self.logger.error(f"Error fetching NSE bulk deals: {e}")

        return deals

    def fetch_bse_bulk_deals(
        self,
        days_back: int = 7
    ) -> List[BulkBlockDeal]:
        """Fetch bulk deals from BSE."""
        deals = []

        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
        to_date = datetime.now().strftime('%Y%m%d')

        params = {
            'ddlfromdate': from_date,
            'ddltodate': to_date,
        }

        try:
            self._rate_limit_wait()

            response = requests.get(
                self.BSE_BULK_DEALS_API,
                headers=self.BSE_HEADERS,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"BSE bulk deals not JSON: {response.text[:200]}")
                    return deals

                for item in data.get('Table', []):
                    try:
                        date_str = item.get('DT_TM', '')
                        if date_str:
                            try:
                                parsed_date = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
                                date_iso = parsed_date.isoformat()[:10]
                            except ValueError:
                                date_iso = date_str
                        else:
                            date_iso = datetime.now().isoformat()[:10]

                        quantity = int(item.get('NOOFTRD', 0) or 0)
                        price = float(item.get('HIGPRIC', 0) or item.get('AVGPRIC', 0) or 0)

                        # Calculate value in crores
                        value = (quantity * price) / 10000000 if quantity and price else 0

                        # Determine trade type
                        trade_type_code = item.get('BUYSELL', '')
                        trade_type = 'buy' if trade_type_code == 'B' else ('sell' if trade_type_code == 'S' else 'unknown')

                        scrip_cd = item.get('SCRIP_CD')
                        ticker = str(scrip_cd) if scrip_cd is not None else ''

                        deal = BulkBlockDeal(
                            id=self._generate_id(
                                ticker,
                                item.get('CLIENTNAME', ''),
                                date_str,
                                quantity
                            ),
                            date=date_iso,
                            ticker=ticker,
                            company_name=item.get('SLONGNAME', '') or item.get('SCRIP_NAME', ''),
                            client_name=item.get('CLIENTNAME', ''),
                            deal_type='bulk',
                            trade_type=trade_type,
                            quantity=quantity,
                            price=price,
                            value=value,
                            exchange='BSE',
                            remarks=item.get('REMARKS', '')
                        )
                        deals.append(deal)

                    except Exception as e:
                        self.logger.warning(f"Error parsing BSE bulk deal: {e}")
                        continue

                self.logger.info(f"Fetched {len(deals)} BSE bulk deals")

            else:
                self.logger.error(f"BSE bulk deals error: {response.status_code}")

        except requests.RequestException as e:
            self.logger.error(f"Error fetching BSE bulk deals: {e}")

        return deals

    def fetch_nse_block_deals(
        self,
        days_back: int = 7
    ) -> List[BulkBlockDeal]:
        """
        Fetch block deals from NSE.

        Block deals: Minimum quantity of 5 lakh shares or Rs 10 crore value.
        These are negotiated deals between large institutions.
        """
        deals = []
        session = self._get_nse_session()

        try:
            self._rate_limit_wait()

            response = session.get(
                self.NSE_BLOCK_DEALS_API,
                timeout=15
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"NSE block deals not JSON: {response.text[:200]}")
                    self._reset_nse_session()
                    return deals

                block_data = data.get('data', []) if isinstance(data, dict) else data

                for item in block_data:
                    try:
                        date_str = item.get('BD_DT_DATE', '') or item.get('date', '')
                        if date_str:
                            try:
                                parsed_date = datetime.strptime(date_str.split('T')[0], '%d-%b-%Y')
                                date_iso = parsed_date.isoformat()[:10]
                            except ValueError:
                                date_iso = date_str
                        else:
                            date_iso = datetime.now().isoformat()[:10]

                        quantity = int(item.get('BD_QTY_TRD', 0) or item.get('quantity', 0))
                        price = float(item.get('BD_TP_WATP', 0) or item.get('price', 0))
                        value = (quantity * price) / 10000000 if quantity and price else 0

                        deal = BulkBlockDeal(
                            id=self._generate_id(
                                item.get('BD_SYMBOL', ''),
                                item.get('BD_CLIENT_NAME', ''),
                                date_str,
                                quantity
                            ),
                            date=date_iso,
                            ticker=item.get('BD_SYMBOL', '') or item.get('symbol', ''),
                            company_name=item.get('BD_SCRIP_NAME', '') or '',
                            client_name=item.get('BD_CLIENT_NAME', '') or item.get('clientName', ''),
                            deal_type='block',
                            trade_type='unknown',
                            quantity=quantity,
                            price=price,
                            value=value,
                            exchange='NSE',
                            remarks=item.get('BD_REMARKS', '')
                        )
                        deals.append(deal)

                    except Exception as e:
                        self.logger.warning(f"Error parsing NSE block deal: {e}")
                        continue

                self.logger.info(f"Fetched {len(deals)} NSE block deals")

            elif response.status_code == 401:
                self._reset_nse_session()
            else:
                self.logger.error(f"NSE block deals error: {response.status_code}")

        except requests.RequestException as e:
            self.logger.error(f"Error fetching NSE block deals: {e}")

        return deals

    def fetch_all_bulk_block_deals(
        self,
        days_back: int = 7,
        ticker: Optional[str] = None
    ) -> List[BulkBlockDeal]:
        """
        Fetch all bulk and block deals from both NSE and BSE.

        Optionally filter by ticker.
        """
        all_deals = []

        # Fetch from all sources
        all_deals.extend(self.fetch_nse_bulk_deals(days_back))
        all_deals.extend(self.fetch_bse_bulk_deals(days_back))
        all_deals.extend(self.fetch_nse_block_deals(days_back))

        # Filter by ticker if specified
        if ticker:
            ticker_clean = ticker.replace('.NS', '').replace('.BO', '').upper()
            all_deals = [
                d for d in all_deals
                if ticker_clean in d.ticker.upper()
            ]

        # Sort by date (newest first) then by value
        all_deals.sort(key=lambda x: (x.date, -x.value), reverse=True)

        return all_deals

    def get_stock_deal_activity(
        self,
        ticker: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get bulk/block deal activity analysis for a specific stock.

        Returns:
        - Total buy/sell volume from deals
        - Major clients involved
        - Sentiment based on deal flow
        """
        ticker_clean = ticker.replace('.NS', '').replace('.BO', '').upper()
        deals = self.fetch_all_bulk_block_deals(days_back=days_back, ticker=ticker_clean)

        if not deals:
            return {
                'ticker': ticker_clean,
                'total_deals': 0,
                'sentiment': 'no_data'
            }

        # Aggregate
        total_buy_value = sum(d.value for d in deals if d.trade_type == 'buy')
        total_sell_value = sum(d.value for d in deals if d.trade_type == 'sell')
        total_buy_qty = sum(d.quantity for d in deals if d.trade_type == 'buy')
        total_sell_qty = sum(d.quantity for d in deals if d.trade_type == 'sell')

        # Find major clients
        from collections import Counter
        clients = Counter(d.client_name for d in deals if d.client_name)
        top_clients = clients.most_common(5)

        # Determine sentiment
        if total_buy_value > total_sell_value * 1.5:
            sentiment = 'bullish'
        elif total_sell_value > total_buy_value * 1.5:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        return {
            'ticker': ticker_clean,
            'total_deals': len(deals),
            'buy': {
                'count': sum(1 for d in deals if d.trade_type == 'buy'),
                'value_cr': round(total_buy_value, 2),
                'quantity': total_buy_qty
            },
            'sell': {
                'count': sum(1 for d in deals if d.trade_type == 'sell'),
                'value_cr': round(total_sell_value, 2),
                'quantity': total_sell_qty
            },
            'net_value_cr': round(total_buy_value - total_sell_value, 2),
            'top_clients': top_clients,
            'sentiment': sentiment,
            'analysis_period_days': days_back
        }

    # =========================================================================
    # SAST Filings (Insider Trading)
    # =========================================================================

    def fetch_nse_insider_trading(
        self,
        ticker: Optional[str] = None,
        days_back: int = 30
    ) -> List[SASTFiling]:
        """
        Fetch SAST (insider trading) disclosures from NSE.

        Insider buying is generally a bullish signal as insiders have
        non-public information about company prospects.
        """
        filings = []
        session = self._get_nse_session()

        # NSE insider trading endpoint
        insider_api = "https://www.nseindia.com/api/corporates-pit"

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
                insider_api,
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    self.logger.error(f"NSE insider not JSON: {response.text[:200]}")
                    self._reset_nse_session()
                    return filings

                insider_data = data.get('data', []) if isinstance(data, dict) else data

                for item in insider_data:
                    try:
                        # Parse acquisition date
                        acq_date = item.get('acqfromDt', '') or item.get('date', '')
                        if acq_date:
                            try:
                                parsed_date = datetime.strptime(acq_date.split('T')[0], '%d-%b-%Y')
                                date_iso = parsed_date.isoformat()[:10]
                            except ValueError:
                                date_iso = acq_date
                        else:
                            date_iso = datetime.now().isoformat()[:10]

                        # Determine trade type
                        acq_mode = (item.get('acqMode', '') or '').lower()
                        trade_type = 'buy' if 'acquisition' in acq_mode or 'purchase' in acq_mode else 'sell'

                        # Parse shareholding percentages
                        pct_before = self._parse_percent(item.get('befAcqSharesPerc', 0))
                        pct_after = self._parse_percent(item.get('aftAcqSharesPerc', 0))

                        filing = SASTFiling(
                            id=self._generate_id(
                                item.get('symbol', ''),
                                item.get('acquirerName', ''),
                                acq_date
                            ),
                            date=date_iso,
                            ticker=item.get('symbol', ''),
                            company_name=item.get('company', ''),
                            acquirer_name=item.get('acquirerName', ''),
                            acquirer_type=item.get('personCategory', ''),
                            trade_type=trade_type,
                            shares_before=int(item.get('befAcqSharesNo', 0) or 0),
                            shares_acquired=int(item.get('secAcq', 0) or 0),
                            shares_after=int(item.get('afterAcqSharesNo', 0) or 0),
                            percent_before=pct_before,
                            percent_after=pct_after,
                            value=float(item.get('secVal', 0) or 0),
                            mode=item.get('acqMode', ''),
                            exchange='NSE'
                        )
                        filings.append(filing)

                    except Exception as e:
                        self.logger.warning(f"Error parsing NSE insider: {e}")
                        continue

                self.logger.info(f"Fetched {len(filings)} NSE insider trading filings")

            elif response.status_code == 401:
                self._reset_nse_session()
            else:
                self.logger.error(f"NSE insider error: {response.status_code}")

        except requests.RequestException as e:
            self.logger.error(f"Error fetching NSE insider: {e}")

        return filings

    def _parse_percent(self, value: Any) -> float:
        """Parse percentage value."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            clean = value.replace('%', '').replace(',', '').strip()
            if clean:
                try:
                    return float(clean)
                except ValueError:
                    return 0.0
        return 0.0

    def get_insider_sentiment(
        self,
        ticker: str,
        days_back: int = 90
    ) -> Dict[str, Any]:
        """
        Get insider trading sentiment for a stock.

        Returns analysis of insider buying/selling activity.
        """
        ticker_clean = ticker.replace('.NS', '').replace('.BO', '').upper()
        filings = self.fetch_nse_insider_trading(ticker=ticker_clean, days_back=days_back)

        if not filings:
            return {
                'ticker': ticker_clean,
                'total_filings': 0,
                'sentiment': 'no_data'
            }

        # Aggregate
        buys = [f for f in filings if f.is_bullish]
        sells = [f for f in filings if not f.is_bullish]

        total_buy_value = sum(f.value for f in buys)
        total_sell_value = sum(f.value for f in sells)

        # Find key insiders
        from collections import Counter
        buyers = Counter(f.acquirer_name for f in buys if f.acquirer_name)
        sellers = Counter(f.acquirer_name for f in sells if f.acquirer_name)

        # Check for promoter activity
        promoter_buys = [f for f in buys if 'promoter' in f.acquirer_type.lower()]
        promoter_sells = [f for f in sells if 'promoter' in f.acquirer_type.lower()]

        # Determine sentiment
        if len(promoter_buys) > 0 and len(promoter_sells) == 0:
            sentiment = 'strongly_bullish'
        elif total_buy_value > total_sell_value * 2:
            sentiment = 'bullish'
        elif len(promoter_sells) > 0 and len(promoter_buys) == 0:
            sentiment = 'strongly_bearish'
        elif total_sell_value > total_buy_value * 2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        return {
            'ticker': ticker_clean,
            'total_filings': len(filings),
            'buys': {
                'count': len(buys),
                'value_lakhs': round(total_buy_value, 2),
                'promoter_buys': len(promoter_buys),
                'top_buyers': buyers.most_common(3)
            },
            'sells': {
                'count': len(sells),
                'value_lakhs': round(total_sell_value, 2),
                'promoter_sells': len(promoter_sells),
                'top_sellers': sellers.most_common(3)
            },
            'net_value_lakhs': round(total_buy_value - total_sell_value, 2),
            'sentiment': sentiment,
            'analysis_period_days': days_back
        }

    # =========================================================================
    # Storage
    # =========================================================================

    def save_fii_dii_activity(self, activities: List[InstitutionalActivity]):
        """Save FII/DII activity to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for activity in activities:
            cursor.execute("""
                INSERT OR REPLACE INTO fii_dii_activity
                (id, date, investor_type, buy_value, sell_value, net_value, category, exchange, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self._generate_id(activity.date, activity.investor_type, activity.category),
                activity.date,
                activity.investor_type,
                activity.buy_value,
                activity.sell_value,
                activity.net_value,
                activity.category,
                activity.exchange,
                datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()

    def save_bulk_block_deals(self, deals: List[BulkBlockDeal]):
        """Save bulk/block deals to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for deal in deals:
            cursor.execute("""
                INSERT OR REPLACE INTO bulk_block_deals
                (id, date, ticker, company_name, client_name, deal_type, trade_type,
                 quantity, price, value, exchange, remarks, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                deal.id,
                deal.date,
                deal.ticker,
                deal.company_name,
                deal.client_name,
                deal.deal_type,
                deal.trade_type,
                deal.quantity,
                deal.price,
                deal.value,
                deal.exchange,
                deal.remarks,
                datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()

    def save_sast_filings(self, filings: List[SASTFiling]):
        """Save SAST filings to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for filing in filings:
            cursor.execute("""
                INSERT OR REPLACE INTO sast_filings
                (id, date, ticker, company_name, acquirer_name, acquirer_type, trade_type,
                 shares_before, shares_acquired, shares_after, percent_before, percent_after,
                 value, mode, exchange, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filing.id,
                filing.date,
                filing.ticker,
                filing.company_name,
                filing.acquirer_name,
                filing.acquirer_type,
                filing.trade_type,
                filing.shares_before,
                filing.shares_acquired,
                filing.shares_after,
                filing.percent_before,
                filing.percent_after,
                filing.value,
                filing.mode,
                filing.exchange,
                datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict:
        """Get statistics about stored institutional data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # FII/DII stats
        cursor.execute("SELECT COUNT(*) FROM fii_dii_activity")
        stats['fii_dii_records'] = cursor.fetchone()[0]

        # Bulk/Block stats
        cursor.execute("SELECT COUNT(*) FROM bulk_block_deals")
        stats['bulk_block_deals'] = cursor.fetchone()[0]

        cursor.execute("SELECT deal_type, COUNT(*) FROM bulk_block_deals GROUP BY deal_type")
        stats['deals_by_type'] = dict(cursor.fetchall())

        # SAST stats
        cursor.execute("SELECT COUNT(*) FROM sast_filings")
        stats['sast_filings'] = cursor.fetchone()[0]

        conn.close()

        return stats

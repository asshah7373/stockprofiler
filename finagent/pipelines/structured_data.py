"""
Structured Market Data Pipeline
Fetches OHLCV data from multiple sources with fallback architecture.
"""

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class MarketDataPipeline:
    """
    Fetches and caches market data with fallback sources.

    Priority:
    1. Local cache (SQLite)
    2. Broker API (if configured)
    3. yfinance
    4. nsepython (for live quotes)
    """

    # Market cap categories for Indian markets (in Crores)
    MARKET_CAP_THRESHOLDS = {
        "LARGE": 20000,   # > 20,000 Cr
        "MID": 5000,      # 5,000 - 20,000 Cr
        "SMALL": 0        # < 5,000 Cr
    }

    def __init__(self, cache_db: str = "data/cache/market_data.db"):
        self.cache_db = Path(cache_db)
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._init_cache()

    def _init_cache(self):
        """Initialize SQLite cache database."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        # Table for OHLCV data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                fetched_at TEXT NOT NULL,
                UNIQUE(ticker, date)
            )
        """)

        # Table for company info
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS company_info_cache (
                ticker TEXT PRIMARY KEY,
                info_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            )
        """)

        # Table for live quotes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_quotes (
                ticker TEXT PRIMARY KEY,
                price REAL NOT NULL,
                change REAL,
                change_percent REAL,
                volume INTEGER,
                fetched_at TEXT NOT NULL
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date
            ON ohlcv_cache(ticker, date)
        """)

        conn.commit()
        conn.close()
        self.logger.info(f"Cache database initialized at {self.cache_db}")

    def _normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker symbol for yfinance."""
        ticker = ticker.upper().strip()
        if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
            ticker = f"{ticker}.NS"  # Default to NSE
        return ticker

    def _check_cache(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Check if data exists in cache."""
        conn = sqlite3.connect(self.cache_db)

        query = "SELECT * FROM ohlcv_cache WHERE ticker = ?"
        params = [ticker]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return None

        # Check if cache is stale (more than 1 day old for historical data)
        if not df.empty:
            fetched_at = datetime.fromisoformat(df['fetched_at'].iloc[-1])
            if datetime.now() - fetched_at > timedelta(days=1):
                self.logger.info(f"Cache stale for {ticker}, will refresh")
                return None

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'adj_close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        return df

    def _save_to_cache(self, ticker: str, df: pd.DataFrame):
        """Save OHLCV data to cache."""
        if df.empty:
            return

        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        fetched_at = datetime.now().isoformat()

        for date, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO ohlcv_cache
                    (ticker, date, open, high, low, close, adj_close, volume, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    date.strftime('%Y-%m-%d'),
                    float(row.get('Open', 0)),
                    float(row.get('High', 0)),
                    float(row.get('Low', 0)),
                    float(row.get('Close', 0)),
                    float(row.get('Adj Close', row.get('Close', 0))),
                    int(row.get('Volume', 0)),
                    fetched_at
                ))
            except Exception as e:
                self.logger.warning(f"Error caching row for {ticker} on {date}: {e}")

        conn.commit()
        conn.close()
        self.logger.info(f"Cached {len(df)} rows for {ticker}")

    def get_historical_data(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            ticker: Stock symbol (e.g., "RELIANCE.NS" for NSE)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, 20y, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
        """
        ticker = self._normalize_ticker(ticker)

        # Try cache first
        if use_cache:
            cached = self._check_cache(ticker)
            if cached is not None and len(cached) > 0:
                self.logger.info(f"Using cached data for {ticker}")
                return cached

        # Fetch from yfinance with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Fetching {ticker} from yfinance (attempt {attempt + 1})")
                stock = yf.Ticker(ticker)
                df = stock.history(period=period, interval=interval)

                if df.empty:
                    self.logger.warning(f"No data returned for {ticker}")
                    return pd.DataFrame()

                # Cache the data
                self._save_to_cache(ticker, df)

                return df

            except Exception as e:
                self.logger.error(f"Error fetching {ticker} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        return pd.DataFrame()

    def get_live_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch current price and basic info.

        Returns:
            {
                "ticker": str,
                "price": float,
                "change": float,
                "change_percent": float,
                "volume": int,
                "timestamp": str,
                "source": str
            }
        """
        ticker = self._normalize_ticker(ticker)

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get current price
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            prev_close = info.get('regularMarketPreviousClose', info.get('previousClose'))

            if price is None:
                # Try to get from history
                hist = stock.history(period='1d')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    prev_close = float(hist['Open'].iloc[0])

            change = None
            change_percent = None
            if price and prev_close:
                change = price - prev_close
                change_percent = (change / prev_close) * 100

            quote = {
                "ticker": ticker,
                "price": price,
                "change": change,
                "change_percent": change_percent,
                "volume": info.get('regularMarketVolume', info.get('volume')),
                "day_high": info.get('dayHigh'),
                "day_low": info.get('dayLow'),
                "open": info.get('open'),
                "prev_close": prev_close,
                "timestamp": datetime.now().isoformat(),
                "source": "yfinance"
            }

            # Cache the quote
            self._cache_live_quote(quote)

            return quote

        except Exception as e:
            self.logger.error(f"Error fetching live quote for {ticker}: {e}")
            return {
                "ticker": ticker,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _cache_live_quote(self, quote: Dict[str, Any]):
        """Cache live quote to database."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO live_quotes
            (ticker, price, change, change_percent, volume, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            quote['ticker'],
            quote.get('price'),
            quote.get('change'),
            quote.get('change_percent'),
            quote.get('volume'),
            quote['timestamp']
        ))

        conn.commit()
        conn.close()

    def get_company_info(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Fetch company fundamentals from yfinance.

        Returns comprehensive company information including:
        - Basic info (name, sector, industry)
        - Financial metrics (P/E, P/B, EPS, etc.)
        - Market data (market cap, volume, etc.)
        """
        ticker = self._normalize_ticker(ticker)

        # Check cache first
        if use_cache:
            cached = self._get_cached_company_info(ticker)
            if cached:
                return cached

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Structure the response
            company_info = {
                "ticker": ticker,
                "name": info.get('longName') or info.get('shortName'),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "country": info.get('country'),
                "website": info.get('website'),
                "description": info.get('longBusinessSummary'),

                # Market data
                "market_cap": info.get('marketCap'),
                "market_cap_category": self._categorize_market_cap(info.get('marketCap')),
                "enterprise_value": info.get('enterpriseValue'),
                "shares_outstanding": info.get('sharesOutstanding'),
                "float_shares": info.get('floatShares'),

                # Valuation metrics
                "pe_ratio": info.get('trailingPE'),
                "forward_pe": info.get('forwardPE'),
                "peg_ratio": info.get('pegRatio'),
                "price_to_book": info.get('priceToBook'),
                "price_to_sales": info.get('priceToSalesTrailing12Months'),
                "enterprise_to_revenue": info.get('enterpriseToRevenue'),
                "enterprise_to_ebitda": info.get('enterpriseToEbitda'),

                # Profitability
                "profit_margin": info.get('profitMargins'),
                "operating_margin": info.get('operatingMargins'),
                "gross_margin": info.get('grossMargins'),
                "return_on_equity": info.get('returnOnEquity'),
                "return_on_assets": info.get('returnOnAssets'),

                # Per share data
                "eps": info.get('trailingEps'),
                "forward_eps": info.get('forwardEps'),
                "book_value": info.get('bookValue'),
                "revenue_per_share": info.get('revenuePerShare'),

                # Financials
                "total_revenue": info.get('totalRevenue'),
                "revenue_growth": info.get('revenueGrowth'),
                "gross_profit": info.get('grossProfits'),
                "ebitda": info.get('ebitda'),
                "net_income": info.get('netIncomeToCommon'),
                "total_debt": info.get('totalDebt'),
                "total_cash": info.get('totalCash'),
                "debt_to_equity": info.get('debtToEquity'),
                "current_ratio": info.get('currentRatio'),
                "quick_ratio": info.get('quickRatio'),

                # Dividends
                "dividend_rate": info.get('dividendRate'),
                "dividend_yield": info.get('dividendYield'),
                "payout_ratio": info.get('payoutRatio'),
                "ex_dividend_date": info.get('exDividendDate'),

                # Price data
                "current_price": info.get('currentPrice') or info.get('regularMarketPrice'),
                "previous_close": info.get('previousClose'),
                "fifty_two_week_high": info.get('fiftyTwoWeekHigh'),
                "fifty_two_week_low": info.get('fiftyTwoWeekLow'),
                "fifty_day_average": info.get('fiftyDayAverage'),
                "two_hundred_day_average": info.get('twoHundredDayAverage'),
                "beta": info.get('beta'),

                # Volume
                "average_volume": info.get('averageVolume'),
                "average_volume_10d": info.get('averageVolume10days'),

                # Analyst data
                "target_mean_price": info.get('targetMeanPrice'),
                "target_high_price": info.get('targetHighPrice'),
                "target_low_price": info.get('targetLowPrice'),
                "recommendation_key": info.get('recommendationKey'),
                "number_of_analyst_opinions": info.get('numberOfAnalystOpinions'),

                # Metadata
                "fetched_at": datetime.now().isoformat(),
                "source": "yfinance"
            }

            # Cache the info
            self._cache_company_info(ticker, company_info)

            return company_info

        except Exception as e:
            self.logger.error(f"Error fetching company info for {ticker}: {e}")
            return {
                "ticker": ticker,
                "error": str(e),
                "fetched_at": datetime.now().isoformat()
            }

    def _categorize_market_cap(self, market_cap: Optional[int]) -> Optional[str]:
        """Categorize market cap (in Crores for Indian markets)."""
        if market_cap is None:
            return None

        # Convert to Crores (1 Cr = 10 Million)
        market_cap_cr = market_cap / 10_000_000

        if market_cap_cr >= self.MARKET_CAP_THRESHOLDS["LARGE"]:
            return "LARGE"
        elif market_cap_cr >= self.MARKET_CAP_THRESHOLDS["MID"]:
            return "MID"
        else:
            return "SMALL"

    def _get_cached_company_info(self, ticker: str) -> Optional[Dict]:
        """Get cached company info if not stale."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT info_json, fetched_at FROM company_info_cache WHERE ticker = ?",
            (ticker,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            fetched_at = datetime.fromisoformat(row[1])
            if datetime.now() - fetched_at < timedelta(hours=24):
                return json.loads(row[0])

        return None

    def _cache_company_info(self, ticker: str, info: Dict):
        """Cache company info to database."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO company_info_cache (ticker, info_json, fetched_at)
            VALUES (?, ?, ?)
        """, (ticker, json.dumps(info), datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def get_multiple_tickers(
        self,
        tickers: List[str],
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple tickers.

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_historical_data(ticker, period=period)
            except Exception as e:
                self.logger.error(f"Error fetching {ticker}: {e}")
                results[ticker] = pd.DataFrame()

        return results

    def get_index_data(
        self,
        index: str = "NIFTY50",
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Fetch index data for benchmarking.

        Supported indices:
        - NIFTY50: ^NSEI
        - SENSEX: ^BSESN
        - NIFTYBANK: ^NSEBANK
        """
        index_map = {
            "NIFTY50": "^NSEI",
            "NIFTY": "^NSEI",
            "SENSEX": "^BSESN",
            "NIFTYBANK": "^NSEBANK",
            "BANKNIFTY": "^NSEBANK"
        }

        yf_ticker = index_map.get(index.upper(), index)
        return self.get_historical_data(yf_ticker, period=period, use_cache=True)

    def clear_cache(self, ticker: Optional[str] = None):
        """Clear cache for a specific ticker or all data."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        if ticker:
            ticker = self._normalize_ticker(ticker)
            cursor.execute("DELETE FROM ohlcv_cache WHERE ticker = ?", (ticker,))
            cursor.execute("DELETE FROM company_info_cache WHERE ticker = ?", (ticker,))
            cursor.execute("DELETE FROM live_quotes WHERE ticker = ?", (ticker,))
            self.logger.info(f"Cleared cache for {ticker}")
        else:
            cursor.execute("DELETE FROM ohlcv_cache")
            cursor.execute("DELETE FROM company_info_cache")
            cursor.execute("DELETE FROM live_quotes")
            self.logger.info("Cleared all cache")

        conn.commit()
        conn.close()

"""
Perceiver Agent - Data Ingestion
Responsible for fetching data from all sources.
NO analysis, NO opinions, NO recommendations - data fetching ONLY.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..pipelines.structured_data import MarketDataPipeline
from ..pipelines.unstructured_data import CircularPipeline
from ..pipelines.sentiment_data import SentimentPipeline
from ..rag.vector_store import VectorStore
from ..rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)


class PerceiverAgent:
    """
    Data Ingestion Agent (Perceiver)

    Responsibilities:
    1. Fetch structured market data (OHLCV, company info)
    2. Fetch unstructured data (circulars, filings)
    3. Fetch news and sentiment data
    4. Store documents in vector database
    5. Retrieve relevant context for analysis

    CONSTRAINTS:
    - NO analysis or interpretation of data
    - NO opinions or recommendations
    - ONLY fetch, store, and retrieve data
    - ALWAYS return raw data with source information
    """

    def __init__(
        self,
        market_pipeline: Optional[MarketDataPipeline] = None,
        circular_pipeline: Optional[CircularPipeline] = None,
        sentiment_pipeline: Optional[SentimentPipeline] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.market = market_pipeline or MarketDataPipeline()
        self.circulars = circular_pipeline or CircularPipeline()
        self.sentiment = sentiment_pipeline or SentimentPipeline()
        self.vector_store = vector_store or VectorStore()
        self.retriever = RAGRetriever(self.vector_store)
        self.logger = logging.getLogger(__name__)

    def perceive(self, ticker: str) -> Dict[str, Any]:
        """
        Comprehensive data gathering for a ticker.

        Fetches all available data without analysis.

        Returns:
            {
                "ticker": str,
                "market_data": {...},
                "company_info": {...},
                "technical_data": {...},  # Raw price data for TA
                "sentiment_data": {...},
                "rag_context": {...},
                "sources": [...],
                "perceived_at": str
            }
        """
        self.logger.info(f"Perceiving data for {ticker}")

        result = {
            "ticker": ticker,
            "market_data": None,
            "company_info": None,
            "technical_data": None,
            "sentiment_data": None,
            "rag_context": None,
            "sources": [],
            "errors": [],
            "perceived_at": datetime.now().isoformat()
        }

        # 1. Fetch live quote
        try:
            result["market_data"] = self.fetch_market_data(ticker)
            result["sources"].append({
                "type": "market_data",
                "source": "yfinance",
                "fetched_at": datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            result["errors"].append(f"Market data: {str(e)}")

        # 2. Fetch company info
        try:
            result["company_info"] = self.fetch_company_info(ticker)
            result["sources"].append({
                "type": "company_info",
                "source": "yfinance",
                "fetched_at": datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"Error fetching company info: {e}")
            result["errors"].append(f"Company info: {str(e)}")

        # 3. Fetch historical data for technical analysis
        try:
            result["technical_data"] = self.fetch_historical_data(ticker)
            result["sources"].append({
                "type": "historical_data",
                "source": "yfinance",
                "fetched_at": datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            result["errors"].append(f"Historical data: {str(e)}")

        # 4. Fetch sentiment data
        try:
            result["sentiment_data"] = self.fetch_sentiment(ticker)
            result["sources"].append({
                "type": "sentiment",
                "source": "news_aggregator",
                "fetched_at": datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"Error fetching sentiment: {e}")
            result["errors"].append(f"Sentiment: {str(e)}")

        # 5. Retrieve RAG context
        try:
            result["rag_context"] = self.retrieve_context(ticker)
            result["sources"].append({
                "type": "documents",
                "source": "vector_store",
                "fetched_at": datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            result["errors"].append(f"RAG context: {str(e)}")

        return result

    def fetch_market_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch current market data.

        Returns:
            {
                "price": float,
                "change": float,
                "change_percent": float,
                "volume": int,
                "day_high": float,
                "day_low": float,
                "timestamp": str,
                "source": str
            }
        """
        quote = self.market.get_live_quote(ticker)

        return {
            "price": quote.get("price"),
            "change": quote.get("change"),
            "change_percent": quote.get("change_percent"),
            "volume": quote.get("volume"),
            "day_high": quote.get("day_high"),
            "day_low": quote.get("day_low"),
            "open": quote.get("open"),
            "prev_close": quote.get("prev_close"),
            "timestamp": quote.get("timestamp"),
            "source": quote.get("source", "yfinance")
        }

    def fetch_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch company fundamentals.

        Returns comprehensive company information.
        """
        return self.market.get_company_info(ticker)

    def fetch_historical_data(
        self,
        ticker: str,
        period: str = "1y"
    ) -> Dict[str, Any]:
        """
        Fetch historical OHLCV data.

        Returns data suitable for technical analysis.
        """
        df = self.market.get_historical_data(ticker, period=period)

        if df.empty:
            return {"error": "No historical data available"}

        return {
            "ticker": ticker,
            "period": period,
            "data_points": len(df),
            "start_date": df.index[0].isoformat() if len(df) > 0 else None,
            "end_date": df.index[-1].isoformat() if len(df) > 0 else None,
            "ohlcv": df.to_dict(orient='index'),
            "latest": {
                "open": float(df['Open'].iloc[-1]) if len(df) > 0 else None,
                "high": float(df['High'].iloc[-1]) if len(df) > 0 else None,
                "low": float(df['Low'].iloc[-1]) if len(df) > 0 else None,
                "close": float(df['Close'].iloc[-1]) if len(df) > 0 else None,
                "volume": int(df['Volume'].iloc[-1]) if len(df) > 0 else None
            },
            "source": "yfinance"
        }

    def fetch_sentiment(
        self,
        ticker: str,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Fetch news sentiment data.

        Returns aggregated sentiment and recent headlines.
        """
        return self.sentiment.get_sentiment_summary(ticker, days_back=days_back)

    def retrieve_context(
        self,
        ticker: str,
        query: str = "financial performance results outlook"
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents from vector store.

        Returns formatted context for analysis.
        """
        retrieval = self.retriever.retrieve_for_analysis(
            ticker=ticker,
            query=query,
            doc_types=["quarterly_results", "circulars", "news"],
            lookback_days=90,
            top_k=10
        )

        return {
            "chunks": retrieval.get("chunks", []),
            "sources": retrieval.get("sources", []),
            "formatted_context": self.retriever.format_context_for_llm(retrieval),
            "summary": self.retriever.get_retrieval_summary(retrieval)
        }

    def fetch_index_data(
        self,
        index: str = "NIFTY50",
        period: str = "1y"
    ) -> Dict[str, Any]:
        """
        Fetch benchmark index data.

        Used for beta calculation and market context.
        """
        df = self.market.get_index_data(index, period=period)

        if df.empty:
            return {"error": "No index data available"}

        return {
            "index": index,
            "period": period,
            "data_points": len(df),
            "ohlcv": df.to_dict(orient='index'),
            "latest_close": float(df['Close'].iloc[-1]) if len(df) > 0 else None,
            "source": "yfinance"
        }

    def update_circular_database(
        self,
        exchange: str = "NSE",
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Fetch and process recent circulars.

        Stores processed circulars in vector database.
        """
        self.logger.info(f"Updating circular database from {exchange}")

        circulars = self.circulars.fetch_recent_circulars(
            exchange=exchange,
            days_back=days_back
        )

        processed = 0
        failed = 0

        for circular in circulars:
            try:
                result = self.circulars.process_circular(circular)

                # Add to vector store if we have chunks
                if result.get('chunks'):
                    self.vector_store.add_document(
                        collection="circulars",
                        doc_id=circular.get('id'),
                        chunks=result['chunks'],
                        metadata={
                            "ticker": circular.get('ticker'),
                            "title": circular.get('title'),
                            "doc_type": circular.get('doc_type'),
                            "filing_date": circular.get('filing_date'),
                            "exchange": exchange
                        }
                    )
                    processed += 1

            except Exception as e:
                self.logger.error(f"Error processing circular: {e}")
                failed += 1

        return {
            "exchange": exchange,
            "days_back": days_back,
            "circulars_found": len(circulars),
            "processed": processed,
            "failed": failed,
            "updated_at": datetime.now().isoformat()
        }

    def perceive_sector(self, sector: str) -> Dict[str, Any]:
        """
        Gather sector-level data.

        Returns sentiment and top movers for a sector.
        """
        return self.sentiment.get_sector_sentiment(sector)

    def perceive_market(self) -> Dict[str, Any]:
        """
        Gather overall market data.

        Returns market sentiment and sector breakdown.
        """
        return self.sentiment.get_market_sentiment()

    def search_documents(
        self,
        query: str,
        ticker: Optional[str] = None,
        collection: str = "circulars"
    ) -> List[Dict]:
        """
        Search stored documents.

        Returns matching document chunks.
        """
        filters = {}
        if ticker:
            filters['ticker'] = ticker.replace('.NS', '').replace('.BO', '').upper()

        return self.vector_store.search(
            collection=collection,
            query=query,
            filters=filters,
            top_k=10
        )

    def get_data_status(self) -> Dict[str, Any]:
        """
        Get status of available data.

        Returns statistics about cached and stored data.
        """
        return {
            "vector_store": self.vector_store.get_all_stats(),
            "status_at": datetime.now().isoformat()
        }

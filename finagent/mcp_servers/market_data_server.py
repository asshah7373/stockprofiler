"""
MCP Server for Market Data
Provides stock price and company data via MCP protocol.
"""

import asyncio
import json
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# MCP Server implementation
# Note: This is a simplified implementation. For production use,
# integrate with the official MCP SDK.


class MarketDataServer:
    """
    MCP Server for market data operations.

    Tools provided:
    - get_ticker_price: Get current stock price
    - get_historical_data: Get historical OHLCV data
    - get_company_info: Get company fundamentals
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Lazy import to avoid circular dependencies
        from ..pipelines.structured_data import MarketDataPipeline
        self.pipeline = MarketDataPipeline()

        self.tools = {
            "get_ticker_price": self.get_ticker_price,
            "get_historical_data": self.get_historical_data,
            "get_company_info": self.get_company_info,
            "get_index_data": self.get_index_data
        }

    def get_tool_definitions(self) -> list:
        """Return tool definitions for MCP registration."""
        return [
            {
                "name": "get_ticker_price",
                "description": "Get current price for a stock ticker",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock symbol (e.g., RELIANCE.NS)"
                        }
                    },
                    "required": ["ticker"]
                }
            },
            {
                "name": "get_historical_data",
                "description": "Get historical OHLCV data for a stock",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "period": {
                            "type": "string",
                            "default": "1y",
                            "description": "Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)"
                        },
                        "interval": {
                            "type": "string",
                            "default": "1d",
                            "description": "Data interval (1d, 1wk, 1mo)"
                        }
                    },
                    "required": ["ticker"]
                }
            },
            {
                "name": "get_company_info",
                "description": "Get company fundamentals and information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"}
                    },
                    "required": ["ticker"]
                }
            },
            {
                "name": "get_index_data",
                "description": "Get benchmark index data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "string",
                            "default": "NIFTY50",
                            "description": "Index name (NIFTY50, SENSEX, NIFTYBANK)"
                        },
                        "period": {"type": "string", "default": "1y"}
                    },
                    "required": []
                }
            }
        ]

    async def handle_tool_call(self, tool_name: str, arguments: Dict) -> Dict:
        """Handle incoming tool call."""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            # Call the appropriate handler
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.tools[tool_name](**arguments)
            )
            return result
        except Exception as e:
            self.logger.error(f"Error in {tool_name}: {e}")
            return {"error": str(e)}

    def get_ticker_price(self, ticker: str) -> Dict:
        """Get current price for a ticker."""
        return self.pipeline.get_live_quote(ticker)

    def get_historical_data(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict:
        """Get historical OHLCV data."""
        df = self.pipeline.get_historical_data(ticker, period=period, interval=interval)

        if df.empty:
            return {"error": "No data available", "ticker": ticker}

        # Convert to serializable format
        return {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "data_points": len(df),
            "start_date": df.index[0].isoformat(),
            "end_date": df.index[-1].isoformat(),
            "data": [
                {
                    "date": idx.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"])
                }
                for idx, row in df.tail(10).iterrows()  # Return last 10 for preview
            ]
        }

    def get_company_info(self, ticker: str) -> Dict:
        """Get company information."""
        return self.pipeline.get_company_info(ticker)

    def get_index_data(
        self,
        index: str = "NIFTY50",
        period: str = "1y"
    ) -> Dict:
        """Get index data."""
        df = self.pipeline.get_index_data(index, period=period)

        if df.empty:
            return {"error": "No data available", "index": index}

        return {
            "index": index,
            "period": period,
            "data_points": len(df),
            "latest_close": float(df["Close"].iloc[-1])
        }


def create_server():
    """Create and return the MCP server instance."""
    return MarketDataServer()


if __name__ == "__main__":
    # Simple test
    server = create_server()
    print("Market Data Server initialized")
    print("Available tools:", list(server.tools.keys()))

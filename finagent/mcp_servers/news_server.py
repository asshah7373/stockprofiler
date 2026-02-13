"""
MCP Server for News and Sentiment
Provides news fetching and sentiment analysis via MCP protocol.
"""

import asyncio
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class NewsServer:
    """
    MCP Server for news and sentiment operations.

    Tools provided:
    - get_ticker_sentiment: Get aggregated news sentiment for a ticker
    - get_sector_sentiment: Get sentiment for a sector
    - get_market_sentiment: Get overall market sentiment
    - get_recent_news: Get recent news headlines
    """

    def __init__(self, newsapi_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        from ..pipelines.sentiment_data import SentimentPipeline
        self.pipeline = SentimentPipeline(newsapi_key=newsapi_key)

        self.tools = {
            "get_ticker_sentiment": self.get_ticker_sentiment,
            "get_sector_sentiment": self.get_sector_sentiment,
            "get_market_sentiment": self.get_market_sentiment,
            "get_recent_news": self.get_recent_news
        }

    def get_tool_definitions(self) -> list:
        """Return tool definitions for MCP registration."""
        return [
            {
                "name": "get_ticker_sentiment",
                "description": "Get aggregated news sentiment for a stock ticker",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock symbol (e.g., RELIANCE)"
                        },
                        "days_back": {
                            "type": "integer",
                            "default": 7,
                            "description": "Number of days to analyze"
                        }
                    },
                    "required": ["ticker"]
                }
            },
            {
                "name": "get_sector_sentiment",
                "description": "Get aggregated sentiment for a sector",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sector": {
                            "type": "string",
                            "description": "Sector name (IT, BANKING, PHARMA, AUTO, FMCG, ENERGY)"
                        }
                    },
                    "required": ["sector"]
                }
            },
            {
                "name": "get_market_sentiment",
                "description": "Get overall market sentiment",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_recent_news",
                "description": "Get recent news headlines for a ticker",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "max_articles": {
                            "type": "integer",
                            "default": 10
                        }
                    },
                    "required": ["ticker"]
                }
            }
        ]

    async def handle_tool_call(self, tool_name: str, arguments: Dict) -> Dict:
        """Handle incoming tool call."""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.tools[tool_name](**arguments)
            )
            return result
        except Exception as e:
            self.logger.error(f"Error in {tool_name}: {e}")
            return {"error": str(e)}

    def get_ticker_sentiment(
        self,
        ticker: str,
        days_back: int = 7
    ) -> Dict:
        """Get sentiment summary for a ticker."""
        return self.pipeline.get_sentiment_summary(ticker, days_back=days_back)

    def get_sector_sentiment(self, sector: str) -> Dict:
        """Get sentiment for a sector."""
        return self.pipeline.get_sector_sentiment(sector)

    def get_market_sentiment(self) -> Dict:
        """Get overall market sentiment."""
        return self.pipeline.get_market_sentiment()

    def get_recent_news(
        self,
        ticker: str,
        max_articles: int = 10
    ) -> Dict:
        """Get recent news articles."""
        articles = self.pipeline.get_ticker_news(
            ticker,
            days_back=7,
            max_articles=max_articles
        )

        return {
            "ticker": ticker,
            "article_count": len(articles),
            "articles": [
                {
                    "headline": a.get("headline"),
                    "source": a.get("source"),
                    "sentiment": a.get("sentiment_score"),
                    "date": a.get("published_at")
                }
                for a in articles
            ]
        }


def create_server(newsapi_key: Optional[str] = None):
    """Create and return the MCP server instance."""
    return NewsServer(newsapi_key=newsapi_key)


if __name__ == "__main__":
    server = create_server()
    print("News Server initialized")
    print("Available tools:", list(server.tools.keys()))

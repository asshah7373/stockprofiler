"""
MCP Server for Document Parsing
Provides PDF parsing and circular fetching via MCP protocol.
"""

import asyncio
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DocumentParserServer:
    """
    MCP Server for document parsing operations.

    Tools provided:
    - fetch_circular: Download and parse a BSE/NSE circular
    - search_circulars: Search indexed circulars
    - get_circular_content: Get parsed content of a circular
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        from ..pipelines.unstructured_data import CircularPipeline
        self.pipeline = CircularPipeline()

        self.tools = {
            "fetch_circular": self.fetch_circular,
            "search_circulars": self.search_circulars,
            "get_recent_circulars": self.get_recent_circulars,
            "get_circular_content": self.get_circular_content
        }

    def get_tool_definitions(self) -> list:
        """Return tool definitions for MCP registration."""
        return [
            {
                "name": "fetch_circular",
                "description": "Download and parse a BSE/NSE circular from URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of the circular/PDF"
                        },
                        "doc_type": {
                            "type": "string",
                            "enum": ["circular", "quarterly", "announcement"],
                            "default": "circular"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "search_circulars",
                "description": "Search indexed circulars by keyword or ticker",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "ticker": {
                            "type": "string",
                            "description": "Filter by ticker symbol"
                        },
                        "days_back": {
                            "type": "integer",
                            "default": 30,
                            "description": "How many days to search"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_recent_circulars",
                "description": "Get list of recent circulars from exchange",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "exchange": {
                            "type": "string",
                            "enum": ["NSE", "BSE"],
                            "default": "NSE"
                        },
                        "ticker": {"type": "string"},
                        "days_back": {"type": "integer", "default": 7}
                    },
                    "required": []
                }
            },
            {
                "name": "get_circular_content",
                "description": "Get parsed content of a specific circular",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "Document ID"
                        }
                    },
                    "required": ["doc_id"]
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

    def fetch_circular(
        self,
        url: str,
        doc_type: str = "circular"
    ) -> Dict:
        """Download and parse a circular."""
        import hashlib
        from datetime import datetime

        # Generate doc_id from URL
        doc_id = hashlib.md5(url.encode()).hexdigest()[:16]

        circular = {
            "id": doc_id,
            "url": url,
            "doc_type": doc_type,
            "filing_date": datetime.now().isoformat()
        }

        result = self.pipeline.process_circular(circular)

        return {
            "doc_id": doc_id,
            "success": result.get("parsed_content") is not None,
            "text_length": len(result.get("parsed_content", {}).get("text", "")),
            "tables_found": len(result.get("parsed_content", {}).get("tables", [])),
            "chunks_created": len(result.get("chunks", []))
        }

    def search_circulars(
        self,
        query: Optional[str] = None,
        ticker: Optional[str] = None,
        days_back: int = 30
    ) -> Dict:
        """Search stored circulars."""
        results = self.pipeline.search_circulars(
            query=query,
            ticker=ticker,
            days_back=days_back
        )

        return {
            "query": query,
            "ticker": ticker,
            "results_count": len(results),
            "results": [
                {
                    "id": r.get("id"),
                    "title": r.get("title"),
                    "ticker": r.get("ticker"),
                    "filing_date": r.get("filing_date"),
                    "doc_type": r.get("doc_type")
                }
                for r in results[:20]  # Limit results
            ]
        }

    def get_recent_circulars(
        self,
        exchange: str = "NSE",
        ticker: Optional[str] = None,
        days_back: int = 7
    ) -> Dict:
        """Get recent circulars from exchange."""
        circulars = self.pipeline.fetch_recent_circulars(
            exchange=exchange,
            ticker=ticker,
            days_back=days_back
        )

        return {
            "exchange": exchange,
            "ticker": ticker,
            "days_back": days_back,
            "count": len(circulars),
            "circulars": circulars[:20]
        }

    def get_circular_content(self, doc_id: str) -> Dict:
        """Get parsed content of a circular."""
        circular = self.pipeline.get_circular_by_id(doc_id)

        if not circular:
            return {"error": "Circular not found", "doc_id": doc_id}

        return {
            "doc_id": doc_id,
            "title": circular.get("title"),
            "ticker": circular.get("ticker"),
            "filing_date": circular.get("filing_date"),
            "text": circular.get("parsed_text", "")[:5000],  # Limit text
            "has_tables": circular.get("tables_json") is not None
        }


def create_server():
    """Create and return the MCP server instance."""
    return DocumentParserServer()


if __name__ == "__main__":
    server = create_server()
    print("Document Parser Server initialized")
    print("Available tools:", list(server.tools.keys()))

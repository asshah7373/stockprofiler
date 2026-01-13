"""
RAG Retriever
Retrieves relevant context for analysis.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging

from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieves relevant document chunks for analysis.
    Enforces citation requirements.
    """

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.logger = logging.getLogger(__name__)

    def retrieve_for_analysis(
        self,
        ticker: str,
        query: str,
        doc_types: List[str] = None,
        lookback_days: int = 90,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for stock analysis.

        Args:
            ticker: Stock ticker symbol
            query: Analysis query or topic
            doc_types: Document types to search (default: quarterly_results, circulars)
            lookback_days: How far back to search
            top_k: Number of results per collection

        Returns:
            {
                "chunks": [...],
                "sources": [{"doc_id": ..., "title": ..., "date": ...}],
                "retrieval_metadata": {...}
            }
        """
        if doc_types is None:
            doc_types = ["quarterly_results", "circulars"]

        # Clean ticker
        clean_ticker = ticker.replace('.NS', '').replace('.BO', '').upper()

        # Build filters
        date_cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        filters = {
            'ticker': clean_ticker,
            'date_after': date_cutoff
        }

        # Search each collection
        all_chunks = []
        sources = {}

        for doc_type in doc_types:
            if doc_type in self.store.collections:
                results = self.store.search(
                    collection=doc_type,
                    query=query,
                    filters=filters,
                    top_k=top_k
                )

                for chunk in results:
                    chunk['collection'] = doc_type
                    all_chunks.append(chunk)

                    # Track unique sources
                    doc_id = chunk.get('doc_id', '')
                    if doc_id and doc_id not in sources:
                        sources[doc_id] = {
                            'doc_id': doc_id,
                            'title': chunk.get('metadata', {}).get('title', ''),
                            'date': chunk.get('metadata', {}).get('filing_date', ''),
                            'type': doc_type
                        }

        # Sort by relevance
        all_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        # Limit to top_k overall
        all_chunks = all_chunks[:top_k]

        return {
            "chunks": all_chunks,
            "sources": list(sources.values()),
            "retrieval_metadata": {
                "ticker": clean_ticker,
                "query": query,
                "doc_types_searched": doc_types,
                "lookback_days": lookback_days,
                "total_chunks_retrieved": len(all_chunks),
                "unique_sources": len(sources),
                "retrieved_at": datetime.now().isoformat()
            }
        }

    def retrieve_financial_data(
        self,
        ticker: str,
        metrics: List[str] = None,
        periods: int = 4
    ) -> Dict[str, Any]:
        """
        Retrieve financial metrics from quarterly results.

        Args:
            ticker: Stock ticker
            metrics: Specific metrics to look for (e.g., ["revenue", "profit", "eps"])
            periods: Number of quarters to retrieve

        Returns:
            Retrieved financial context
        """
        if metrics is None:
            metrics = ["revenue", "net profit", "eps", "ebitda", "margin"]

        # Build query from metrics
        query = f"financial results {' '.join(metrics)}"

        return self.retrieve_for_analysis(
            ticker=ticker,
            query=query,
            doc_types=["quarterly_results", "annual_reports"],
            lookback_days=365,  # Last year for quarterly data
            top_k=periods * 3  # Multiple chunks per period
        )

    def retrieve_corporate_actions(
        self,
        ticker: str,
        action_types: List[str] = None,
        lookback_days: int = 180
    ) -> Dict[str, Any]:
        """
        Retrieve corporate action announcements.

        Args:
            ticker: Stock ticker
            action_types: Types like ["dividend", "split", "bonus", "buyback"]
            lookback_days: How far back to search

        Returns:
            Retrieved corporate action context
        """
        if action_types is None:
            action_types = ["dividend", "split", "bonus", "rights", "buyback", "merger"]

        query = f"corporate action {' '.join(action_types)}"

        return self.retrieve_for_analysis(
            ticker=ticker,
            query=query,
            doc_types=["circulars"],
            lookback_days=lookback_days,
            top_k=10
        )

    def retrieve_management_commentary(
        self,
        ticker: str,
        topics: List[str] = None,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Retrieve management discussion and analysis.

        Args:
            ticker: Stock ticker
            topics: Topics to look for (e.g., ["outlook", "guidance", "growth"])
            lookback_days: How far back to search

        Returns:
            Retrieved management commentary
        """
        if topics is None:
            topics = ["outlook", "guidance", "growth", "strategy", "expansion"]

        query = f"management discussion {' '.join(topics)}"

        return self.retrieve_for_analysis(
            ticker=ticker,
            query=query,
            doc_types=["quarterly_results", "annual_reports"],
            lookback_days=lookback_days,
            top_k=8
        )

    def format_context_for_llm(
        self,
        retrieval_result: Dict,
        max_tokens: int = 4000
    ) -> str:
        """
        Format retrieved chunks for LLM consumption.
        Includes source IDs for citation.

        Args:
            retrieval_result: Result from retrieve_for_analysis
            max_tokens: Approximate maximum tokens (chars / 4)

        Returns:
            Formatted context string with source references
        """
        chunks = retrieval_result.get('chunks', [])
        sources = {s['doc_id']: s for s in retrieval_result.get('sources', [])}

        if not chunks:
            return "No relevant documents found."

        # Build context with citations
        context_parts = []
        current_length = 0
        max_chars = max_tokens * 4  # Rough approximation

        for chunk in chunks:
            doc_id = chunk.get('doc_id', 'unknown')
            source_info = sources.get(doc_id, {})
            collection = chunk.get('collection', '')

            # Format source reference
            source_ref = f"[Source: {doc_id}"
            if source_info.get('title'):
                source_ref += f" - {source_info['title']}"
            if source_info.get('date'):
                source_ref += f" ({source_info['date'][:10]})"
            source_ref += "]"

            # Format chunk
            chunk_text = chunk.get('chunk_text', '')
            relevance = chunk.get('relevance_score', 0)

            formatted = f"""
---
{source_ref}
Collection: {collection}
Relevance: {relevance:.2f}

{chunk_text}
---
"""
            # Check length
            if current_length + len(formatted) > max_chars:
                break

            context_parts.append(formatted)
            current_length += len(formatted)

        # Add header
        header = f"""
=== RETRIEVED CONTEXT ===
Ticker: {retrieval_result.get('retrieval_metadata', {}).get('ticker', 'N/A')}
Query: {retrieval_result.get('retrieval_metadata', {}).get('query', 'N/A')}
Documents Found: {len(retrieval_result.get('sources', []))}
Chunks Retrieved: {len(chunks)}

When citing information, use the Source ID in square brackets.
Example: "Revenue grew 15% [Source: abc123]"
"""

        return header + "\n".join(context_parts)

    def get_source_details(
        self,
        doc_id: str,
        collection: str = "circulars"
    ) -> Optional[Dict]:
        """
        Get detailed information about a source document.

        Args:
            doc_id: Document ID
            collection: Collection to search

        Returns:
            Document details including full text if available
        """
        chunks = self.store.get_document_chunks(collection, doc_id)

        if not chunks:
            return None

        # Combine chunks to get full document
        full_text = "\n".join([c.get('text', '') for c in chunks])

        return {
            'doc_id': doc_id,
            'collection': collection,
            'chunk_count': len(chunks),
            'full_text': full_text,
            'metadata': chunks[0].get('metadata', {}) if chunks else {}
        }

    def verify_citation(
        self,
        claim: str,
        doc_id: str,
        collection: str = "circulars"
    ) -> Dict[str, Any]:
        """
        Verify if a claim exists in a cited source.

        Args:
            claim: The claim to verify
            doc_id: Source document ID
            collection: Collection containing the document

        Returns:
            {
                "verified": bool,
                "supporting_text": str or None,
                "confidence": float
            }
        """
        # Get document chunks
        chunks = self.store.get_document_chunks(collection, doc_id)

        if not chunks:
            return {
                "verified": False,
                "supporting_text": None,
                "confidence": 0.0,
                "reason": "Source document not found"
            }

        # Simple keyword-based verification
        claim_lower = claim.lower()
        claim_words = set(claim_lower.split())

        best_match = None
        best_score = 0

        for chunk in chunks:
            text = chunk.get('text', '').lower()

            # Count matching words
            text_words = set(text.split())
            overlap = claim_words.intersection(text_words)
            score = len(overlap) / len(claim_words) if claim_words else 0

            if score > best_score:
                best_score = score
                best_match = chunk.get('text', '')

        # Determine verification result
        if best_score > 0.6:
            return {
                "verified": True,
                "supporting_text": best_match[:500] if best_match else None,
                "confidence": best_score
            }
        elif best_score > 0.3:
            return {
                "verified": False,
                "supporting_text": best_match[:500] if best_match else None,
                "confidence": best_score,
                "reason": "Partial match found, claim may need verification"
            }
        else:
            return {
                "verified": False,
                "supporting_text": None,
                "confidence": best_score,
                "reason": "Claim not found in source document"
            }

    def search_across_tickers(
        self,
        query: str,
        tickers: List[str],
        doc_types: List[str] = None,
        top_k_per_ticker: int = 3
    ) -> Dict[str, List[Dict]]:
        """
        Search for a query across multiple tickers.

        Useful for sector-wide analysis.

        Returns:
            {ticker: [results]}
        """
        results = {}

        for ticker in tickers:
            retrieval = self.retrieve_for_analysis(
                ticker=ticker,
                query=query,
                doc_types=doc_types,
                top_k=top_k_per_ticker
            )
            results[ticker] = retrieval.get('chunks', [])

        return results

    def get_retrieval_summary(self, retrieval_result: Dict) -> str:
        """
        Get a brief summary of what was retrieved.

        Useful for logging and transparency.
        """
        metadata = retrieval_result.get('retrieval_metadata', {})
        sources = retrieval_result.get('sources', [])

        summary_lines = [
            f"Retrieval Summary for {metadata.get('ticker', 'N/A')}:",
            f"  Query: {metadata.get('query', 'N/A')}",
            f"  Documents searched: {', '.join(metadata.get('doc_types_searched', []))}",
            f"  Lookback period: {metadata.get('lookback_days', 0)} days",
            f"  Chunks retrieved: {metadata.get('total_chunks_retrieved', 0)}",
            f"  Unique sources: {metadata.get('unique_sources', 0)}",
            "",
            "Sources:"
        ]

        for source in sources[:5]:  # Show top 5
            summary_lines.append(
                f"  - {source.get('doc_id', 'N/A')}: {source.get('title', 'Untitled')} ({source.get('date', 'N/A')[:10] if source.get('date') else 'N/A'})"
            )

        if len(sources) > 5:
            summary_lines.append(f"  ... and {len(sources) - 5} more")

        return "\n".join(summary_lines)

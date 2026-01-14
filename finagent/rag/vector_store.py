"""
Vector Store for RAG
Stores and retrieves document embeddings.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for financial documents.

    Collections:
    - circulars: Regulatory filings and announcements
    - quarterly_results: Company financial results
    - news: Recent news articles
    """

    COLLECTION_CONFIGS = {
        "circulars": {
            "description": "BSE/NSE regulatory circulars and corporate announcements",
            "metadata": {"type": "regulatory"}
        },
        "quarterly_results": {
            "description": "Company quarterly and annual financial results",
            "metadata": {"type": "financial"}
        },
        "news": {
            "description": "News articles and market commentary",
            "metadata": {"type": "news"}
        },
        "annual_reports": {
            "description": "Company annual reports and investor presentations",
            "metadata": {"type": "report"}
        }
    }

    def __init__(self, persist_dir: str = "data/vectors"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        self._init_collections()

    def _init_collections(self):
        """Initialize document collections."""
        self.collections = {}

        for name, config in self.COLLECTION_CONFIGS.items():
            try:
                self.collections[name] = self.client.get_or_create_collection(
                    name=name,
                    metadata=config["metadata"]
                )
                self.logger.info(f"Initialized collection: {name}")
            except Exception as e:
                self.logger.error(f"Error creating collection {name}: {e}")

    def get_collection(self, name: str):
        """Get a collection by name."""
        if name not in self.collections:
            raise ValueError(f"Unknown collection: {name}")
        return self.collections[name]

    def add_document(
        self,
        collection: str,
        doc_id: str,
        chunks: List[Dict],
        metadata: Dict
    ) -> int:
        """
        Add document chunks to collection.

        Args:
            collection: Collection name
            doc_id: Unique document identifier
            chunks: List of {text, metadata, chunk_index}
            metadata: Document-level metadata

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        coll = self.get_collection(collection)

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = f"{doc_id}_chunk_{chunk.get('chunk_index', 0)}"
            ids.append(chunk_id)
            documents.append(chunk.get('text', ''))

            # Combine document and chunk metadata
            chunk_metadata = {
                **metadata,
                **chunk.get('metadata', {}),
                'doc_id': doc_id,
                'chunk_index': chunk.get('chunk_index', 0),
                'added_at': datetime.now().isoformat()
            }

            # Ensure all values are serializable
            chunk_metadata = self._sanitize_metadata(chunk_metadata)
            metadatas.append(chunk_metadata)

        try:
            # Add in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                coll.add(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )

            self.logger.info(f"Added {len(chunks)} chunks for {doc_id} to {collection}")
            return len(chunks)

        except Exception as e:
            self.logger.error(f"Error adding document {doc_id}: {e}")
            return 0

    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Ensure metadata values are ChromaDB-compatible."""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, dict)):
                sanitized[key] = json.dumps(value)
            elif value is None:
                sanitized[key] = ""
            else:
                sanitized[key] = str(value)
        return sanitized

    def search(
        self,
        collection: str,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Hybrid search (semantic + metadata filtering).

        Args:
            collection: Collection name
            query: Search query
            filters: Metadata filters {ticker, date_range, doc_type}
            top_k: Number of results

        Returns:
            List of {chunk_text, doc_id, page, relevance_score, metadata}
        """
        coll = self.get_collection(collection)

        # Build where clause from filters
        where = None
        if filters:
            where = self._build_where_clause(filters)

        try:
            results = coll.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted = []
            if results and results.get('documents'):
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                    distance = results['distances'][0][i] if results.get('distances') else 0

                    formatted.append({
                        'chunk_text': doc,
                        'doc_id': metadata.get('doc_id', ''),
                        'chunk_index': metadata.get('chunk_index', 0),
                        'relevance_score': 1 - distance,  # Convert distance to similarity
                        'metadata': metadata
                    })

            return formatted

        except Exception as e:
            self.logger.error(f"Search error in {collection}: {e}")
            return []

    def _build_where_clause(self, filters: Dict) -> Optional[Dict]:
        """Build ChromaDB where clause from filters.

        Note: ChromaDB only supports numeric comparisons with $gte/$lte.
        Date filtering is done post-query in Python instead.
        """
        conditions = []

        if 'ticker' in filters and filters['ticker']:
            conditions.append({'ticker': {'$eq': filters['ticker']}})

        if 'doc_type' in filters and filters['doc_type']:
            conditions.append({'doc_type': {'$eq': filters['doc_type']}})

        # Note: ChromaDB doesn't support string comparisons with $gte/$lte
        # Date filtering removed - should be done post-query if needed

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {'$and': conditions}

    def search_multiple_collections(
        self,
        query: str,
        collections: List[str],
        filters: Optional[Dict] = None,
        top_k_per_collection: int = 3
    ) -> Dict[str, List[Dict]]:
        """
        Search across multiple collections.

        Returns:
            {collection_name: [results]}
        """
        results = {}

        for coll_name in collections:
            if coll_name in self.collections:
                results[coll_name] = self.search(
                    coll_name,
                    query,
                    filters,
                    top_k_per_collection
                )

        return results

    def delete_document(self, collection: str, doc_id: str) -> bool:
        """Delete all chunks of a document."""
        try:
            coll = self.get_collection(collection)

            # Find all chunks with this doc_id
            results = coll.get(
                where={'doc_id': {'$eq': doc_id}},
                include=['metadatas']
            )

            if results and results.get('ids'):
                coll.delete(ids=results['ids'])
                self.logger.info(f"Deleted {len(results['ids'])} chunks for {doc_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def get_document_chunks(
        self,
        collection: str,
        doc_id: str
    ) -> List[Dict]:
        """Get all chunks for a document."""
        try:
            coll = self.get_collection(collection)

            results = coll.get(
                where={'doc_id': {'$eq': doc_id}},
                include=['documents', 'metadatas']
            )

            chunks = []
            if results and results.get('documents'):
                for i, doc in enumerate(results['documents']):
                    chunks.append({
                        'text': doc,
                        'metadata': results['metadatas'][i] if results.get('metadatas') else {}
                    })

            # Sort by chunk index
            chunks.sort(key=lambda x: x.get('metadata', {}).get('chunk_index', 0))

            return chunks

        except Exception as e:
            self.logger.error(f"Error getting document {doc_id}: {e}")
            return []

    def get_collection_stats(self, collection: str) -> Dict:
        """Get statistics for a collection."""
        try:
            coll = self.get_collection(collection)
            count = coll.count()

            return {
                'collection': collection,
                'document_count': count,
                'description': self.COLLECTION_CONFIGS.get(collection, {}).get('description', '')
            }

        except Exception as e:
            self.logger.error(f"Error getting stats for {collection}: {e}")
            return {'collection': collection, 'error': str(e)}

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all collections."""
        return {
            name: self.get_collection_stats(name)
            for name in self.collections
        }

    def list_documents(
        self,
        collection: str,
        limit: int = 100
    ) -> List[str]:
        """List unique document IDs in a collection."""
        try:
            coll = self.get_collection(collection)

            results = coll.get(
                limit=limit * 10,  # Get more to account for chunks
                include=['metadatas']
            )

            doc_ids = set()
            if results and results.get('metadatas'):
                for metadata in results['metadatas']:
                    if 'doc_id' in metadata:
                        doc_ids.add(metadata['doc_id'])

            return list(doc_ids)[:limit]

        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            return []

    def clear_collection(self, collection: str) -> bool:
        """Clear all documents from a collection."""
        try:
            coll = self.get_collection(collection)

            # Get all IDs
            results = coll.get(include=[])

            if results and results.get('ids'):
                coll.delete(ids=results['ids'])
                self.logger.info(f"Cleared {len(results['ids'])} items from {collection}")

            return True

        except Exception as e:
            self.logger.error(f"Error clearing {collection}: {e}")
            return False

    def reset(self):
        """Reset all collections (dangerous!)."""
        try:
            self.client.reset()
            self._init_collections()
            self.logger.warning("All vector store data has been reset")
        except Exception as e:
            self.logger.error(f"Error resetting vector store: {e}")

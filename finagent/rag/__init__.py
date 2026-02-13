"""
FinAgent RAG Module

This module contains the Retrieval-Augmented Generation system:
- VectorStore: ChromaDB-based document storage
- EmbeddingGenerator: Document embedding logic
- RAGRetriever: Context retrieval for analysis
"""

from .vector_store import VectorStore
from .embeddings import EmbeddingGenerator
from .retriever import RAGRetriever

__all__ = ["VectorStore", "EmbeddingGenerator", "RAGRetriever"]

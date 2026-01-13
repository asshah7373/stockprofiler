"""
Document Embedding Logic
Generates embeddings for documents and queries.
"""

from typing import List, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text using sentence-transformers.

    Uses the 'all-MiniLM-L6-v2' model by default, which provides
    a good balance of speed and quality for document retrieval.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.logger = logging.getLogger(__name__)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir) if self.cache_dir else None
            )
            self.logger.info(f"Loaded embedding model: {self.model_name}")

        except ImportError:
            self.logger.warning(
                "sentence-transformers not installed. "
                "Embeddings will not be available."
            )
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")

    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            List of floats representing the embedding, or None if failed
        """
        if not self.model:
            return None

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

        except Exception as e:
            self.logger.error(f"Error embedding text: {e}")
            return None

    def embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embeddings, or None if failed
        """
        if not self.model:
            return None

        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10
            )
            return embeddings.tolist()

        except Exception as e:
            self.logger.error(f"Error embedding texts: {e}")
            return None

    def embed_query(self, query: str) -> Optional[List[float]]:
        """
        Generate embedding for a search query.

        This is an alias for embed_text but kept separate
        in case query-specific preprocessing is needed.
        """
        return self.embed_text(query)

    def embed_document(
        self,
        title: str,
        content: str,
        prepend_title: bool = True
    ) -> Optional[List[float]]:
        """
        Generate embedding for a document.

        Optionally prepends the title to improve retrieval quality.

        Args:
            title: Document title
            content: Document content
            prepend_title: Whether to prepend title

        Returns:
            Document embedding
        """
        if prepend_title and title:
            text = f"{title}\n\n{content}"
        else:
            text = content

        return self.embed_text(text)

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Returns:
            Similarity score between 0 and 1
        """
        try:
            import numpy as np

            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.0

    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find the most similar embeddings to a query.

        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results

        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            import numpy as np

            query = np.array(query_embedding)
            candidates = np.array(candidate_embeddings)

            # Compute similarities
            similarities = np.dot(candidates, query) / (
                np.linalg.norm(candidates, axis=1) * np.linalg.norm(query)
            )

            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            return [
                (int(idx), float(similarities[idx]))
                for idx in top_indices
            ]

        except Exception as e:
            self.logger.error(f"Error finding similar: {e}")
            return []

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if not self.model:
            return 0

        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception:
            return 384  # Default for MiniLM

    def batch_embed_with_progress(
        self,
        texts: List[str],
        batch_size: int = 32,
        callback: Optional[callable] = None
    ) -> Optional[List[List[float]]]:
        """
        Embed texts in batches with progress callback.

        Args:
            texts: List of texts to embed
            batch_size: Batch size
            callback: Function called with (processed, total) after each batch

        Returns:
            List of embeddings
        """
        if not self.model:
            return None

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_texts(batch)

            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)

            if callback:
                callback(min(i + batch_size, len(texts)), len(texts))

        return all_embeddings if all_embeddings else None


class CachedEmbeddingGenerator(EmbeddingGenerator):
    """
    Embedding generator with caching to avoid recomputing embeddings.
    """

    def __init__(
        self,
        model_name: str = EmbeddingGenerator.DEFAULT_MODEL,
        cache_dir: Optional[str] = None,
        embedding_cache_path: Optional[str] = None
    ):
        super().__init__(model_name, cache_dir)

        self.embedding_cache_path = Path(embedding_cache_path) if embedding_cache_path else None
        self.embedding_cache = {}

        if self.embedding_cache_path and self.embedding_cache_path.exists():
            self._load_cache()

    def _load_cache(self):
        """Load embedding cache from disk."""
        try:
            import json
            with open(self.embedding_cache_path, 'r') as f:
                self.embedding_cache = json.load(f)
            self.logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            self.logger.warning(f"Could not load embedding cache: {e}")

    def _save_cache(self):
        """Save embedding cache to disk."""
        if not self.embedding_cache_path:
            return

        try:
            import json
            self.embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.embedding_cache_path, 'w') as f:
                json.dump(self.embedding_cache, f)
        except Exception as e:
            self.logger.warning(f"Could not save embedding cache: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Embed with caching."""
        cache_key = self._get_cache_key(text)

        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        embedding = super().embed_text(text)

        if embedding:
            self.embedding_cache[cache_key] = embedding

        return embedding

    def save(self):
        """Persist the cache to disk."""
        self._save_cache()

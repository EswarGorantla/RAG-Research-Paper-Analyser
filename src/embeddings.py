import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer

# Default path to the locally saved all-MiniLM-L6-v2 model
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "all-MiniLM-L6-v2"
)


class EmbeddingModel:
    """
    SentenceTransformer embedding wrapper.
    Standardized with 'embed_documents' and 'embed_query' methods to ensure
    seamless compatibility with LangChain and vector databases like ChromaDB.
    Includes LRU caching for repeated query vectorization.
    """

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL_PATH):
        if os.path.exists(model_name_or_path):
            self.model = SentenceTransformer(model_name_or_path)
        else:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Cache internal query encoder
        self._cached_encode = lru_cache(maxsize=256)(self._raw_encode)

    def _raw_encode(self, query: str) -> tuple[float, ...]:
        vector = self.model.encode(
            query, show_progress_bar=False, normalize_embeddings=True
        )
        return tuple(float(x) for x in vector)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embeds a list of text strings into normalized dense vectors in optimized batches.

        Args:
            texts (list[str]): List of texts to encode.
            batch_size (int): Batch size for GPU/CPU inference.

        Returns:
            list[list[float]]: List of 384-dimensional embedding vectors.
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embeds a single query string into a normalized dense vector with LRU caching.

        Args:
            query (str): Search query string.

        Returns:
            list[float]: 384-dimensional embedding vector.
        """
        cleaned_query = query.strip().lower()
        cached_tuple = self._cached_encode(cleaned_query)
        return list(cached_tuple)


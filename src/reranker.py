import os
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "ms-marco-MiniLM-L-6-v2"
)


class Reranker:
    """
    Precision Reranker powered by a Cross-Encoder transformer.
    Jointly evaluates (Query, Document) pairs with full cross-attention.
    Includes in-memory LRU prediction caching and optimized batch inference.
    """

    def __init__(self, model_name_or_path: str = DEFAULT_RERANKER_PATH):
        if os.path.exists(model_name_or_path):
            self.model = CrossEncoder(model_name_or_path)
        else:
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # In-memory cache for query-passage scores to prevent redundant forward passes
        self._cache: dict[tuple[str, str], float] = {}

    def predict_pairs(self, pairs: list[list[str]], batch_size: int = 32) -> list[float]:
        """
        Computes relevance scores for a list of [query, passage] pairs with caching.
        """
        if not pairs:
            return []

        needed_pairs = []
        needed_indices = []
        results = [0.0] * len(pairs)

        for idx, (query, text) in enumerate(pairs):
            cache_key = (query.strip().lower(), text.strip()[:150])
            if cache_key in self._cache:
                results[idx] = self._cache[cache_key]
            else:
                needed_pairs.append([query, text])
                needed_indices.append((idx, cache_key))

        if needed_pairs:
            # Batch forward pass with PyTorch
            computed_scores = self.model.predict(
                needed_pairs,
                batch_size=batch_size,
                show_progress_bar=False
            )
            for (idx, cache_key), score in zip(needed_indices, computed_scores):
                float_score = round(float(score), 4)
                self._cache[cache_key] = float_score
                results[idx] = float_score

        return results

    def rerank(self, query: str, candidate_chunks: list[dict], top_k: int = 3) -> list[dict]:
        """
        Reranks a list of retrieved candidate chunks for a query.

        Args:
            query (str): The user search question.
            candidate_chunks (list[dict]): Chunks from Hybrid Retriever.
            top_k (int): Number of top reranked chunks to return.

        Returns:
            list[dict]: Chunks reordered by Cross-Encoder relevance score.
        """
        if not candidate_chunks:
            return []

        # Prepare (Query, Passage) pairs for full cross-attention
        pairs = [[query, chunk["text"]] for chunk in candidate_chunks]

        # Compute cross-encoder relevance scores using batched caching
        scores = self.predict_pairs(pairs, batch_size=32)

        # Attach scores to the candidate dicts
        reranked = []
        for chunk, score in zip(candidate_chunks, scores):
            updated_chunk = chunk.copy()
            updated_chunk["rerank_score"] = score
            reranked.append(updated_chunk)

        # Sort descending by cross-encoder score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]

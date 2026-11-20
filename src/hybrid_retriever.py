from src.vector_store import VectorStoreManager
from src.bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Combines Dense Semantic Search (ChromaDB) and Sparse Keyword Search (BM25)
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, vector_store: VectorStoreManager, bm25_retriever: BM25Retriever):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever

    def search(
        self,
        query: str,
        k: int = 4,
        semantic_candidates: int = 10,
        bm25_candidates: int = 10,
        rrf_constant: int = 60,
    ) -> list[dict]:
        """
        Executes hybrid retrieval by fusing rankings from dense and sparse search.

        RRF Formula:
            Score(doc) = 1 / (rrf_constant + rank_semantic) + 1 / (rrf_constant + rank_bm25)

        Args:
            query (str): User question or search query.
            k (int): Number of final hybrid results to return.
            semantic_candidates (int): Candidate pool size from ChromaDB.
            bm25_candidates (int): Candidate pool size from BM25.
            rrf_constant (int): Smoothing factor (default 60, standard in IR literature).

        Returns:
            list[dict]: RRF-ranked results with combined scores and citations.
        """
        # 1. Fetch dense candidates from ChromaDB
        dense_results = self.vector_store.similarity_search(query, k=semantic_candidates)

        # 2. Fetch sparse candidates from BM25
        sparse_results = self.bm25.search(query, k=bm25_candidates)

        # 3. Apply Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[str, float] = {}
        doc_store: dict[str, dict] = {}
        dense_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}

        # Process dense ranks
        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            doc_store[cid] = item
            dense_ranks[cid] = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_constant + rank))

        # Process sparse ranks
        for rank, item in enumerate(sparse_results, start=1):
            cid = item["chunk_id"]
            if cid not in doc_store:
                doc_store[cid] = item
            bm25_ranks[cid] = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_constant + rank))

        # 4. Sort documents by final fused RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        final_results = []
        for cid in sorted_cids[:k]:
            item = doc_store[cid].copy()
            item["rrf_score"] = round(rrf_scores[cid], 5)
            item["dense_rank"] = dense_ranks.get(cid, None)
            item["bm25_rank"] = bm25_ranks.get(cid, None)
            final_results.append(item)

        return final_results

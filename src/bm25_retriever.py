import re
import heapq
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


def tokenize(text: str) -> list[str]:
    """
    Fast alphanumeric tokenizer for BM25.
    Converts to lowercase and extracts contiguous word tokens.
    """
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """
    Optimized keyword-based BM25 retriever for exact term, acronym, and numerical matching.
    Uses heapq.nlargest for O(N log k) top-k selection instead of full list sorting.
    """

    def __init__(self, documents: list[Document]):
        self.documents = documents
        # Tokenize each document's content
        self.corpus = [tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Retrieves the top-k documents according to BM25 keyword matching score.

        Args:
            query (str): The search query text.
            k (int): Number of top results to return.

        Returns:
            list[dict]: List of matched chunks with metadata, page number, and BM25 score.
        """
        tokenized_query = tokenize(query)
        if not tokenized_query or not self.documents:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Use heapq.nlargest for fast O(N log k) selection
        top_indices = heapq.nlargest(
            min(k, len(scores)),
            range(len(scores)),
            key=lambda i: scores[i]
        )

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue
            doc = self.documents[idx]
            results.append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "text": doc.page_content,
                "metadata": doc.metadata,
                "page": doc.metadata.get("page"),
                "paper_name": doc.metadata.get("paper_name"),
                "score": round(float(score), 4),
            })

        return results

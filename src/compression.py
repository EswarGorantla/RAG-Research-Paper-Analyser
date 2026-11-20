import re
from src.reranker import Reranker


def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into cohesive sentences and table statements.
    Intelligently merges short table cells and numeric values (e.g., '48,796', '0.997 ms')
    with their preceding labels so critical numbers are never lost.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    segments = []
    current = ""

    for line in lines:
        if not current:
            current = line
        elif len(line) < 20 or len(current) < 30 or not current.endswith((".", ":", ";", "?", "!")):
            # Merge short lines, numbers, or uncompleted statements with current segment
            current += " " + line
        else:
            segments.append(current)
            current = line

    if current:
        segments.append(current)

    return segments


class ContextualCompressor:
    """
    Compresses retrieved chunks by extracting only the most query-relevant sentences.
    Optimized with global multi-chunk sentence batching to run all scoring in a single
    cross-encoder forward pass.
    """

    def __init__(self, reranker: Reranker | None = None, max_sentences_per_chunk: int = 3):
        self.reranker = reranker or Reranker()
        self.max_sentences = max_sentences_per_chunk

    def compress(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Compresses a list of candidate chunks with a single batched cross-encoder inference call.

        Args:
            query (str): The search question.
            chunks (list[dict]): Candidate chunks to compress.

        Returns:
            list[dict]: Compressed chunks with 'compressed_text' and character metrics.
        """
        if not chunks:
            return []

        # 1. Parse sentences for all chunks and collect needed pairs
        chunk_sentences = []
        all_pairs = []
        pair_to_chunk_map = []  # tracks which chunk and sentence index each pair belongs to

        for chunk_idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            sentences = split_into_sentences(text)
            chunk_sentences.append(sentences)

            # Only score sentences if chunk exceeds max_sentences
            if len(sentences) > self.max_sentences:
                for sent_idx, sent in enumerate(sentences):
                    all_pairs.append([query, sent])
                    pair_to_chunk_map.append((chunk_idx, sent_idx))

        # 2. Execute a single global batched forward pass if needed
        all_scores = {}
        if all_pairs:
            scores = self.reranker.predict_pairs(all_pairs, batch_size=32)
            for (c_idx, s_idx), score in zip(pair_to_chunk_map, scores):
                all_scores[(c_idx, s_idx)] = score

        # 3. Assemble compressed chunks
        compressed_chunks = []
        for chunk_idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            sentences = chunk_sentences[chunk_idx]

            if len(sentences) <= self.max_sentences:
                compressed_chunk = chunk.copy()
                compressed_chunk["compressed_text"] = text
                compressed_chunk["original_chars"] = len(text)
                compressed_chunk["compressed_chars"] = len(text)
                compressed_chunk["reduction_pct"] = 0.0
                compressed_chunks.append(compressed_chunk)
                continue

            # Retrieve scores for sentences in this chunk
            scored_sentences = [
                (sent_idx, sent, all_scores.get((chunk_idx, sent_idx), 0.0))
                for sent_idx, sent in enumerate(sentences)
            ]

            # Sort descending by score to pick top sentences
            scored_sentences.sort(key=lambda x: x[2], reverse=True)
            top_selected = scored_sentences[: self.max_sentences]

            # Re-sort back to original chronological order
            top_selected.sort(key=lambda x: x[0])
            compressed_text = "\n".join([item[1] for item in top_selected])

            compressed_chunk = chunk.copy()
            compressed_chunk["compressed_text"] = compressed_text
            compressed_chunk["original_chars"] = len(text)
            compressed_chunk["compressed_chars"] = len(compressed_text)
            compressed_chunk["reduction_pct"] = round(
                (1.0 - (len(compressed_text) / max(1, len(text)))) * 100, 1
            )
            compressed_chunks.append(compressed_chunk)

        return compressed_chunks

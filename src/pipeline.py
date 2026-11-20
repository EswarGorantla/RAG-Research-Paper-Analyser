import os
import time
from langchain_core.documents import Document
from src.pdf_reader import extract_text
from src.chunking import chunk_pages
from src.vector_store import VectorStoreManager
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.compression import ContextualCompressor
from src.generator import LLMGenerator


class RAGPipeline:
    """
    End-to-End Advanced RAG Pipeline Orchestrator.
    Manages document ingestion, multi-paper indexing, hybrid retrieval,
    cross-encoder reranking, contextual compression, and grounded LLM generation.
    Supports both batch and real-time token streaming.
    """

    def __init__(self, api_key: str | None = None):
        self.vector_store = VectorStoreManager()
        self.reranker = Reranker()
        self.compressor = ContextualCompressor(reranker=self.reranker, max_sentences_per_chunk=3)
        self.generator = LLMGenerator(api_key=api_key)

        # In-memory document list for BM25
        self.all_documents: list[Document] = []
        self.bm25_retriever: BM25Retriever | None = None
        self._init_bm25_from_chroma()

    def _init_bm25_from_chroma(self):
        """Reconstructs the BM25 index from persistent ChromaDB collection."""
        all_data = self.vector_store.collection.get(include=["documents", "metadatas"])
        if all_data and all_data.get("documents"):
            self.all_documents = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(all_data["documents"], all_data["metadatas"])
            ]
            self.bm25_retriever = BM25Retriever(self.all_documents)
        else:
            self.all_documents = []
            self.bm25_retriever = None

    def ingest_paper(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> dict:
        """
        Extracts, chunks, embeds, and indexes a research paper PDF.
        Skips embedding for chunks that already exist in the database.

        Args:
            pdf_path (str): File system path to the PDF.
            chunk_size (int): Max chunk size in characters.
            chunk_overlap (int): Overlap between chunks.

        Returns:
            dict: Summary containing paper name, total pages, and total chunks.
        """
        paper_name = os.path.basename(pdf_path)

        # 1. Page-preserving text extraction with normalization
        pages = extract_text(pdf_path)

        # 2. Page-level chunking
        chunks = chunk_pages(pages, paper_name=paper_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 3. Add to ChromaDB vector store
        self.vector_store.add_documents(chunks)

        # 4. Update BM25 index
        self._init_bm25_from_chroma()

        return {
            "paper_name": paper_name,
            "total_pages": len(pages),
            "total_chunks": len(chunks)
        }

    def delete_paper(self, paper_name: str) -> int:
        """
        Deletes a paper from ChromaDB and synchronizes the BM25 index.

        Args:
            paper_name (str): Name of the paper to delete.

        Returns:
            int: Number of deleted chunks.
        """
        deleted_count = self.vector_store.delete_paper(paper_name)
        self._init_bm25_from_chroma()
        return deleted_count

    def clear_database(self):
        """Wipes all documents and resets ChromaDB and BM25 indexes."""
        self.vector_store.clear_all()
        self.all_documents = []
        self.bm25_retriever = None

    def retrieve_and_compress(
        self,
        question: str,
        target_paper: str | None = None,
        compare_papers: list[str] | None = None,
        top_k: int = 3,
        candidate_k: int = 10,
    ) -> dict:
        """
        Runs Hybrid Retrieval, Cross-Encoder Reranking, and Contextual Compression.

        Returns:
            dict: Contains 'compressed_context', 'is_comparison', 'timings_ms', and 'audit_trail'.
        """
        if not self.bm25_retriever:
            self._init_bm25_from_chroma()

        if not self.all_documents:
            return {
                "compressed_context": [],
                "is_comparison": False,
                "timings_ms": {},
                "audit_trail": {},
                "error": "No research papers have been uploaded or indexed yet. Please upload a PDF first."
            }

        start_total = time.perf_counter()

        # Step 1: Hybrid Candidate Retrieval
        t0 = time.perf_counter()
        hybrid = HybridRetriever(self.vector_store, self.bm25_retriever)
        is_comparison = bool(compare_papers and len(compare_papers) >= 2)

        if is_comparison:
            # Balanced retrieval across compared papers
            candidate_pool = []
            per_paper_k = max(3, candidate_k // len(compare_papers))
            seen_ids = set()
            for p in compare_papers:
                p_filter = {"paper_name": p}
                p_dense = self.vector_store.similarity_search(question, k=per_paper_k, filter_dict=p_filter)
                p_bm25 = [d for d in self.bm25_retriever.search(question, k=per_paper_k * 3) if d.get("paper_name") == p][:per_paper_k]
                
                for c in p_dense + p_bm25:
                    cid = c.get("chunk_id")
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        candidate_pool.append(c)
        elif target_paper and target_paper != "All Papers":
            p_filter = {"paper_name": target_paper}
            p_dense = self.vector_store.similarity_search(question, k=candidate_k, filter_dict=p_filter)
            p_bm25 = [d for d in self.bm25_retriever.search(question, k=candidate_k * 3) if d.get("paper_name") == target_paper][:candidate_k]
            
            seen_ids = set()
            candidate_pool = []
            for c in p_dense + p_bm25:
                cid = c.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    candidate_pool.append(c)
        else:
            candidate_pool = hybrid.search(question, k=candidate_k)

        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

        if not candidate_pool:
            return {
                "compressed_context": [],
                "is_comparison": is_comparison,
                "timings_ms": {"retrieval": retrieval_ms},
                "audit_trail": {},
                "error": "No relevant excerpts matching your paper selection were found."
            }

        # Step 2: Cross-Encoder Precision Reranking
        t1 = time.perf_counter()
        reranked = self.reranker.rerank(question, candidate_pool, top_k=top_k)
        rerank_ms = round((time.perf_counter() - t1) * 1000, 1)

        # Step 3: Contextual Compression (batched across all sentences)
        t2 = time.perf_counter()
        compressed = self.compressor.compress(question, reranked)
        compression_ms = round((time.perf_counter() - t2) * 1000, 1)

        # Compression efficiency stats
        orig_chars = sum(c.get("original_chars", len(c.get("text", ""))) for c in compressed)
        comp_chars = sum(c.get("compressed_chars", len(c.get("compressed_text", ""))) for c in compressed)
        savings_pct = round((1.0 - (comp_chars / max(1, orig_chars))) * 100, 1)

        return {
            "compressed_context": compressed,
            "is_comparison": is_comparison,
            "timings_ms": {
                "retrieval": retrieval_ms,
                "rerank": rerank_ms,
                "compression": compression_ms,
            },
            "audit_trail": {
                "initial_candidates_count": len(candidate_pool),
                "reranked_count": len(reranked),
                "original_characters": orig_chars,
                "compressed_characters": comp_chars,
                "tokens_saved_percent": savings_pct,
                "candidates_pre_rerank": candidate_pool[:5],
                "candidates_post_rerank": reranked
            },
            "start_total": start_total
        }

    def query(
        self,
        question: str,
        target_paper: str | None = None,
        compare_papers: list[str] | None = None,
        top_k: int = 3,
        candidate_k: int = 10,
        temperature: float = 0.2,
        model_name: str | None = None,
    ) -> dict:
        """
        Executes the synchronous Advanced RAG pipeline for a user question with telemetry timing.
        """
        prep = self.retrieve_and_compress(
            question=question,
            target_paper=target_paper,
            compare_papers=compare_papers,
            top_k=top_k,
            candidate_k=candidate_k
        )

        if "error" in prep:
            return {
                "answer": prep["error"],
                "citations": [],
                "retrieved_context": [],
                "model_used": "none",
                "timings_ms": prep.get("timings_ms", {}),
                "audit_trail": prep.get("audit_trail", {})
            }

        compressed = prep["compressed_context"]
        is_comparison = prep["is_comparison"]
        timings = prep["timings_ms"]
        start_total = prep["start_total"]

        # Step 4: Grounded LLM Answer Generation
        t3 = time.perf_counter()
        result = self.generator.generate_answer(
            query=question,
            context_chunks=compressed,
            temperature=temperature,
            model_name=model_name,
            is_comparison=is_comparison,
        )
        llm_ms = round((time.perf_counter() - t3) * 1000, 1)
        total_ms = round((time.perf_counter() - start_total) * 1000, 1)

        timings["llm"] = llm_ms
        timings["total"] = total_ms

        result["retrieved_context"] = compressed
        result["timings_ms"] = timings
        result["audit_trail"] = prep["audit_trail"]

        return result

    def query_stream(
        self,
        question: str,
        target_paper: str | None = None,
        compare_papers: list[str] | None = None,
        top_k: int = 3,
        candidate_k: int = 10,
        temperature: float = 0.2,
        model_name: str | None = None,
    ):
        """
        Prepares the retrieved and compressed context, and returns a generator
        yielding LLM tokens in real time, along with context and citation metadata.
        """
        prep = self.retrieve_and_compress(
            question=question,
            target_paper=target_paper,
            compare_papers=compare_papers,
            top_k=top_k,
            candidate_k=candidate_k
        )

        if "error" in prep:
            def err_gen():
                yield prep["error"]
            return err_gen(), [], [], prep.get("timings_ms", {}), prep.get("audit_trail", {}), "none"

        compressed = prep["compressed_context"]
        is_comparison = prep["is_comparison"]

        # Extract citations
        citations = []
        seen = set()
        for c in compressed:
            ref = (c.get("paper_name"), c.get("page"))
            if ref not in seen:
                seen.add(ref)
                citations.append({
                    "paper_name": c.get("paper_name"),
                    "page": c.get("page"),
                    "chunk_id": c.get("chunk_id")
                })

        stream = self.generator.generate_answer_stream(
            query=question,
            context_chunks=compressed,
            temperature=temperature,
            model_name=model_name,
            is_comparison=is_comparison,
        )

        chosen_model = model_name or self.generator.model_name
        return stream, citations, compressed, prep["timings_ms"], prep["audit_trail"], chosen_model

    def get_indexed_papers(self) -> list[str]:
        """Returns list of currently indexed paper names."""
        return self.vector_store.get_paper_list()

    def get_stats(self) -> dict:
        """Returns detailed statistics on the indexed papers and total chunk count."""
        papers = self.get_indexed_papers()
        paper_stats = self.vector_store.get_paper_stats()
        return {
            "total_papers": len(papers),
            "paper_names": papers,
            "paper_stats": paper_stats,
            "total_chunks": self.vector_store.count()
        }

import os
from src.pdf_reader import extract_text
from src.chunking import chunk_pages
from src.vector_store import VectorStoreManager
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.compression import ContextualCompressor

# 1. Pipeline integration: PDF -> Chunks
pdf_path = os.path.join("data", "papers", "sample.pdf")
paper_name = os.path.basename(pdf_path)

print(f"[1] Extracting text and chunking {pdf_path}...")
pages = extract_text(pdf_path)
chunks = chunk_pages(pages, paper_name=paper_name, chunk_size=1000, chunk_overlap=200)

# 2. Stage 1: Hybrid Retrieval
print("[2] Initializing Hybrid Retrieval...")
vector_store = VectorStoreManager()
vector_store.add_documents(chunks)
bm25 = BM25Retriever(chunks)
hybrid_retriever = HybridRetriever(vector_store, bm25)

# 3. Stage 2: Cross-Encoder Reranking
print("[3] Initializing Cross-Encoder Reranker...")
reranker = Reranker()

# 4. Stage 3: Contextual Compression
print("[4] Initializing Contextual Compressor...")
compressor = ContextualCompressor(reranker=reranker, max_sentences_per_chunk=3)

# 5. Execute full retrieval pipeline
query = "What is the total parameter count and single CPU latency?"

print("\n" + "=" * 70)
print(f"QUERY: '{query}'")
print("=" * 70)

# Step A: Hybrid candidate retrieval
candidates = hybrid_retriever.search(query, k=6)

# Step B: Cross-Encoder precision rerank
top_chunks = reranker.rerank(query, candidates, top_k=2)

# Step C: Contextual compression
compressed_results = compressor.compress(query, top_chunks)

print("\n" + "=" * 70)
print("CONTEXTUAL COMPRESSION RESULTS (Before vs After)")
print("=" * 70)

for i, res in enumerate(compressed_results, start=1):
    print(f"\n[Result #{i}]")
    print(f"Source Paper     : {res['paper_name']}")
    print(f"Citation         : Page {res['page']} (Chunk ID: {res['chunk_id']})")
    print(f"Original Length  : {res.get('original_chars', len(res['text']))} characters")
    print(f"Compressed Length: {res.get('compressed_chars', len(res['compressed_text']))} characters ({res.get('reduction_pct', 0)}% noise eliminated)")
    print("\n--- ORIGINAL TEXT (Full Chunk) ---")
    print(res["text"][:280].strip() + "...\n")
    print("--- COMPRESSED TEXT (Pinpointed Focus for LLM) ---")
    print(res["compressed_text"])

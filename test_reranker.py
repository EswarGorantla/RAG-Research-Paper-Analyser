import os
from src.pdf_reader import extract_text
from src.chunking import chunk_pages
from src.vector_store import VectorStoreManager
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker

# 1. Pipeline integration: PDF -> Chunks
pdf_path = os.path.join("data", "papers", "sample.pdf")
paper_name = os.path.basename(pdf_path)

print(f"[1] Extracting text and chunking {pdf_path}...")
pages = extract_text(pdf_path)
chunks = chunk_pages(pages, paper_name=paper_name, chunk_size=1000, chunk_overlap=200)
print(f"[+] Total chunks: {len(chunks)}")

# 2. Stage 1: Fast Candidate Generation (Hybrid Retrieval)
print("\n[2] Setting up Hybrid Retrieval (ChromaDB + BM25)...")
vector_store = VectorStoreManager()
vector_store.add_documents(chunks)
bm25 = BM25Retriever(chunks)
hybrid_retriever = HybridRetriever(vector_store, bm25)

# 3. Stage 2: Precision Reranker (Cross-Encoder)
print("[3] Loading Cross-Encoder Reranker...")
reranker = Reranker()
print("[+] Reranker ready!")

# 4. Test query to demonstrate reranking impact
query = "What is the total parameter count and latency on Intel CPU?"

print("\n" + "=" * 70)
print(f"QUERY: '{query}'")
print("=" * 70)

# Fetch 6 candidates from Stage 1 (Hybrid)
candidates = hybrid_retriever.search(query, k=6)

print(f"\n--- STAGE 1: Top 6 Hybrid Candidates (RRF Ranked) ---")
for i, c in enumerate(candidates, start=1):
    print(f"  [{i}] ID: {c['chunk_id']} | Page: {c['page']} | RRF Score: {c['rrf_score']}")
    print(f"      Preview: {c['text'][:120].strip()}...\n")

# Apply Stage 2 (Cross-Encoder Reranker)
top_reranked = reranker.rerank(query, candidates, top_k=3)

print("=" * 70)
print("--- STAGE 2: Top 3 Reranked Results (Cross-Encoder Precision) ---")
print("=" * 70)

for i, res in enumerate(top_reranked, start=1):
    print(f"\n  Final Rank #{i} | Cross-Encoder Score: {res['rerank_score']}")
    print(f"  Source Paper   : {res['paper_name']}")
    print(f"  Citation       : Page {res['page']} (Chunk ID: {res['chunk_id']})")
    print(f"  Excerpt        :\n{res['text'][:320].strip()}...")

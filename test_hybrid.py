import os
from src.pdf_reader import extract_text
from src.chunking import chunk_pages
from src.vector_store import VectorStoreManager
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever

# 1. Load document and chunks
pdf_path = os.path.join("data", "papers", "sample.pdf")
paper_name = os.path.basename(pdf_path)

print(f"[1] Extracting text and chunking {pdf_path}...")
pages = extract_text(pdf_path)
chunks = chunk_pages(pages, paper_name=paper_name, chunk_size=1000, chunk_overlap=200)
print(f"[+] Total chunks: {len(chunks)}")

# 2. Initialize ChromaDB (Dense)
print("\n[2] Initializing ChromaDB vector store...")
vector_store = VectorStoreManager()
vector_store.add_documents(chunks)

# 3. Initialize BM25 (Sparse)
print("[3] Initializing BM25 keyword index...")
bm25_retriever = BM25Retriever(chunks)

# 4. Initialize Hybrid Retriever (Dense + Sparse with RRF)
print("[4] Initializing Hybrid Retriever with Reciprocal Rank Fusion (RRF)...")
hybrid_retriever = HybridRetriever(vector_store, bm25_retriever)

# 5. Test with specific queries demonstrating the hybrid strength
test_queries = [
    # Query A: Keyword-heavy with specific author/dataset terms
    "AdamW optimizer and ASL Alphabet Dataset 87,000 images",
    # Query B: Conceptual question where synonyms matter
    "How are the biological hand joint angles and kinematic chains modeled?"
]

print("\n" + "=" * 70)
print("TESTING HYBRID RETRIEVAL (DENSE + BM25 via RRF)")
print("=" * 70)

for query in test_queries:
    print(f"\nQuery: '{query}'")
    results = hybrid_retriever.search(query, k=2)

    for i, res in enumerate(results, start=1):
        print(f"\n  Rank #{i} | RRF Score: {res['rrf_score']}")
        print(f"  Component Ranks -> Dense Rank: {res['dense_rank']}, BM25 Rank: {res['bm25_rank']}")
        print(f"  Source Paper   : {res['paper_name']}")
        print(f"  Citation       : Page {res['page']} (Chunk ID: {res['chunk_id']})")
        print(f"  Excerpt        : {res['text'][:240].strip()}...")

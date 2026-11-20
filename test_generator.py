import os
from src.pdf_reader import extract_text
from src.chunking import chunk_pages
from src.vector_store import VectorStoreManager
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.compression import ContextualCompressor
from src.generator import LLMGenerator

# 1. Pipeline integration: PDF -> Chunks
pdf_path = os.path.join("data", "papers", "sample.pdf")
paper_name = os.path.basename(pdf_path)

print(f"[1] Extracting text and chunking {pdf_path}...")
pages = extract_text(pdf_path)
chunks = chunk_pages(pages, paper_name=paper_name, chunk_size=1000, chunk_overlap=200)

# 2. Hybrid Retrieval (ChromaDB + BM25)
print("[2] Initializing Hybrid Index...")
vector_store = VectorStoreManager()
vector_store.add_documents(chunks)
bm25 = BM25Retriever(chunks)
hybrid_retriever = HybridRetriever(vector_store, bm25)

# 3. Precision Reranker
print("[3] Initializing Cross-Encoder Reranker...")
reranker = Reranker()

# 4. Contextual Compressor
print("[4] Initializing Contextual Compressor...")
compressor = ContextualCompressor(reranker=reranker, max_sentences_per_chunk=3)

# 5. LLM Generator
print("[5] Initializing LLM Answer Generator...")
generator = LLMGenerator()

# 6. Execute full End-to-End RAG query
query = "What is the model architecture, total parameter count, and CPU latency?"

print("\n" + "=" * 70)
print(f"QUESTION: '{query}'")
print("=" * 70)

# Step A: Hybrid retrieve 6 candidates
candidates = hybrid_retriever.search(query, k=6)

# Step B: Rerank to top 2
top_ranked = reranker.rerank(query, candidates, top_k=2)

# Step C: Contextual compression
compressed = compressor.compress(query, top_ranked)

# Step D: LLM Generation
result = generator.generate_answer(query, compressed)

print("\n" + "=" * 70)
print("GENERATED ANSWER")
print("=" * 70)
print(result["answer"])

print("\n" + "=" * 70)
print("STRUCTURED CITATIONS & SOURCES")
print("=" * 70)
for cit in result["citations"]:
    print(f"• Source Paper : {cit['paper_name']}")
    print(f"  Page Cited   : Page {cit['page']}")
    print(f"  Chunk ID     : {cit['chunk_id']}\n")

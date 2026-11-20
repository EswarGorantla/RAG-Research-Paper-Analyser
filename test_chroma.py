import os
from src.pdf_reader import extract_text
from src.chunking import chunk_pages
from src.vector_store import VectorStoreManager

# 1. Pipeline integration: PDF -> Chunks
pdf_path = os.path.join("data", "papers", "sample.pdf")
paper_name = os.path.basename(pdf_path)

print(f"[1] Extracting text from {pdf_path}...")
pages = extract_text(pdf_path)

print(f"[2] Chunking pages...")
chunks = chunk_pages(pages, paper_name=paper_name, chunk_size=1000, chunk_overlap=200)
print(f"[+] Total chunks created: {len(chunks)}")

# 2. Store in ChromaDB
print(f"\n[3] Initializing ChromaDB vector store...")
vector_store = VectorStoreManager()

print(f"[4] Storing {len(chunks)} chunks into ChromaDB...")
num_added = vector_store.add_documents(chunks)
print(f"[+] Successfully stored {num_added} chunks into collection!")
print(f"[+] Total stored documents in ChromaDB: {vector_store.count()}")
print(f"[+] Indexed papers in DB: {vector_store.get_paper_list()}")

# 3. Test Retrieval with two academic questions
test_queries = [
    "What dataset was used for training or benchmarking?",
    "Explain the multi-head attention and transformer block design."
]

print("\n" + "=" * 60)
print("TESTING RETRIEVAL FROM CHROMADB")
print("=" * 60)

for query in test_queries:
    print(f"\nQuery: '{query}'")
    results = vector_store.similarity_search(query, k=2)

    for i, res in enumerate(results, start=1):
        print(f"\n  Result #{i} [Score: {res['score']}]")
        print(f"  Source Paper : {res['paper_name']}")
        print(f"  Citation     : Page {res['page']} (Chunk ID: {res['chunk_id']})")
        print(f"  Excerpt      : {res['text'][:220].strip()}...")

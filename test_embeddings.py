import os
import numpy as np
from src.pdf_reader import extract_text
from src.chunking import chunk_pages
from src.embeddings import EmbeddingModel

# 1. Pipeline integration: PDF -> Chunks
pdf_path = os.path.join("data", "papers", "sample.pdf")
paper_name = os.path.basename(pdf_path)

print(f"[1] Extracting text from {pdf_path}...")
pages = extract_text(pdf_path)

print(f"[2] Chunking pages...")
chunks = chunk_pages(pages, paper_name=paper_name, chunk_size=1000, chunk_overlap=200)
print(f"[+] Total chunks: {len(chunks)}")

# 2. Initialize embedding model
print(f"\n[3] Loading SentenceTransformer embedding model...")
embedder = EmbeddingModel()
print("[+] Model loaded successfully!")

# 3. Generate embeddings for all chunks
print(f"\n[4] Generating embeddings for {len(chunks)} chunks...")
chunk_texts = [chunk.page_content for chunk in chunks]
chunk_embeddings = embedder.embed_documents(chunk_texts)

dim = len(chunk_embeddings[0])
print(f"[+] Successfully generated {len(chunk_embeddings)} embeddings!")
print(f"[+] Vector dimensionality: {dim} dimensions")
print(f"[+] Sample vector values (first 5 numbers): {chunk_embeddings[0][:5]}")

# 4. Demonstrate semantic similarity with a sample query
query = "What is the model architecture and parameter count?"
print(f"\n[5] Testing Semantic Search:")
print(f"Query: '{query}'")

query_embedding = np.array(embedder.embed_query(query))
chunk_embeddings_np = np.array(chunk_embeddings)

# Cosine similarity (since embeddings are normalized, dot product = cosine similarity)
similarities = np.dot(chunk_embeddings_np, query_embedding)
top_idx = int(np.argmax(similarities))
top_score = similarities[top_idx]

print(f"\n[+] Most Relevant Chunk Found (Similarity Score: {top_score:.4f}):")
print(f"Chunk ID  : {chunks[top_idx].metadata['chunk_id']}")
print(f"Page      : {chunks[top_idx].metadata['page']}")
print(f"Content Preview:\n{chunks[top_idx].page_content[:300].strip()}...")

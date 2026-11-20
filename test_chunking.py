import os
from src.pdf_reader import extract_text
from src.chunking import chunk_pages

pdf_path = os.path.join("data", "papers", "sample.pdf")
paper_name = os.path.basename(pdf_path)

print(f"Step 1: Extracting text from '{pdf_path}'...")
pages = extract_text(pdf_path)
print(f"[+] Total pages extracted: {len(pages)}")

print("\nStep 2: Splitting text into chunks (chunk_size=1000, chunk_overlap=200)...")
chunks = chunk_pages(pages, paper_name=paper_name, chunk_size=1000, chunk_overlap=200)
print(f"[+] Total chunks created: {len(chunks)}\n")

# Display first 3 chunks to verify content and metadata
print("=== Sample Chunk Inspection ===")
for i, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {i} ---")
    print(f"Metadata : {chunk.metadata}")
    print(f"Length   : {len(chunk.page_content)} characters")
    print("Content preview:")
    print(chunk.page_content[:300].strip() + ("..." if len(chunk.page_content) > 300 else ""))

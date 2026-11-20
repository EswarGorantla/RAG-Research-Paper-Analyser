import os
from src.pdf_reader import extract_text

pdf_path = os.path.join("data", "papers", "sample.pdf")

if not os.path.exists(pdf_path):
    print(f"[!] Error: File not found at '{pdf_path}'.")
    print("Please place a research paper PDF in 'data/papers/' named 'sample.pdf'.")
else:
    pages = extract_text(pdf_path)
    print(f"[+] Successfully loaded: {pdf_path}")
    print(f"[+] Total pages extracted: {len(pages)}")

    for page in pages:
        print(f"\n--- Page {page['page']} ---")
        # Display preview of first 500 characters
        preview = page["text"][:500].strip()
        print(preview if preview else "[Empty page or scanned image]")

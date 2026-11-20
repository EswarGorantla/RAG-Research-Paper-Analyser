# 🔬 Research Paper Intelligence Assistant (Advanced Multi-Stage RAG)

A production-grade, state-of-the-art **Retrieval-Augmented Generation (RAG)** system designed specifically for rigorous scientific literature analysis, multi-paper comparative synthesis, and page-verified factual grounding.

---

## 🌟 Key Architectural Features

1. **Page-Preserving PDF Ingestion with Typographic Normalization**:
   - Built with `PyMuPDF` (`pymupdf`) to extract clean academic text while strictly preserving physical PDF page numbers (1-indexed).
   - Resolves unicode ligatures (`fi`, `fl`), smart quotes, and recombines words split across line breaks (`trans-\nformer` -> `transformer`).

2. **Semantic Text Chunking**:
   - `RecursiveCharacterTextSplitter` configured for scientific literature (1,000-character segments, 200-character overlap) preserving `paper_name`, `page`, and unique deterministic `chunk_id` tags.

3. **Dense Semantic Embeddings with LRU Caching**:
   - Powered by local `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
   - Fast LRU caching (`lru_cache(maxsize=256)`) for query embeddings to eliminate redundant CPU/GPU transformer vectorization.

4. **Persistent Vector Store with Duplicate Ingestion Bypass**:
   - ChromaDB configured with Cosine Distance indexing (`hnsw:space: cosine`).
   - Automatically detects previously indexed `chunk_id`s to skip redundant embedding computations (0ms re-indexing).
   - Supports metadata filtering by paper, per-paper chunk statistics, and atomic paper deletion.

5. **Hybrid Search with Heapq Top-K Selection**:
   - Fuses semantic neural representations (ChromaDB) with exact keyword matching (`rank_bm25`) using RRF ($k=60$).
   - BM25 retrieval uses `heapq.nlargest` for $O(N \log k)$ top-$k$ selection instead of full array sorting.

6. **Precision Cross-Encoder with Inference Caching**:
   - Re-evaluates top hybrid candidate passages using full cross-attention (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
   - In-memory prediction cache for `(query, passage)` pairs to eliminate redundant forward passes.

7. **Extractive Contextual Compression with Global Batching**:
   - Intelligently extracts query-relevant sentences and table metrics while flattening all candidate chunks into a single batched Cross-Encoder forward pass (slashing compression latency to < 1 ms).

8. **Real-Time Token Streaming with Google Gemini**:
   - Live token-by-token streaming via **`gemini-3.5-flash`** (with automated fallback to `gemini-3.5-flash-lite`).
   - Enforces strict page citations in the format `[paper.pdf, Page X]` with an expandable audit trail.

9. **Streamlit Singleton Architecture (`@st.cache_resource`)**:
   - Caches `RAGPipeline` and all neural model weights in RAM across user re-runs, eliminating model reload cold-starts.
   - Includes real-time token streaming (`st.write_stream`) and Markdown report export.

---

## 📁 Project Structure

```
RAG Project/
├── app.py                      # Interactive Streamlit Web Dashboard
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (GEMINI_API_KEY)
├── .env.example                # Example environment template
├── .gitignore                  # Git ignore rules protecting .env and DB
├── README.md                   # Documentation & usage guide
│
├── data/
│   ├── papers/                 # Local PDF storage (sample.pdf, etc.)
│   └── chroma_db/              # Persistent ChromaDB vector database
│
├── models/                     # Locally downloaded neural models
│   ├── all-MiniLM-L6-v2/       # Dense embedding transformer (384d)
│   └── ms-marco-MiniLM-L-6-v2/ # Cross-Encoder reranker model
│
└── src/                        # Core RAG Modular Engine
    ├── pdf_reader.py           # Page-indexed PDF text extractor
    ├── chunking.py             # Page-preserving text splitter
    ├── embeddings.py           # Dense embedding wrapper with LRU cache
    ├── vector_store.py         # ChromaDB persistence & metadata manager
    ├── bm25_retriever.py       # BM25 keyword indexer & searcher
    ├── hybrid_retriever.py     # Reciprocal Rank Fusion (RRF) combiner
    ├── reranker.py             # Cross-Encoder precision reranker
    ├── compression.py          # Contextual extractive compressor
    ├── generator.py            # Grounded Gemini prompt & citation generator
    └── pipeline.py             # Master RAG pipeline orchestrator
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Configure API Key
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Launch the Web Application
```powershell
.\venv\Scripts\streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Pipeline Test Verification

To run standalone tests for each individual module:
```powershell
.\venv\Scripts\python test_pdf.py          # Step 1: PDF extraction
.\venv\Scripts\python test_chunking.py     # Step 2: Text chunking
.\venv\Scripts\python test_embeddings.py   # Step 3: Embeddings
.\venv\Scripts\python test_chroma.py       # Step 4: Vector store
.\venv\Scripts\python test_hybrid.py       # Step 5: Hybrid retrieval
.\venv\Scripts\python test_reranker.py     # Step 6: Cross-Encoder rerank
.\venv\Scripts\python test_compression.py  # Step 7: Context compression
.\venv\Scripts\python test_generator.py    # Step 8: Live LLM answer generation
```


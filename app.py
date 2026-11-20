import os
import re
import json
import time
import streamlit as st
from dotenv import load_dotenv
from src.pipeline import RAGPipeline
from src.pdf_reader import extract_text

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Research Paper Intelligence Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()


# ----------------- OPTIMIZED SINGLETON PIPELINE -----------------
@st.cache_resource(show_spinner="Initializing Neural Models & Vector Store...")
def get_shared_pipeline() -> RAGPipeline:
    """
    Loads SentenceTransformer, CrossEncoder, and ChromaDB once per server process.
    Prevents re-allocating model weights and memory churn across Streamlit re-runs.
    """
    return RAGPipeline()


pipeline: RAGPipeline = get_shared_pipeline()

# Session State for user interactions
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "extracted_pages_cache" not in st.session_state:
    st.session_state.extracted_pages_cache = {}


# ----------------- CUSTOM CSS THEME -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        padding: 1.25rem 1.6rem;
        background: linear-gradient(135deg, #0B1329 0%, #1E293B 50%, #2E1065 100%);
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #CBD5E1;
        line-height: 1.45;
    }
    
    /* Real-Time Status Pills */
    .status-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.85rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
        color: #F8FAFC;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
    }
    
    /* Metric Cards */
    .stat-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    .stat-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E293B;
    }
    .stat-lbl {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Verified Citation Badges */
    .citation-badge-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        color: #166534;
    }
    .page-tag {
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: 600;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-size: 0.78rem;
        border: 1px solid #86EFAC;
    }
    
    /* Telemetry Pill */
    .telemetry-pill {
        display: inline-block;
        font-size: 0.75rem;
        padding: 0.25rem 0.6rem;
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        color: #475569;
        margin-bottom: 0.75rem;
    }
    
    .stButton>button {
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### 📚 Research Repository")

    # Quick Sample Paper Loader
    sample_pdf_path = os.path.join("data", "papers", "sample.pdf")
    indexed_papers = pipeline.get_indexed_papers()

    col_sample, col_clear = st.columns(2)
    with col_sample:
        if st.button("✨ Load Sample", help="Index the 3-page Landmark Transformer sample paper", use_container_width=True):
            if os.path.exists(sample_pdf_path):
                with st.spinner("Indexing sample paper..."):
                    summary = pipeline.ingest_paper(sample_pdf_path)
                    st.success(f"Indexed {summary['paper_name']}!")
                    st.rerun()
            else:
                st.error("sample.pdf not found in data/papers/")

    with col_clear:
        if st.button("🗑️ Reset DB", help="Clear all indexed papers and reset vector database", use_container_width=True):
            pipeline.clear_database()
            st.session_state.chat_history = []
            st.session_state.extracted_pages_cache = {}
            st.warning("Database cleared!")
            st.rerun()

    st.divider()

    # Upload PDF Files
    st.markdown("#### 📤 Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Upload PDF files for RAG analysis:",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload single or multiple papers for synthesis and comparison"
    )

    if uploaded_files:
        if st.button("📥 Ingest Uploaded Papers", type="primary", use_container_width=True):
            papers_dir = os.path.join("data", "papers")
            os.makedirs(papers_dir, exist_ok=True)

            prog_bar = st.progress(0)
            total = len(uploaded_files)

            for i, uploaded_file in enumerate(uploaded_files):
                save_path = os.path.join(papers_dir, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Processing ({i+1}/{total}) '{uploaded_file.name}'..."):
                    summary = pipeline.ingest_paper(save_path)

                prog_bar.progress((i + 1) / total)

            st.success(f"Successfully indexed {len(uploaded_files)} paper(s)!")
            st.rerun()

    # Indexed Papers List & Stats
    st.divider()
    stats = pipeline.get_stats()
    st.markdown("#### 📑 Indexed Documents")

    stat_c1, stat_c2 = st.columns(2)
    with stat_c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-val">{stats['total_papers']}</div>
            <div class="stat-lbl">Papers</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-val">{stats['total_chunks']}</div>
            <div class="stat-lbl">Total Chunks</div>
        </div>
        """, unsafe_allow_html=True)

    if stats["paper_names"]:
        for p in stats["paper_names"]:
            p_chunks = stats["paper_stats"].get(p, 0)
            p_col1, p_col2 = st.columns([4, 1])
            with p_col1:
                st.caption(f"📄 **{p}** ({p_chunks} chunks)")
            with p_col2:
                if st.button("✕", key=f"del_{p}", help=f"Delete {p} from database"):
                    pipeline.delete_paper(p)
                    if p in st.session_state.extracted_pages_cache:
                        del st.session_state.extracted_pages_cache[p]
                    st.rerun()
    else:
        st.info("No papers indexed yet. Click 'Load Sample' or upload a PDF above.")

    # Advanced Settings Accordion
    st.divider()
    with st.expander("⚙️ RAG Engine Hyperparameters", expanded=False):
        selected_model = st.selectbox(
            "Gemini Model",
            options=["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-pro"],
            index=0,
            help="gemini-3.1-flash-lite delivers ~1s ultra-low latency with instant token streaming"
        )
        temp_val = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05, help="Lower = strictly grounded")
        candidate_k = st.slider("Initial Hybrid Candidates (k)", min_value=4, max_value=16, value=6, step=2)
        top_k = st.slider("Reranker Top-K Chunks", min_value=1, max_value=6, value=2, step=1)
        max_sentences = st.slider("Compression Sentences / Chunk", min_value=1, max_value=6, value=3, step=1)
        pipeline.compressor.max_sentences_per_chunk = max_sentences

        api_key_input = st.text_input("Gemini API Key Override", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        if api_key_input and api_key_input != pipeline.generator.api_key:
            pipeline.generator = pipeline.generator.__class__(api_key=api_key_input, model=selected_model)


# ----------------- HERO HEADER -----------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🔬</span> Research Paper Intelligence Assistant
    </div>
    <div class="hero-subtitle">
        Enterprise Advanced RAG: PyMuPDF Text Cleaning • ChromaDB Dense (384d) with LRU Cache • BM25 Top-K Heap • Batched Cross-Encoder Reranker • Extractive Sentence Compression • Live Streaming Gemini 3.5 Flash
    </div>
    <div class="status-bar">
        <span class="status-pill"><span class="status-dot"></span> ChromaDB Dense (384d)</span>
        <span class="status-pill"><span class="status-dot"></span> BM25 Keyword Search</span>
        <span class="status-pill"><span class="status-dot"></span> Cross-Encoder MS-MARCO</span>
        <span class="status-pill"><span class="status-dot"></span> Batched Context Compressor</span>
        <span class="status-pill"><span class="status-dot"></span> Real-Time Streaming LLM</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------- MAIN TABS -----------------
tab_qa, tab_compare, tab_xray, tab_viewer = st.tabs([
    "💬 Research Q&A",
    "⚖️ Multi-Paper Comparative Matrix",
    "🔬 Pipeline X-Ray (Retrieval & Rerank Inspector)",
    "📖 PDF Document & Page Browser"
])


# ==============================================================================
# TAB 1: RESEARCH Q&A (WITH REAL-TIME TOKEN STREAMING)
# ==============================================================================
with tab_qa:
    indexed_papers = pipeline.get_indexed_papers()

    # Scope Bar
    scope_c1, scope_c2 = st.columns([1, 2])
    with scope_c1:
        qa_scope = st.radio(
            "Search Scope:",
            options=["All Indexed Papers", "Specific Paper"],
            horizontal=True
        )
    with scope_c2:
        selected_paper_target = None
        if qa_scope == "Specific Paper":
            if indexed_papers:
                selected_paper_target = st.selectbox("Select Paper to Analyze:", options=indexed_papers)
            else:
                st.warning("No papers indexed yet.")

    # Quick Suggestions
    st.markdown("**💡 Quick Research Inquiries:**")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)

    prompt_trigger = None
    if q_col1.button("🧠 Model Architecture", use_container_width=True):
        prompt_trigger = "What is the detailed model architecture, attention mechanism, and total parameter count?"
    if q_col2.button("⏱️ Hardware Latency & Specs", use_container_width=True):
        prompt_trigger = "What is the CPU and GPU latency, model size, and real-time execution performance?"
    if q_col3.button("📊 Datasets & Validations", use_container_width=True):
        prompt_trigger = "What datasets, cross-validations, and benchmarks were evaluated in the paper?"
    if q_col4.button("🔬 Ablations & Findings", use_container_width=True):
        prompt_trigger = "What key findings, ablation studies, or architectural comparisons are reported?"

    # Chat Input
    user_query = st.chat_input("Ask any question regarding the research papers...")
    if prompt_trigger:
        user_query = prompt_trigger

    # Live streaming execution when user submits question
    if user_query:
        if not indexed_papers:
            st.error("Please upload a paper or click '✨ Load Sample' in the sidebar to get started.")
        else:
            # 1. Render User Message
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**{user_query}**")

            # 2. Render Assistant Message with Streaming
            with st.chat_message("assistant", avatar="🤖"):
                t_start = time.perf_counter()
                with st.spinner("🔍 Running Hybrid Retrieval, Cross-Encoder Reranking & Compression..."):
                    stream, citations, context, prep_timings, audit, model_name = pipeline.query_stream(
                        question=user_query,
                        target_paper=selected_paper_target if qa_scope == "Specific Paper" else None,
                        top_k=top_k,
                        candidate_k=candidate_k,
                        temperature=temp_val,
                        model_name=selected_model
                    )

                # Stream response tokens dynamically onto UI
                full_answer = st.write_stream(stream)
                total_duration_ms = round((time.perf_counter() - t_start) * 1000, 1)

                prep_timings["total"] = total_duration_ms

                # Render Citation Cards below streamed response
                if citations:
                    st.markdown("##### 📌 Verified Citations & Page References")
                    cit_cols = st.columns(min(3, len(citations)))
                    for idx, cit in enumerate(citations):
                        with cit_cols[idx % len(cit_cols)]:
                            st.markdown(
                                f"""<div class="citation-badge-card">
                                📄 <b>{cit['paper_name']}</b><br>
                                <span class="page-tag">Page {cit['page']}</span> • Chunk: <code>{cit['chunk_id']}</code>
                                </div>""",
                                unsafe_allow_html=True
                            )

                # Telemetry Pill
                saved = audit.get("tokens_saved_percent", 0)
                st.markdown(
                    f"""<div class="telemetry-pill">
                    ⚡ <b>{total_duration_ms} ms</b> (Retrieval: {prep_timings.get('retrieval', 0)}ms | Rerank: {prep_timings.get('rerank', 0)}ms | Compression: {prep_timings.get('compression', 0)}ms) • <b>{saved}% Tokens Saved</b> • Model: <code>{model_name}</code>
                    </div>""",
                    unsafe_allow_html=True
                )

            # Store in session chat history
            st.session_state.chat_history.append({
                "question": user_query,
                "answer": full_answer,
                "citations": citations,
                "context": context,
                "timings": prep_timings,
                "audit": audit,
                "model": model_name
            })

    # Render Prior Chat History (when not actively streaming new prompt)
    if st.session_state.chat_history and not user_query:
        top_btn_col1, top_btn_col2 = st.columns([5, 1])
        with top_btn_col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        for chat_item in reversed(st.session_state.chat_history):
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**{chat_item['question']}**")

            with st.chat_message("assistant", avatar="🤖"):
                t = chat_item.get("timings", {})
                if t:
                    audit = chat_item.get("audit", {})
                    saved = audit.get("tokens_saved_percent", 0)
                    st.markdown(
                        f"""<div class="telemetry-pill">
                        ⚡ <b>{t.get('total', 0)} ms</b> (Retrieval: {t.get('retrieval', 0)}ms | Rerank: {t.get('rerank', 0)}ms | Compression: {t.get('compression', 0)}ms) • <b>{saved}% Tokens Saved</b> • Model: <code>{chat_item.get('model', 'gemini')}</code>
                        </div>""",
                        unsafe_allow_html=True
                    )

                st.markdown(chat_item["answer"])

                if chat_item["citations"]:
                    st.markdown("##### 📌 Verified Citations & Page References")
                    cit_cols = st.columns(min(3, len(chat_item["citations"])))
                    for idx, cit in enumerate(chat_item["citations"]):
                        with cit_cols[idx % len(cit_cols)]:
                            st.markdown(
                                f"""<div class="citation-badge-card">
                                📄 <b>{cit['paper_name']}</b><br>
                                <span class="page-tag">Page {cit['page']}</span> • Chunk: <code>{cit['chunk_id']}</code>
                                </div>""",
                                unsafe_allow_html=True
                            )

                if chat_item["context"]:
                    with st.expander("🔍 View Retrieved & Compressed Context Chunks"):
                        for i, ctx in enumerate(chat_item["context"], start=1):
                            st.markdown(f"**Excerpt #{i} — [{ctx.get('paper_name')}, Page {ctx.get('page')}]**")
                            if "rerank_score" in ctx:
                                st.caption(f"Cross-Encoder Score: {ctx['rerank_score']} | Original: {ctx.get('original_chars', len(ctx['text']))} chars → Compressed: {ctx.get('compressed_chars', len(ctx.get('compressed_text', '')))} chars")
                            st.code(ctx.get("compressed_text", ctx.get("text")), language="text")
                            st.divider()

    elif not st.session_state.chat_history and not user_query:
        st.info("👋 Welcome! Select a quick inquiry above or type your research question to start.")


# ==============================================================================
# TAB 2: MULTI-PAPER COMPARATIVE MATRIX
# ==============================================================================
with tab_compare:
    st.markdown("### ⚖️ Multi-Paper Comparative Analysis")
    st.markdown("Generate an automated side-by-side comparative matrix evaluating architectures, datasets, parameter counts, and benchmark results across papers.")

    indexed_papers = pipeline.get_indexed_papers()

    if len(indexed_papers) < 2:
        st.warning(f"Currently {len(indexed_papers)} paper(s) indexed. Please upload at least 2 papers in the sidebar (or load the sample and upload another) to use side-by-side comparison.")
    else:
        selected_for_comp = st.multiselect(
            "Select Papers to Compare:",
            options=indexed_papers,
            default=indexed_papers[:min(3, len(indexed_papers))]
        )

        if st.button("⚡ Generate Comprehensive Comparative Matrix", type="primary"):
            comp_query = (
                f"Compare the following papers: {', '.join(selected_for_comp)} across these specific criteria:\n"
                f"1. Core Problem & Novelty\n"
                f"2. Model Architecture & Layer Design\n"
                f"3. Total Parameter Count & Memory\n"
                f"4. Hardware Latency & Efficiency\n"
                f"5. Datasets, Evaluation Metrics & Accuracy\n\n"
                "Format the comparison into an executive summary followed by a comprehensive Markdown Comparison Table with verified page citations for every cell."
            )

            with st.spinner("Extracting balanced evidence from selected papers and synthesizing comparison..."):
                comp_result = pipeline.query(
                    question=comp_query,
                    compare_papers=selected_for_comp,
                    top_k=max(4, len(selected_for_comp) * 2),
                    candidate_k=16,
                    temperature=0.1,
                    model_name=selected_model
                )

            st.markdown("#### 📊 Comparative Synthesis Result")
            st.markdown(comp_result["answer"])

            # Export Button
            st.download_button(
                label="📥 Download Comparison Report (Markdown)",
                data=comp_result["answer"],
                file_name="comparative_research_matrix.md",
                mime="text/markdown",
                use_container_width=True
            )

            if comp_result["citations"]:
                st.markdown("##### 📌 Citations Across Compared Papers")
                for cit in comp_result["citations"]:
                    st.markdown(f"- 📄 **{cit['paper_name']}** — Page {cit['page']} (`{cit['chunk_id']}`)")


# ==============================================================================
# TAB 3: PIPELINE X-RAY (RETRIEVAL & RERANK INSPECTOR)
# ==============================================================================
with tab_xray:
    st.markdown("### 🔬 Pipeline X-Ray & Retrieval Audit Trail")
    st.markdown("Inspect how the RAG pipeline transforms raw document chunks through Hybrid Retrieval, Reciprocal Rank Fusion, Cross-Encoder Reranking, and Contextual Compression.")

    if not st.session_state.chat_history:
        st.info("Run a query in the **Research Q&A** tab to inspect real-time retrieval audit logs.")
    else:
        latest_interaction = st.session_state.chat_history[-1]
        st.markdown(f"**Inspecting Latest Query:** *'{latest_interaction['question']}'*")

        audit = latest_interaction.get("audit", {})
        timings = latest_interaction.get("timings", {})

        eff_c1, eff_c2, eff_c3, eff_c4 = st.columns(4)
        with eff_c1:
            st.metric("Initial Hybrid Candidates", audit.get("initial_candidates_count", 0))
        with eff_c2:
            st.metric("Top-K Post-Rerank", audit.get("reranked_count", 0))
        with eff_c3:
            st.metric("Context Tokens Saved", f"{audit.get('tokens_saved_percent', 0)}%")
        with eff_c4:
            st.metric("Total Latency", f"{timings.get('total', 0)} ms")

        # Transformation Stages
        st.markdown("#### 1. Cross-Encoder Reranking Transformation")
        reranked_items = audit.get("candidates_post_rerank", [])
        if reranked_items:
            table_rows = []
            for rank, item in enumerate(reranked_items, start=1):
                table_rows.append({
                    "Rank": rank,
                    "Paper": item.get("paper_name"),
                    "Page": item.get("page"),
                    "Chunk ID": item.get("chunk_id"),
                    "Cross-Encoder Score": item.get("rerank_score", "N/A"),
                    "Preview": item.get("text", "")[:120] + "..."
                })
            st.table(table_rows)

        st.markdown("#### 2. Sentence-Level Contextual Compression Breakdown")
        compressed_items = latest_interaction.get("context", [])
        for idx, item in enumerate(compressed_items, start=1):
            with st.expander(f"Chunk #{idx} — [{item.get('paper_name')}, Page {item.get('page')}]"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Original Extracted Chunk**")
                    st.caption(f"Length: {item.get('original_chars', len(item.get('text', '')))} characters")
                    st.text(item.get("text", ""))
                with c2:
                    st.markdown("**Compressed Query-Focused Content**")
                    st.caption(f"Length: {item.get('compressed_chars', len(item.get('compressed_text', '')))} characters")
                    st.text(item.get("compressed_text", item.get("text", "")))


# ==============================================================================
# TAB 4: PDF DOCUMENT & PAGE BROWSER
# ==============================================================================
with tab_viewer:
    st.markdown("### 📖 PDF Document & Page Browser")
    st.markdown("Directly browse any page of your indexed research papers to manually verify citations and inspect extracted text.")

    indexed_papers = pipeline.get_indexed_papers()

    if not indexed_papers:
        st.info("No papers indexed yet.")
    else:
        v_col1, v_col2 = st.columns([2, 1])
        with v_col1:
            browse_paper = st.selectbox("Select Paper:", options=indexed_papers, key="browse_paper_select")

        paper_path = os.path.join("data", "papers", browse_paper)
        if browse_paper not in st.session_state.extracted_pages_cache:
            if os.path.exists(paper_path):
                st.session_state.extracted_pages_cache[browse_paper] = extract_text(paper_path)
            else:
                st.session_state.extracted_pages_cache[browse_paper] = []

        pages_data = st.session_state.extracted_pages_cache.get(browse_paper, [])

        if not pages_data:
            st.warning(f"Could not load PDF source file for {browse_paper}. Please ensure it exists in data/papers/.")
        else:
            with v_col2:
                selected_page_num = st.selectbox(
                    "Select Page:",
                    options=[p["page"] for p in pages_data],
                    format_func=lambda x: f"Page {x}"
                )

            search_highlight = st.text_input("Highlight keyword on this page:", value="")

            page_entry = next((p for p in pages_data if p["page"] == selected_page_num), None)

            if page_entry:
                st.markdown(f"#### 📄 {browse_paper} — Page {selected_page_num} of {len(pages_data)}")
                page_text = page_entry["text"]

                if search_highlight.strip():
                    highlighted_text = re.sub(
                        re.escape(search_highlight.strip()),
                        f"**:orange[{search_highlight.strip()}]**",
                        page_text,
                        flags=re.IGNORECASE
                    )
                    st.markdown(highlighted_text)
                else:
                    st.code(page_text, language="text")

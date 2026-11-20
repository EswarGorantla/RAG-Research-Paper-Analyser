import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def build_rag_prompt(query: str, context_chunks: list[dict], is_comparison: bool = False) -> str:
    """
    Constructs a grounded academic prompt enforcing strict citation rules.
    If is_comparison is True, instructs the LLM to provide a structured Markdown comparison.
    """
    context_sections = []
    papers_present = set()
    for i, chunk in enumerate(context_chunks, start=1):
        paper = chunk.get("paper_name", "Unknown Paper")
        page = chunk.get("page", "Unknown Page")
        text = chunk.get("compressed_text", chunk.get("text", "")).strip()
        papers_present.add(paper)

        section = f"--- [EXCERPT #{i} | Paper: {paper} | Page: {page}] ---\n{text}"
        context_sections.append(section)

    formatted_context = "\n\n".join(context_sections)

    comparison_instruction = ""
    if is_comparison or len(papers_present) > 1 or "compare" in query.lower() or "versus" in query.lower():
        comparison_instruction = """
5. Comparison Format: Since this request involves multiple papers or comparison, provide:
   - A high-level Executive Comparison Summary.
   - A structured Markdown Comparison Table highlighting key dimensions (e.g., Architecture, Parameters, Dataset, Latency, Accuracy) with inline citations for each cell.
   - Key Differences & Trade-offs synthesized from the papers.
"""

    prompt = f"""You are an expert AI Research Assistant specialized in scientific literature analysis.

Answer the user's question accurately using ONLY the research paper excerpts provided below.

STRICT INSTRUCTIONS:
1. Grounding: Answer strictly using facts and metrics from the excerpts. Do not hallucinate or extrapolate beyond this evidence.
2. In-text Citations: Whenever you state a fact, finding, or number, append the exact inline citation in square brackets, e.g. [{context_chunks[0].get('paper_name', 'paper.pdf') if context_chunks else 'paper.pdf'}, Page {context_chunks[0].get('page', 1) if context_chunks else 1}].
3. Format: Be concise, structured, and factual. Use bullet points or tables for key metrics where helpful.
4. If the provided excerpts do not contain enough information to answer the question, state:
   "The provided paper excerpts do not contain sufficient information to answer this question."{comparison_instruction}

EXCERPTS FROM UPLOADED PAPERS:
============================================================
{formatted_context}
============================================================

USER QUESTION: {query}

ANSWER (with page-level citations):"""
    return prompt


class LLMGenerator:
    """
    Generates answers from retrieved context using Google Gemini with page-level citations.
    Optimized for ultra-low latency with gemini-3.1-flash-lite and token streaming.
    """

    def __init__(self, api_key: str | None = None, model: str = "gemini-3.1-flash-lite"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[!] Warning: Could not initialize Gemini client: {e}")

    def generate_answer(
        self,
        query: str,
        context_chunks: list[dict],
        temperature: float = 0.2,
        model_name: str | None = None,
        is_comparison: bool = False,
    ) -> dict:
        """
        Generates an answer based on compressed context chunks.
        """
        if not context_chunks:
            return {
                "answer": "No relevant excerpts were found in the uploaded research papers.",
                "citations": [],
                "model_used": "none",
                "prompt": ""
            }

        prompt = build_rag_prompt(query, context_chunks, is_comparison=is_comparison)

        # Unique citations list
        citations = []
        seen = set()
        for c in context_chunks:
            ref = (c.get("paper_name"), c.get("page"))
            if ref not in seen:
                seen.add(ref)
                citations.append({
                    "paper_name": c.get("paper_name"),
                    "page": c.get("page"),
                    "chunk_id": c.get("chunk_id")
                })

        chosen_model = model_name or self.model_name

        # If Gemini API key is configured, call the live LLM
        if self.client:
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=1024
            )

            models_to_try = [chosen_model]
            if chosen_model != "gemini-3.1-flash-lite":
                models_to_try.append("gemini-3.1-flash-lite")
            if "gemini-3.5-flash-lite" not in models_to_try:
                models_to_try.append("gemini-3.5-flash-lite")

            for m in models_to_try:
                try:
                    response = self.client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=config
                    )
                    answer_text = response.text
                    return {
                        "answer": answer_text,
                        "citations": citations,
                        "model_used": m,
                        "prompt": prompt
                    }
                except Exception as e:
                    print(f"[!] Live LLM call failed with {m}: {e}. Retrying next model...")

        # Fallback simulation if no API key is provided
        preview_points = []
        for c in context_chunks:
            paper = c.get("paper_name")
            page = c.get("page")
            text = c.get("compressed_text", c.get("text", "")).strip()
            preview_points.append(f"- Excerpt from [{paper}, Page {page}]:\n  \"{text[:200]}...\"")

        simulated_answer = (
            f"[Note: Set your GEMINI_API_KEY in .env to enable live LLM generation]\n\n"
            f"Based on the retrieved context for: '{query}'\n\n"
            + "\n\n".join(preview_points)
        )

        return {
            "answer": simulated_answer,
            "citations": citations,
            "model_used": "simulated_preview",
            "prompt": prompt
        }

    def generate_answer_stream(
        self,
        query: str,
        context_chunks: list[dict],
        temperature: float = 0.2,
        model_name: str | None = None,
        is_comparison: bool = False,
    ):
        """
        Yields text chunks in real-time as they are generated by the Gemini API.
        Falls back seamlessly to standard generation if streaming fails.
        """
        if not context_chunks:
            yield "No relevant excerpts were found in the uploaded research papers."
            return

        prompt = build_rag_prompt(query, context_chunks, is_comparison=is_comparison)
        chosen_model = model_name or self.model_name

        if self.client:
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=1024
            )

            models_to_try = [chosen_model]
            if chosen_model != "gemini-3.1-flash-lite":
                models_to_try.append("gemini-3.1-flash-lite")
            if "gemini-3.5-flash-lite" not in models_to_try:
                models_to_try.append("gemini-3.5-flash-lite")

            for m in models_to_try:
                try:
                    stream = self.client.models.generate_content_stream(
                        model=m,
                        contents=prompt,
                        config=config
                    )
                    for chunk in stream:
                        if chunk and chunk.text:
                            yield chunk.text
                    return
                except Exception as e:
                    print(f"[!] Streaming failed with {m}: {e}. Retrying next model...")

        # Fallback to synchronous generation if stream unavailable
        res = self.generate_answer(
            query=query,
            context_chunks=context_chunks,
            temperature=temperature,
            model_name=model_name,
            is_comparison=is_comparison
        )
        yield res["answer"]




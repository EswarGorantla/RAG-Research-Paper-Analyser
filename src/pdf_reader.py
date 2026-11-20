import re
import pymupdf  # Modern PyMuPDF API (avoids deprecation warnings)


def clean_pdf_text(text: str) -> str:
    """
    Normalizes academic PDF text:
    - Resolves unicode ligatures (e.g. 'fi', 'fl')
    - Recombines hyphenated line breaks (e.g. 'trans-\nformer' -> 'transformer')
    - Normalizes inconsistent whitespace while preserving paragraph structure
    """
    if not text:
        return ""

    # Fix common academic typography ligatures and quotes
    text = (
        text.replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", " - ")
        .replace("–", "-")
    )

    # Recombine words split across line breaks by hyphens
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Replace multiple spaces/tabs with single space (preserve single newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse more than 2 consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_text(pdf_path: str) -> list[dict]:
    """
    Extracts text from each page of a PDF document while preserving page numbers
    and normalizing typographic artifacts.

    Args:
        pdf_path (str): File system path to the PDF document.

    Returns:
        list[dict]: A list of dictionaries containing page number and cleaned text.
                    Example: [{"page": 1, "text": "..."}]
    """
    document = pymupdf.open(pdf_path)
    extracted_pages = []

    for page_number, page in enumerate(document, start=1):
        raw_text = page.get_text()
        cleaned = clean_pdf_text(raw_text)
        extracted_pages.append({
            "page": page_number,
            "text": cleaned
        })

    document.close()
    return extracted_pages

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """
    Creates a RecursiveCharacterTextSplitter configured with hierarchical separators
    ideal for academic documents (paragraphs, sentences, words).
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def chunk_pages(extracted_pages: list[dict], paper_name: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Splits text from extracted PDF pages into chunks while preserving metadata
    (page number, paper name, and chunk ID) for each chunk.

    Args:
        extracted_pages (list[dict]): Output from extract_text() containing 'page' and 'text'.
        paper_name (str): The filename or identifier of the research paper.
        chunk_size (int): Target maximum characters per chunk.
        chunk_overlap (int): Number of overlapping characters between consecutive chunks.

    Returns:
        list[Document]: LangChain Document objects with page_content and metadata.
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []

    for page_data in extracted_pages:
        page_num = page_data["page"]
        text = page_data["text"]

        if not text.strip():
            continue

        # Split text of the current page into chunks
        page_chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(page_chunks, start=1):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "paper_name": paper_name,
                    "page": page_num,
                    "chunk_id": f"{paper_name}_p{page_num}_c{i}"
                }
            )
            documents.append(doc)

    return documents

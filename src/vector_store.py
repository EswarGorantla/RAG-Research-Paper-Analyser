import os
import chromadb
from langchain_core.documents import Document
from src.embeddings import EmbeddingModel

DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db"
)


class VectorStoreManager:
    """
    Manages persistent ChromaDB vector storage and semantic retrieval for research papers.
    Preserves document content, page-level metadata, and custom IDs.
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = "research_papers",
        embedding_model: EmbeddingModel | None = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = embedding_model or EmbeddingModel()

        # Initialize persistent Chroma client (creates folder if not exists)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Get or create collection configured with cosine similarity distance
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: list[Document]) -> int:
        """
        Embeds and stores a list of LangChain Document objects with their metadata.
        Optimized to skip re-embedding for chunks that already exist in ChromaDB.

        Args:
            documents (list[Document]): Chunks with page_content and metadata.

        Returns:
            int: Number of documents added or already present.
        """
        if not documents:
            return 0

        target_ids = [doc.metadata.get("chunk_id", f"chunk_{i}") for i, doc in enumerate(documents)]

        # Check existing chunk IDs in ChromaDB to avoid redundant embeddings
        existing = self.collection.get(ids=target_ids)
        existing_ids = set(existing.get("ids", [])) if existing else set()

        new_docs = [
            doc for doc, cid in zip(documents, target_ids)
            if cid not in existing_ids
        ]

        if not new_docs:
            # All chunks already indexed
            return len(documents)

        new_ids = [doc.metadata.get("chunk_id", f"chunk_{i}") for i, doc in enumerate(new_docs)]
        new_texts = [doc.page_content for doc in new_docs]
        new_metas = [doc.metadata for doc in new_docs]

        # Generate dense embeddings only for new chunks
        embeddings = self.embedder.embed_documents(new_texts)

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=new_ids,
            documents=new_texts,
            metadatas=new_metas,
            embeddings=embeddings,
        )

        return len(documents)


    def similarity_search(
        self, query: str, k: int = 3, filter_dict: dict | None = None
    ) -> list[dict]:
        """
        Retrieves the top-k most semantically relevant chunks for a given query.

        Args:
            query (str): The search query text.
            k (int): Number of top results to retrieve.
            filter_dict (dict, optional): Metadata filter (e.g. {"paper_name": "sample.pdf"}).

        Returns:
            list[dict]: List of retrieved chunks with text, metadata (page, paper_name), and similarity score.
        """
        query_vector = self.embedder.embed_query(query)

        query_params = {
            "query_embeddings": [query_vector],
            "n_results": k,
        }
        if filter_dict:
            query_params["where"] = filter_dict

        results = self.collection.query(**query_params)

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            ids = results["ids"][0]
            distances = results["distances"][0]

            for doc_text, meta, chunk_id, dist in zip(docs, metadatas, ids, distances):
                # Cosine distance ranges from 0 to 2; similarity = 1 - distance
                similarity_score = 1.0 - dist
                retrieved.append({
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "metadata": meta,
                    "page": meta.get("page"),
                    "paper_name": meta.get("paper_name"),
                    "score": round(similarity_score, 4),
                })

        return retrieved

    def count(self) -> int:
        """Returns total number of chunks currently stored in the collection."""
        return self.collection.count()

    def get_paper_list(self) -> list[str]:
        """Returns a sorted list of unique paper names currently stored."""
        all_meta = self.collection.get(include=["metadatas"])
        if not all_meta or not all_meta.get("metadatas"):
            return []
        papers = {m["paper_name"] for m in all_meta["metadatas"] if m and "paper_name" in m}
        return sorted(list(papers))

    def get_paper_stats(self) -> dict[str, int]:
        """
        Returns a dictionary mapping each paper name to its number of stored chunks.
        """
        all_meta = self.collection.get(include=["metadatas"])
        if not all_meta or not all_meta.get("metadatas"):
            return {}
        stats: dict[str, int] = {}
        for m in all_meta["metadatas"]:
            if m and "paper_name" in m:
                name = m["paper_name"]
                stats[name] = stats.get(name, 0) + 1
        return stats

    def delete_paper(self, paper_name: str) -> int:
        """
        Deletes all chunks belonging to a specific paper.

        Args:
            paper_name (str): Name of the paper to delete.

        Returns:
            int: Number of chunks removed.
        """
        existing = self.collection.get(where={"paper_name": paper_name})
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def clear_all(self):
        """Empties the entire ChromaDB collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


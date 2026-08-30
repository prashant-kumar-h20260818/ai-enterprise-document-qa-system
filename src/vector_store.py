from __future__ import annotations

from typing import Any, Dict, List

import chromadb
import numpy as np
from langchain_core.documents import Document


class VectorStore:
    """Persistent ChromaDB collection configured for cosine distance."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./vector_store",
        reset_collection: bool = False,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        if reset_collection:
            self.reset()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Document embeddings for the RAG assistant",
                "hnsw:space": "cosine",
            },
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

    def count(self) -> int:
        return int(self.collection.count())

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        if not documents:
            return

        ids, metadatas, texts, vectors = [], [], [], []

        for doc, embedding in zip(documents, embeddings):
            metadata = dict(doc.metadata)
            chunk_id = str(metadata.get("chunk_id"))
            if not chunk_id or chunk_id == "None":
                raise ValueError("Every chunk must have metadata['chunk_id']")

            # Chroma metadata values must be scalar values.
            clean_metadata = {
                str(k): v
                for k, v in metadata.items()
                if isinstance(v, (str, int, float, bool)) and v is not None
            }

            ids.append(chunk_id)
            metadatas.append(clean_metadata)
            texts.append(doc.page_content)
            vectors.append(embedding.tolist())

        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
            documents=texts,
        )

    def query(self, query_embedding: np.ndarray, top_k: int) -> Dict[str, Any]:
        total = self.count()
        if total == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(int(top_k), total),
            include=["documents", "metadatas", "distances"],
        )

    def list_sources(self) -> List[str]:
        if self.count() == 0:
            return []
        data = self.collection.get(include=["metadatas"])
        sources = {
            meta.get("source_file", meta.get("source", "unknown"))
            for meta in data.get("metadatas", [])
        }
        return sorted(str(s) for s in sources)

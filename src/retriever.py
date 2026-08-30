from __future__ import annotations

from typing import Any, Dict, List

from .embeddings import EmbeddingManager
from .vector_store import VectorStore


def cosine_distance_to_similarity(distance: float) -> float:
    """Chroma cosine distance is 1 - cosine_similarity."""
    return 1.0 - float(distance)


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.30,
    ) -> List[Dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        query_embedding = self.embedding_manager.encode([query])[0]
        results = self.vector_store.query(query_embedding, top_k=top_k)

        retrieved: List[Dict[str, Any]] = []
        rows = zip(
            results.get("ids", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        )

        for rank, (doc_id, content, metadata, distance) in enumerate(rows, start=1):
            similarity = cosine_distance_to_similarity(distance)
            if similarity >= score_threshold:
                retrieved.append(
                    {
                        "id": doc_id,
                        "content": content,
                        "metadata": metadata or {},
                        "distance": float(distance),
                        "similarity_score": float(similarity),
                        "rank": rank,
                    }
                )
        return retrieved

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .embeddings import EmbeddingManager
from .vector_store import VectorStore


def cosine_distance_to_similarity(distance: float) -> float:
    """Chroma cosine distance is 1 - cosine_similarity."""
    return 1.0 - float(distance)


class RAGRetriever:
    """Hybrid retriever: dense semantic search + lightweight lexical search.

    Dense retrieval is performed by Chroma/HNSW using MiniLM embeddings. A local
    TF-IDF index adds an exact-term/phrase signal that is useful for identifiers,
    formulas, acronyms and wording that dense retrieval can under-rank. The two
    ranked lists are merged using reciprocal-rank fusion (RRF).

    This in-process lexical index is intentionally lightweight for the demo-sized
    corpus. At enterprise scale it should be replaced by a production sparse index
    such as BM25 in Elasticsearch/OpenSearch or an equivalent managed service.
    """

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

        corpus = vector_store.get_all_documents()
        self._ids = list(corpus.get("ids", []) or [])
        self._documents = list(corpus.get("documents", []) or [])
        self._metadatas = list(corpus.get("metadatas", []) or [])
        self._record_by_id = {
            str(doc_id): {
                "id": str(doc_id),
                "content": content or "",
                "metadata": metadata or {},
            }
            for doc_id, content, metadata in zip(self._ids, self._documents, self._metadatas)
        }

        self._tfidf: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        if self._documents:
            try:
                self._tfidf = TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    norm="l2",
                )
                self._tfidf_matrix = self._tfidf.fit_transform(self._documents)
            except ValueError:
                # Extremely small/empty corpora can have no usable vocabulary.
                self._tfidf = None
                self._tfidf_matrix = None

    def _lexical_candidates(self, query: str, candidate_k: int) -> List[Dict[str, Any]]:
        if self._tfidf is None or self._tfidf_matrix is None or not self._documents:
            return []
        try:
            q = self._tfidf.transform([query])
            scores = (self._tfidf_matrix @ q.T).toarray().ravel()
        except Exception:
            return []

        if scores.size == 0:
            return []
        order = np.argsort(scores)[::-1]
        candidates: List[Dict[str, Any]] = []
        for idx in order[:candidate_k]:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            doc_id = str(self._ids[idx])
            base = self._record_by_id.get(doc_id)
            if not base:
                continue
            candidates.append(
                {
                    **base,
                    "lexical_score": score,
                }
            )
        return candidates

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.30,
    ) -> List[Dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        candidate_k = max(int(top_k) * 4, 20)
        query_embedding = self.embedding_manager.encode([query])[0]
        dense = self.vector_store.query(query_embedding, top_k=candidate_k)

        semantic_rows = list(
            zip(
                dense.get("ids", [[]])[0],
                dense.get("documents", [[]])[0],
                dense.get("metadatas", [[]])[0],
                dense.get("distances", [[]])[0],
            )
        )

        merged: Dict[str, Dict[str, Any]] = {}
        semantic_rank: Dict[str, int] = {}
        lexical_rank: Dict[str, int] = {}

        for rank, (doc_id, content, metadata, distance) in enumerate(semantic_rows, start=1):
            doc_id = str(doc_id)
            similarity = cosine_distance_to_similarity(distance)
            semantic_rank[doc_id] = rank
            merged[doc_id] = {
                "id": doc_id,
                "content": content or "",
                "metadata": metadata or {},
                "distance": float(distance),
                "similarity_score": float(similarity),
                "lexical_score": 0.0,
            }

        lexical = self._lexical_candidates(query, candidate_k)
        for rank, item in enumerate(lexical, start=1):
            doc_id = str(item["id"])
            lexical_rank[doc_id] = rank
            if doc_id not in merged:
                merged[doc_id] = {
                    "id": doc_id,
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "distance": 1.0,
                    "similarity_score": 0.0,
                    "lexical_score": float(item["lexical_score"]),
                }
            else:
                merged[doc_id]["lexical_score"] = float(item["lexical_score"])

        # Reciprocal Rank Fusion. Dense retrieval remains the dominant signal,
        # while lexical retrieval rescues exact-term/identifier matches.
        rrf_constant = 60.0
        for doc_id, item in merged.items():
            s_rank = semantic_rank.get(doc_id)
            l_rank = lexical_rank.get(doc_id)
            dense_rrf = (1.0 / (rrf_constant + s_rank)) if s_rank else 0.0
            lexical_rrf = (1.0 / (rrf_constant + l_rank)) if l_rank else 0.0
            item["hybrid_score"] = 0.75 * dense_rrf + 0.25 * lexical_rrf

        ranked = sorted(
            merged.values(),
            key=lambda x: (x["hybrid_score"], x["similarity_score"], x["lexical_score"]),
            reverse=True,
        )

        retrieved: List[Dict[str, Any]] = []
        lexical_rescue_threshold = 0.15
        for item in ranked:
            semantic_ok = float(item["similarity_score"]) >= float(score_threshold)
            lexical_ok = float(item["lexical_score"]) >= lexical_rescue_threshold
            if not (semantic_ok or lexical_ok):
                continue
            item = dict(item)
            item["rank"] = len(retrieved) + 1
            item["retrieval_mode"] = (
                "hybrid" if item["similarity_score"] > 0 and item["lexical_score"] > 0
                else "semantic" if item["similarity_score"] > 0
                else "lexical"
            )
            retrieved.append(item)
            if len(retrieved) >= int(top_k):
                break

        return retrieved

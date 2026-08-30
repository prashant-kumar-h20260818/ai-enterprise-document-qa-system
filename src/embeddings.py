from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """SentenceTransformer embeddings, normalized for cosine similarity."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], show_progress_bar: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        return self.model.encode(
            list(texts),
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

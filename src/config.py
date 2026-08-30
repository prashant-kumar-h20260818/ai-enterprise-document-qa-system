from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    """Central configuration for the RAG pipeline."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama-3.3-70b-versatile"
    # Vision model is configurable because provider model availability can change.
    vision_model: str = "qwen/qwen3.6-27b"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    score_threshold: float = 0.30
    max_tokens: int = 1024
    temperature: float = 0.10
    max_vision_items_per_file: int = 20
    persist_directory: str = "./vector_store"

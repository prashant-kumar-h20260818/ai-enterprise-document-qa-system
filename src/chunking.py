from __future__ import annotations

import hashlib
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def stable_chunk_id(doc: Document, chunk_index: int) -> str:
    """Build a deterministic id so re-ingestion upserts instead of duplicating."""
    source = str(doc.metadata.get("source_file", doc.metadata.get("source", "unknown")))
    locator = str(doc.metadata.get("locator", doc.metadata.get("page_number", doc.metadata.get("page", ""))))
    content_type = str(doc.metadata.get("content_type", "text"))
    payload = f"{source}|{locator}|{content_type}|{chunk_index}|{doc.page_content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["content_length"] = len(chunk.page_content)
        chunk.metadata["chunk_id"] = stable_chunk_id(chunk, i)

    return chunks

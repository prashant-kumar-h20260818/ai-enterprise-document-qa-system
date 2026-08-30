import pytest
from langchain_core.documents import Document
from src.chunking import split_documents


def test_split_documents_preserves_source_metadata():
    docs = [Document(page_content="alpha beta gamma " * 200, metadata={"source_file": "sample.txt"})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(c.metadata["source_file"] == "sample.txt" for c in chunks)
    assert all("chunk_id" in c.metadata for c in chunks)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        split_documents([Document(page_content="hello", metadata={})], chunk_size=100, chunk_overlap=100)

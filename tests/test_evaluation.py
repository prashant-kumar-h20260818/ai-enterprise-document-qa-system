from src.evaluation import retrieval_metrics


def test_retrieval_hit_and_mrr():
    sources = [
        {"source": "wrong.pdf", "page": 1},
        {"source": "resume.pdf", "page": 2},
    ]
    metrics = retrieval_metrics(sources, "resume.pdf", 2)
    assert metrics["hit_at_k"] == 1.0
    assert metrics["reciprocal_rank"] == 0.5


def test_retrieval_miss():
    metrics = retrieval_metrics([{"source": "wrong.pdf", "page": 1}], "resume.pdf", 2)
    assert metrics["hit_at_k"] == 0.0
    assert metrics["reciprocal_rank"] == 0.0

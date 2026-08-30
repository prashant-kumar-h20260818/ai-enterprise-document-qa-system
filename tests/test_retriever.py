from src.retriever import cosine_distance_to_similarity


def test_cosine_distance_conversion():
    assert cosine_distance_to_similarity(0.0) == 1.0
    assert cosine_distance_to_similarity(1.0) == 0.0
    assert cosine_distance_to_similarity(2.0) == -1.0

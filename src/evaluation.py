from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def normalize_source(value: Any) -> str:
    return str(value or "").strip().lower().replace("\\", "/")


def source_matches(retrieved_source: str, expected_source: str) -> bool:
    r = normalize_source(retrieved_source)
    e = normalize_source(expected_source)
    return bool(e) and (r == e or r.endswith("/" + e) or r.endswith(e))


def locator_matches(retrieved_locator: str, expected_locator: str) -> bool:
    r = str(retrieved_locator or "").strip().lower()
    e = str(expected_locator or "").strip().lower()
    return bool(e) and (r == e or e in r)


def retrieval_metrics(
    sources: List[Dict[str, Any]],
    expected_source: str,
    expected_page: Optional[int] = None,
    expected_locator: Optional[str] = None,
) -> Dict[str, float]:
    """Hit@K and reciprocal rank for a gold source/location.

    `expected_page` remains supported for PDFs. `expected_locator` generalizes the
    benchmark to slides, sheets, tables, images, email bodies and other units.
    """
    first_match_rank = None

    for rank, src in enumerate(sources, start=1):
        if not source_matches(src.get("source", ""), expected_source):
            continue

        if expected_locator is not None and not pd.isna(expected_locator) and str(expected_locator).strip():
            if not locator_matches(src.get("locator", ""), str(expected_locator)):
                continue
        elif expected_page is not None and not pd.isna(expected_page):
            try:
                retrieved_page = int(src.get("page"))
                if retrieved_page != int(expected_page):
                    continue
            except (TypeError, ValueError):
                continue

        first_match_rank = rank
        break

    return {
        "hit_at_k": 1.0 if first_match_rank is not None else 0.0,
        "reciprocal_rank": 1.0 / first_match_rank if first_match_rank else 0.0,
    }


def semantic_answer_similarity(answer: str, expected_answer: str, embedding_manager) -> float:
    """Cosine similarity between normalized answer embeddings.

    This is a useful automated proxy, not a complete correctness metric.
    """
    if not answer or not expected_answer:
        return 0.0
    vectors = embedding_manager.encode([answer, expected_answer])
    return float(np.dot(vectors[0], vectors[1]))


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Judge did not return JSON")
    return json.loads(match.group(0))


def llm_judge_answer(llm, question: str, expected_answer: str, actual_answer: str, context: str) -> Dict[str, Any]:
    """Optional LLM-as-judge scoring for correctness and faithfulness.

    Scores are 0..1. This should complement, not replace, human review.
    """
    prompt = f"""You are evaluating a RAG answer. Return ONLY valid JSON.

Question:
{question}

Expected answer:
{expected_answer}

Retrieved context:
{context}

Actual answer:
{actual_answer}

Score:
- correctness: how well the actual answer matches the expected answer (0 to 1)
- faithfulness: whether every factual claim is supported by the retrieved context (0 to 1)

Return:
{{"correctness": 0.0, "faithfulness": 0.0, "reason": "brief reason"}}
"""
    response = llm.invoke(prompt)
    result = _extract_json(response.content)
    return {
        "correctness": float(result.get("correctness", 0.0)),
        "faithfulness": float(result.get("faithfulness", 0.0)),
        "judge_reason": str(result.get("reason", "")),
    }


def validate_eval_dataframe(df: pd.DataFrame) -> None:
    required = {"question", "expected_answer", "expected_source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Evaluation CSV is missing columns: {sorted(missing)}")


def evaluate_dataset(
    df: pd.DataFrame,
    pipeline,
    embedding_manager,
    top_k: int = 5,
    score_threshold: float = 0.30,
    run_generation: bool = True,
    use_llm_judge: bool = False,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate retrieval and optionally end-to-end answer quality."""
    validate_eval_dataframe(df)
    records = []

    for _, row in df.iterrows():
        question = str(row["question"])
        expected_answer = str(row["expected_answer"])
        expected_source = str(row["expected_source"])
        expected_page = row.get("expected_page", None)
        expected_locator = row.get("expected_locator", None)

        if run_generation:
            output = pipeline.answer(
                question,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        else:
            prepared = pipeline.prepare(
                question,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            output = {
                "answer": "",
                "sources": prepared.sources,
                "context": prepared.context,
            }

        metrics = retrieval_metrics(
            output["sources"],
            expected_source=expected_source,
            expected_page=expected_page,
            expected_locator=expected_locator,
        )

        record = {
            "question": question,
            "expected_source": expected_source,
            "expected_page": expected_page,
            "expected_locator": expected_locator,
            **metrics,
        }

        if run_generation:
            record["answer"] = output["answer"]
            record["expected_answer"] = expected_answer
            record["semantic_answer_similarity"] = semantic_answer_similarity(
                output["answer"], expected_answer, embedding_manager
            )

            if use_llm_judge:
                record.update(
                    llm_judge_answer(
                        pipeline.llm,
                        question,
                        expected_answer,
                        output["answer"],
                        output["context"],
                    )
                )

        records.append(record)

    results = pd.DataFrame(records)
    summary = {
        "retrieval_hit_rate_at_k": float(results["hit_at_k"].mean()) if len(results) else 0.0,
        "mrr": float(results["reciprocal_rank"].mean()) if len(results) else 0.0,
    }

    if run_generation and len(results):
        summary["mean_semantic_answer_similarity"] = float(
            results["semantic_answer_similarity"].mean()
        )
        if use_llm_judge and "correctness" in results:
            summary["mean_llm_judge_correctness"] = float(results["correctness"].mean())
            summary["mean_llm_judge_faithfulness"] = float(results["faithfulness"].mean())

    return results, summary

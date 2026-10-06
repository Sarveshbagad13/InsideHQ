"""Evaluation Harness Runner (Feature 15).

Runs full end-to-end RAG pipeline evaluation against the golden dataset, computes
Recall@5, Faithfulness, and Answer Relevance metrics, prints a formatted summary table,
and saves a timestamped scorecard JSON artifact.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.eval.metrics import (
    compute_answer_relevance,
    compute_faithfulness,
    compute_recall_at_k,
)
from app.models.user import User
from app.services.generation import GenerationService, get_generation_service
from app.services.retrieval import RetrievalResult, retrieve_and_rerank

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
SCORECARDS_DIR = EVAL_DIR.parent.parent / "eval_results"


def load_golden_dataset(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load evaluation questions and ground truth from golden JSON file."""
    dataset_file = path or GOLDEN_DATASET_PATH
    with open(dataset_file, "r", encoding="utf-8") as f:
        return json.load(f)


async def evaluate_single_item(
    item: Dict[str, Any],
    db: AsyncSession,
    user: User,
    generation_service: Optional[GenerationService] = None,
    top_n_rerank: int = 5,
) -> Dict[str, Any]:
    """Execute end-to-end retrieval and generation pipeline for one golden question."""
    if generation_service is None:
        generation_service = get_generation_service()

    question = item["question"]
    expected_doc = item.get("expected_document", "")
    expected_sec = item.get("expected_section")
    ground_truth = item.get("ground_truth_answer", "")

    start_time = time.perf_counter()

    # 1. Permission-aware retrieval & reranking
    retrieved_chunks = await retrieve_and_rerank(
        db=db,
        query=question,
        user=user,
        top_k_candidates=20,
        top_n_rerank=top_n_rerank,
        department=item.get("department"),
    )

    # 2. Grounded Answer Generation
    answer, citations = await generation_service.agenerate_answer(
        question=question,
        chunks=retrieved_chunks,
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # 3. Compute Metrics
    recall_5 = compute_recall_at_k(
        retrieved_chunks=retrieved_chunks,
        expected_document=expected_doc,
        expected_section=expected_sec,
        k=5,
    )
    faithfulness = compute_faithfulness(
        answer=answer,
        context_chunks=retrieved_chunks,
    )
    relevance = compute_answer_relevance(
        question=question,
        answer=answer,
    )

    return {
        "id": item.get("id"),
        "question": question,
        "expected_document": expected_doc,
        "expected_section": expected_sec,
        "ground_truth_answer": ground_truth,
        "retrieved_chunk_ids": [str(c.chunk_id) for c in retrieved_chunks],
        "retrieved_count": len(retrieved_chunks),
        "generated_answer": answer,
        "citations_count": len(citations),
        "recall_at_5": recall_5,
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "latency_ms": latency_ms,
    }


async def run_evaluation_suite(
    db: AsyncSession,
    user: Optional[User] = None,
    dataset: Optional[List[Dict[str, Any]]] = None,
    generation_service: Optional[GenerationService] = None,
    save_scorecard: bool = True,
) -> Dict[str, Any]:
    """Run full evaluation harness on golden dataset and generate scorecard.

    Args:
        db: Async database session.
        user: Evaluator user (defaults to admin with full read permissions).
        dataset: Optional list of test items (defaults to golden_dataset.json).
        generation_service: Optional generation service instance.
        save_scorecard: Whether to persist timestamped JSON scorecard artifact.

    Returns:
        Scorecard dictionary containing summary metrics and per-item results.
    """
    items = dataset if dataset is not None else load_golden_dataset()
    if user is None:
        user = User(
            id=uuid.uuid4(),
            name="Evaluation Runner",
            email="evaluator@insidehq.com",
            role="admin",
            department="Engineering",
            is_active=True,
        )

    results: List[Dict[str, Any]] = []

    for item in items:
        res = await evaluate_single_item(
            item=item,
            db=db,
            user=user,
            generation_service=generation_service,
        )
        results.append(res)

    total_items = len(results)
    if total_items > 0:
        mean_recall = round(sum(r["recall_at_5"] for r in results) / total_items, 4)
        mean_faithfulness = round(sum(r["faithfulness"] for r in results) / total_items, 4)
        mean_relevance = round(sum(r["answer_relevance"] for r in results) / total_items, 4)
        mean_latency = round(sum(r["latency_ms"] for r in results) / total_items, 2)
    else:
        mean_recall = 0.0
        mean_faithfulness = 0.0
        mean_relevance = 0.0
        mean_latency = 0.0

    from datetime import timezone

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    scorecard: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": total_items,
        "summary": {
            "recall_at_5": mean_recall,
            "faithfulness": mean_faithfulness,
            "answer_relevance": mean_relevance,
            "average_latency_ms": mean_latency,
        },
        "item_results": results,
    }

    # Print formatted summary table
    print("\n" + "=" * 65)
    print(f"  INSIDEHQ RAG PIPELINE EVALUATION SCORECARD ({timestamp_str})")
    print("=" * 65)
    print(f" Total Questions Evaluated: {total_items}")
    print(f" Recall@5:                  {mean_recall:.2%}")
    print(f" RAGAS Faithfulness:        {mean_faithfulness:.4f}")
    print(f" RAGAS Answer Relevance:    {mean_relevance:.4f}")
    print(f" Average Latency:           {mean_latency:.1f}ms")
    print("=" * 65 + "\n")

    if save_scorecard:
        SCORECARDS_DIR.mkdir(parents=True, exist_ok=True)
        scorecard_file = SCORECARDS_DIR / f"scorecard_{timestamp_str}.json"
        with open(scorecard_file, "w", encoding="utf-8") as f:
            json.dump(scorecard, f, indent=2)
        print(f"Saved scorecard to: {scorecard_file}\n")

    return scorecard


if __name__ == "__main__":
    from app.db.session import AsyncSessionLocal

    async def main():
        async with AsyncSessionLocal() as session:
            await run_evaluation_suite(db=session)

    asyncio.run(main())

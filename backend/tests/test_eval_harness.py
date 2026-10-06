"""Unit and Integration Tests for Evaluation Harness (Feature 15).

Tests:
1. Golden dataset structure and validity (30-50 questions with expected docs/ground truth).
2. Metrics calculation:
   - Recall@5 correctly evaluates matches and mismatches.
   - Faithfulness metric awards 1.0 for grounded answers and penalizes hallucinations.
   - Answer relevance metric evaluates query-answer semantic alignment.
3. Evaluation runner execution and timestamped scorecard generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from app.eval.metrics import (
    compute_answer_relevance,
    compute_faithfulness,
    compute_recall_at_k,
)
from app.eval.runner import (
    GOLDEN_DATASET_PATH,
    SCORECARDS_DIR,
    evaluate_single_item,
    load_golden_dataset,
    run_evaluation_suite,
)
from app.models.user import User
from app.services.generation import NO_INFO_PHRASE, GenerationService
from app.services.retrieval import RetrievalResult


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_eval_chunks():
    """Create sample retrieval chunks for eval test."""
    c1 = RetrievalResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        chunk_text="Primary caregivers receive 16 weeks of fully paid leave after birth or adoption.",
        chunk_metadata={"filename": "leave_policy.pdf", "section_heading": "Parental Leave"},
        document_title="Employee Benefits and Leave Policy 2026",
        rrf_score=0.045,
        rerank_score=9.2,
        rerank_rank=1,
    )
    return [c1]


# ---------------------------------------------------------------------------
# 1. Golden Dataset Tests
# ---------------------------------------------------------------------------
def test_golden_dataset_contains_sufficient_questions():
    """Verify golden dataset exists and contains 30-50 enterprise QA evaluation items."""
    dataset = load_golden_dataset()
    assert len(dataset) >= 30, f"Expected at least 30 questions, found {len(dataset)}"
    assert len(dataset) <= 50

    for item in dataset:
        assert "id" in item
        assert "question" in item and len(item["question"].strip()) > 5
        assert "expected_document" in item
        assert "ground_truth_answer" in item
        assert "allowed_roles" in item
        assert isinstance(item["allowed_roles"], list)


# ---------------------------------------------------------------------------
# 2. Metric Computation Tests
# ---------------------------------------------------------------------------
def test_recall_at_k_match(sample_eval_chunks):
    """Verify Recall@5 returns 1.0 when matching document is present in top-k."""
    score = compute_recall_at_k(
        retrieved_chunks=sample_eval_chunks,
        expected_document="Employee Benefits and Leave Policy 2026",
        expected_section="Parental Leave",
        k=5,
    )
    assert score == 1.0


def test_recall_at_k_mismatch(sample_eval_chunks):
    """Verify Recall@5 returns 0.0 when expected document is missing from top-k."""
    score = compute_recall_at_k(
        retrieved_chunks=sample_eval_chunks,
        expected_document="Corporate Travel and Expense Policy",
        k=5,
    )
    assert score == 0.0


def test_faithfulness_grounded_answer(sample_eval_chunks):
    """Verify Faithfulness returns high score when claims are grounded in context chunks."""
    grounded_answer = "Primary caregivers receive 16 weeks of fully paid leave after birth or adoption."
    score = compute_faithfulness(grounded_answer, sample_eval_chunks)
    assert score == 1.0


def test_faithfulness_abstention_is_faithful():
    """Verify legitimate abstention (insufficient context) receives 1.0 faithfulness."""
    score = compute_faithfulness(NO_INFO_PHRASE, [])
    assert score == 1.0


def test_answer_relevance_high():
    """Verify Answer Relevance returns high score for direct, relevant answer."""
    question = "How many weeks of parental leave do primary caregivers get?"
    answer = "Primary caregivers receive 16 weeks of fully paid parental leave."
    score = compute_answer_relevance(question, answer)
    assert score >= 0.8


# ---------------------------------------------------------------------------
# 3. Runner & Scorecard Persistence Tests
# ---------------------------------------------------------------------------
class MockStreamChunk:
    def __init__(self, text: str):
        self.text = text


@pytest.mark.asyncio
async def test_run_evaluation_suite_generates_and_saves_scorecard(sample_eval_chunks, tmp_path):
    """Verify run_evaluation_suite processes items, computes metrics, and writes scorecard JSON."""
    mock_db = AsyncMock()

    sample_items = [
        {
            "id": "eval_test_1",
            "question": "How many weeks of paid parental leave do primary caregivers receive?",
            "ground_truth_answer": "16 weeks of paid leave.",
            "expected_document": "Employee Benefits and Leave Policy 2026",
            "expected_section": "Parental Leave",
            "department": "HR",
            "allowed_roles": ["employee"],
        }
    ]

    mock_gen = GenerationService()
    mock_gen._client = MagicMock()
    mock_gen._client.models.generate_content_stream.return_value = [
        MockStreamChunk(f"Primary caregivers get 16 weeks of leave [{sample_eval_chunks[0].chunk_id}].")
    ]

    with patch("app.eval.runner.retrieve_and_rerank", new_callable=AsyncMock) as mock_retrieve, \
         patch("app.eval.runner.SCORECARDS_DIR", tmp_path):

        mock_retrieve.return_value = sample_eval_chunks

        scorecard = await run_evaluation_suite(
            db=mock_db,
            dataset=sample_items,
            generation_service=mock_gen,
            save_scorecard=True,
        )

        assert scorecard["total_questions"] == 1
        assert scorecard["summary"]["recall_at_5"] == 1.0
        assert scorecard["summary"]["faithfulness"] >= 0.8
        assert scorecard["summary"]["answer_relevance"] >= 0.8

        # Check file was saved in tmp_path
        saved_files = list(tmp_path.glob("scorecard_*.json"))
        assert len(saved_files) == 1

        with open(saved_files[0], "r", encoding="utf-8") as f:
            saved_data = json.load(f)
            assert saved_data["total_questions"] == 1
            assert "summary" in saved_data
            assert len(saved_data["item_results"]) == 1

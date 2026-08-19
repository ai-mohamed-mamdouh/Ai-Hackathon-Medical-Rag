"""Shared fixtures for medical RAG orchestration tests."""

from __future__ import annotations

import pytest

from src.generation.orchestration.medical_rag.schemas import RetrieveResponse
from src.retrieval.query.query import Query
from tests.helpers import make_document


@pytest.fixture
def single_retrieval_response() -> RetrieveResponse:
    """Return one query aligned with one document group."""
    query = Query(
        original_query="What causes peripheral vertigo?",
        normalized_query="What causes peripheral vertigo?",
    )
    document = make_document(
        "file-1",
        "chunk-1",
        file_name="medical.pdf",
        page_number=12,
        rerank_score=0.9231,
        section="Clinical Evaluation",
    )
    return RetrieveResponse(
        queries=[query],
        documents=[[document]],
    )

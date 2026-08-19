"""Tests for the existing generation prompt builder integration."""

from __future__ import annotations

from langchain_core.documents import Document

from src.generation.generation.core.prompt_builder import build_prompt_template
from src.retrieval.query.query import Query
from tests.helpers import make_document


def test_single_query_prompt_contains_source_identifiers() -> None:
    """A single-query prompt exposes exact file and chunk identifiers."""
    query = Query(
        original_query="What causes vertigo?",
        normalized_query="What causes vertigo?",
    )
    document = make_document(
        "file-1",
        "chunk-1",
        file_name="medical.pdf",
        page_number=12,
        rerank_score=0.9231,
        section="Clinical Evaluation",
        internal_only="do-not-expose",
    )

    prompt = build_prompt_template([query], [[document]])

    assert "ORIGINAL QUERY:\nWhat causes vertigo?" in prompt
    assert "CONTEXT:" in prompt
    assert "File ID: file-1" in prompt
    assert "Chunk ID: chunk-1" in prompt
    assert "File: medical.pdf" in prompt
    assert "Page: 12" in prompt
    assert "Relevance Score: 0.9231" in prompt
    assert "Section: Clinical Evaluation" in prompt
    assert "internal_only" not in prompt
    assert "do-not-expose" not in prompt
    assert document.page_content in prompt


def test_decomposed_prompt_preserves_query_group_relationship() -> None:
    """Each decomposed query is followed by only its own document group."""
    queries = [
        Query(
            original_query="Compare two treatments.",
            normalized_query="What are the benefits of treatment A?",
        ),
        Query(
            original_query="Compare two treatments.",
            normalized_query="What are the risks of treatment B?",
        ),
    ]
    documents = [
        [make_document("file-a", "chunk-a", content="Treatment A evidence")],
        [make_document("file-b", "chunk-b", content="Treatment B evidence")],
    ]

    prompt = build_prompt_template(queries, documents)

    first_question = prompt.index("QUESTION 1:")
    first_content = prompt.index("Treatment A evidence")
    second_question = prompt.index("QUESTION 2:")
    second_content = prompt.index("Treatment B evidence")
    assert first_question < first_content < second_question < second_content
    assert "File ID: file-a" in prompt
    assert "Chunk ID: chunk-b" in prompt


def test_empty_document_group_is_rendered_without_failure() -> None:
    """An empty aligned group remains explicit in the prompt."""
    query = Query(
        original_query="Medical question",
        normalized_query="Medical question",
    )

    prompt = build_prompt_template([query], [[]])

    assert "No relevant documents found." in prompt


def test_missing_optional_metadata_does_not_crash() -> None:
    """Missing human-readable metadata fields use safe defaults."""
    query = Query(
        original_query="Medical question",
        normalized_query="Medical question",
    )
    document = Document(
        page_content="Content",
        metadata={"file_id": "file-1", "chunk_id": "chunk-1"},
    )

    prompt = build_prompt_template([query], [[document]])

    assert "File: unknown" in prompt
    assert "Page: unknown" in prompt
    assert "Relevance Score: unknown" in prompt


def test_invalid_rerank_score_does_not_crash() -> None:
    """Unexpected score types are rendered safely instead of formatted blindly."""
    query = Query(
        original_query="Medical question",
        normalized_query="Medical question",
    )
    document = make_document(
        "file-1",
        "chunk-1",
        rerank_score="not-a-number",
    )

    prompt = build_prompt_template([query], [[document]])

    assert "Relevance Score: not-a-number" in prompt


def test_prompt_builder_rejects_group_count_mismatch() -> None:
    """The prompt builder keeps its existing cardinality validation."""
    query = Query(
        original_query="Medical question",
        normalized_query="Medical question",
    )

    try:
        build_prompt_template([query], [[], []])
    except ValueError as exc:
        assert str(exc) == (
            "Number of queries must match number of document groups."
        )
    else:
        raise AssertionError("Expected query/document-group mismatch to fail.")

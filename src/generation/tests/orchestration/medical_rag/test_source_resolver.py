"""Tests for deterministic source metadata resolution."""

from __future__ import annotations

from copy import deepcopy

from langchain_core.documents import Document

from src.generation.orchestration.medical_rag.nodes import MedicalRAGNodes
from src.generation.orchestration.medical_rag.schemas import (
    GenerationOutput,
    RetrieveResponse,
    UsedChunk,
)
from src.generation.orchestration.medical_rag.source_resolver import (
    resolve_source_metadata,
)
from src.retrieval.query.query import Query
from tests.helpers import StubAsyncChain, StubRetrievalClient, make_document


def test_valid_pair_returns_complete_metadata_copy() -> None:
    """Only retrieved metadata is authoritative for the final source."""
    document = make_document(
        "file-1",
        "chunk-1",
        file_name="medical.pdf",
        page_number=7,
        custom_field="complete metadata",
    )
    original_metadata = deepcopy(document.metadata)

    result = resolve_source_metadata(
        [UsedChunk(file_id="file-1", chunk_id="chunk-1")],
        [[document]],
    )

    assert result.sources == [original_metadata]
    assert result.sources[0] is not document.metadata
    assert document.metadata == original_metadata
    assert "page_content" not in result.sources[0]


def test_page_content_is_returned_only_when_already_in_metadata() -> None:
    """The resolver never copies Document.page_content into source metadata."""
    document = Document(
        page_content="Document body",
        metadata={
            "file_id": "file-1",
            "chunk_id": "chunk-1",
            "page_content": "Metadata-owned content",
        },
    )

    result = resolve_source_metadata(
        [UsedChunk(file_id="file-1", chunk_id="chunk-1")],
        [[document]],
    )

    assert result.sources[0]["page_content"] == "Metadata-owned content"


def test_invalid_pair_is_ignored_and_warned() -> None:
    """Hallucinated identifiers do not fail the complete answer."""
    document = make_document("file-1", "chunk-1")

    result = resolve_source_metadata(
        [UsedChunk(file_id="file-1", chunk_id="missing")],
        [[document]],
    )

    assert result.sources == []
    assert len(result.warnings) == 1
    assert "did not match" in result.warnings[0]


def test_generated_references_are_deduplicated_in_original_order() -> None:
    """Repeated model references appear once while first-seen order is kept."""
    first = make_document("file-1", "chunk-1", order="first")
    second = make_document("file-2", "chunk-2", order="second")

    result = resolve_source_metadata(
        [
            UsedChunk(file_id="file-2", chunk_id="chunk-2"),
            UsedChunk(file_id="file-2", chunk_id="chunk-2"),
            UsedChunk(file_id="file-1", chunk_id="chunk-1"),
        ],
        [[first, second]],
    )

    assert [source["order"] for source in result.sources] == [
        "second",
        "first",
    ]


def test_duplicate_retrieved_pairs_use_first_match_deterministically() -> None:
    """The first retrieved document remains authoritative for duplicate IDs."""
    first = make_document("file-1", "chunk-1", version="first")
    duplicate = make_document("file-1", "chunk-1", version="second")

    result = resolve_source_metadata(
        [UsedChunk(file_id="file-1", chunk_id="chunk-1")],
        [[first], [duplicate]],
    )

    assert result.sources[0]["version"] == "first"
    assert any("Duplicate retrieved source pair" in item for item in result.warnings)


def test_source_node_does_not_call_any_llm_chain() -> None:
    """Source resolution is a pure Python operation."""
    rewrite_chain = StubAsyncChain(error=AssertionError("must not be called"))
    router_chain = StubAsyncChain(error=AssertionError("must not be called"))
    generation_chain = StubAsyncChain(error=AssertionError("must not be called"))
    nodes = MedicalRAGNodes(
        rewrite_chain=rewrite_chain,
        router_chain=router_chain,
        generation_chain=generation_chain,
        retrieval_client=StubRetrievalClient(),
    )
    query = Query(
        original_query="Question",
        normalized_query="Question",
    )
    document = make_document("file-1", "chunk-1")
    state = {
        "generation_output": GenerationOutput(
            answer="Answer",
            used_chunks=[UsedChunk(file_id="file-1", chunk_id="chunk-1")],
        ),
        "retrieval_response": RetrieveResponse(
            queries=[query],
            documents=[[document]],
        ),
    }

    update = nodes.source_metadata_node(state)

    assert update["sources"] == [document.metadata]
    assert rewrite_chain.calls == []
    assert router_chain.calls == []
    assert generation_chain.calls == []

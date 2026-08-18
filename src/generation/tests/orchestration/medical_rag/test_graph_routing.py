"""End-to-end routing tests for the compiled medical RAG graph."""

from __future__ import annotations

import pytest

from src.generation.orchestration.medical_rag.graph import create_medical_rag_graph
from src.generation.orchestration.medical_rag.nodes import OUT_OF_DOMAIN_RESPONSE
from src.generation.orchestration.medical_rag.schemas import RetrieveResponse
from src.retrieval.query.query import Query
from tests.helpers import StubAsyncChain, StubRetrievalClient, make_document


@pytest.mark.asyncio
async def test_medical_query_without_decomposition(
    single_retrieval_response: RetrieveResponse,
) -> None:
    """A simple medical query follows the complete retrieval branch."""
    rewrite_chain = StubAsyncChain(
        result={"normalized_query": "What causes peripheral vertigo?"}
    )
    router_chain = StubAsyncChain(
        result={"is_medical": True, "decomposition": False}
    )
    generation_chain = StubAsyncChain(
        result={
            "answer": "Common causes include benign peripheral vestibular disorders.",
            "used_chunks": [
                {"file_id": "file-1", "chunk_id": "chunk-1"}
            ],
        }
    )
    retrieval_client = StubRetrievalClient(result=single_retrieval_response)
    graph = create_medical_rag_graph(
        rewrite_chain,
        router_chain,
        generation_chain,
        retrieval_client,
    )

    result = await graph.ainvoke(
        {"original_query": "What causes peripheral vertigo?"}
    )

    assert result["final_response"].answer.startswith("Common causes")
    assert result["final_response"].sources[0]["file_id"] == "file-1"
    assert set(result["final_response"].model_dump()) == {"answer", "sources"}
    assert len(retrieval_client.calls) == 1
    assert retrieval_client.calls[0][1] is False
    assert len(generation_chain.calls) == 1
    assert result["query"].original_query == (
        "What causes peripheral vertigo?"
    )


@pytest.mark.asyncio
async def test_medical_query_with_decomposition() -> None:
    """The graph forwards decomposition to retrieval without adding a node."""
    original_query = "Compare the benefits and risks of two treatments."
    retrieval_response = RetrieveResponse(
        queries=[
            Query(
                original_query=original_query,
                normalized_query="What are the benefits of treatment A?",
            ),
            Query(
                original_query=original_query,
                normalized_query="What are the risks of treatment B?",
            ),
        ],
        documents=[
            [make_document("file-a", "chunk-a", content="Benefits evidence")],
            [make_document("file-b", "chunk-b", content="Risks evidence")],
        ],
    )
    retrieval_client = StubRetrievalClient(result=retrieval_response)
    generation_chain = StubAsyncChain(
        result={
            "answer": "Treatment A has benefits, while treatment B has risks.",
            "used_chunks": [
                {"file_id": "file-a", "chunk_id": "chunk-a"},
                {"file_id": "file-b", "chunk_id": "chunk-b"},
            ],
        }
    )
    graph = create_medical_rag_graph(
        StubAsyncChain(result={"normalized_query": original_query}),
        StubAsyncChain(result={"is_medical": True, "decomposition": True}),
        generation_chain,
        retrieval_client,
    )

    result = await graph.ainvoke({"original_query": original_query})

    assert retrieval_client.calls[0][1] is True
    assert "QUESTION 1:" in generation_chain.calls[0]["generation_prompt"]
    assert "QUESTION 2:" in generation_chain.calls[0]["generation_prompt"]
    assert [
        source["file_id"] for source in result["final_response"].sources
    ] == ["file-a", "file-b"]


@pytest.mark.asyncio
async def test_non_medical_query_skips_retrieval_and_generation() -> None:
    """Out-of-domain requests terminate through the fixed response branch."""
    retrieval_client = StubRetrievalClient(
        error=AssertionError("retrieval must not be called")
    )
    generation_chain = StubAsyncChain(
        error=AssertionError("generation must not be called")
    )
    graph = create_medical_rag_graph(
        StubAsyncChain(result={"normalized_query": "How do I sort a list?"}),
        StubAsyncChain(
            result={"is_medical": False, "decomposition": False}
        ),
        generation_chain,
        retrieval_client,
    )

    result = await graph.ainvoke(
        {"original_query": "How do I sort a Python list?"}
    )

    assert result["final_response"].answer == OUT_OF_DOMAIN_RESPONSE
    assert result["final_response"].sources == []
    assert retrieval_client.calls == []
    assert generation_chain.calls == []
    assert "rewrite_output" in result
    assert "router_output" in result


@pytest.mark.asyncio
async def test_one_valid_and_one_invalid_generated_source_reference(
    single_retrieval_response: RetrieveResponse,
) -> None:
    """An invalid source warning does not discard the valid answer or source."""
    graph = create_medical_rag_graph(
        StubAsyncChain(
            result={"normalized_query": "What causes peripheral vertigo?"}
        ),
        StubAsyncChain(
            result={"is_medical": True, "decomposition": False}
        ),
        StubAsyncChain(
            result={
                "answer": "Answer grounded in one valid source.",
                "used_chunks": [
                    {"file_id": "file-1", "chunk_id": "chunk-1"},
                    {"file_id": "file-x", "chunk_id": "chunk-x"},
                ],
            }
        ),
        StubRetrievalClient(result=single_retrieval_response),
    )

    result = await graph.ainvoke(
        {"original_query": "What causes peripheral vertigo?"}
    )

    assert result["final_response"].answer == (
        "Answer grounded in one valid source."
    )
    assert len(result["final_response"].sources) == 1
    assert result["final_response"].sources[0]["chunk_id"] == "chunk-1"
    assert any("did not match" in warning for warning in result["warnings"])
    assert not result.get("error")

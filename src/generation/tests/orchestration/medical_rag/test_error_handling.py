"""Tests for graceful graph failures and safe external responses."""

from __future__ import annotations

import pytest

from src.generation.orchestration.medical_rag.graph import create_medical_rag_graph
from src.generation.orchestration.medical_rag.nodes import SAFE_ERROR_RESPONSE
from src.generation.orchestration.medical_rag.schemas import RetrieveResponse
from tests.helpers import StubAsyncChain, StubRetrievalClient


@pytest.mark.asyncio
async def test_rewrite_failure_routes_to_safe_error_response() -> None:
    """Structured-output failures terminate through the error handler."""
    router_chain = StubAsyncChain(
        error=AssertionError("router must not be called")
    )
    graph = create_medical_rag_graph(
        StubAsyncChain(result={"normalized_query": ""}),
        router_chain,
        StubAsyncChain(),
        StubRetrievalClient(),
    )

    result = await graph.ainvoke({"original_query": "Medical question"})

    assert result["error_node"] == "rewrite_node"
    assert "ValidationError" in result["error"]
    assert result["final_response"].answer == SAFE_ERROR_RESPONSE
    assert result["final_response"].sources == []
    assert "ValidationError" not in result["final_response"].answer
    assert router_chain.calls == []


@pytest.mark.asyncio
async def test_retrieval_failure_produces_safe_response() -> None:
    """Retrieval service failures retain internal details but not externally."""
    generation_chain = StubAsyncChain(
        error=AssertionError("generation must not be called")
    )
    graph = create_medical_rag_graph(
        StubAsyncChain(result={"normalized_query": "Medical question"}),
        StubAsyncChain(
            result={"is_medical": True, "decomposition": False}
        ),
        generation_chain,
        StubRetrievalClient(error=RuntimeError("private retrieval detail")),
    )

    result = await graph.ainvoke({"original_query": "Medical question"})

    assert result["error_node"] == "retrieve_node"
    assert "private retrieval detail" in result["error"]
    assert result["final_response"].answer == SAFE_ERROR_RESPONSE
    assert "private retrieval detail" not in result["final_response"].answer
    assert result["final_response"].sources == []
    assert generation_chain.calls == []


@pytest.mark.asyncio
async def test_generation_failure_produces_safe_response(
    single_retrieval_response: RetrieveResponse,
) -> None:
    """Generation failures terminate safely after successful retrieval."""
    graph = create_medical_rag_graph(
        StubAsyncChain(result={"normalized_query": "Medical question"}),
        StubAsyncChain(
            result={"is_medical": True, "decomposition": False}
        ),
        StubAsyncChain(error=RuntimeError("private generation detail")),
        StubRetrievalClient(result=single_retrieval_response),
    )

    result = await graph.ainvoke({"original_query": "Medical question"})

    assert result["error_node"] == "generation_node"
    assert "private generation detail" in result["error"]
    assert result["final_response"].answer == SAFE_ERROR_RESPONSE
    assert "private generation detail" not in result["final_response"].answer
    assert result["final_response"].sources == []

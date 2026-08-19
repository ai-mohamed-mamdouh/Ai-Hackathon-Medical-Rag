"""Tests for the query rewrite node."""

from __future__ import annotations

import pytest

from src.generation.orchestration.medical_rag.nodes import MedicalRAGNodes
from tests.helpers import StubAsyncChain, StubRetrievalClient


@pytest.mark.asyncio
async def test_rewrite_preserves_original_and_stores_normalized_query() -> None:
    """The exact original text must remain unchanged after rewriting."""
    original_query = "  What causes peripheral vertigo?  "
    rewrite_chain = StubAsyncChain(
        result={"normalized_query": "What causes peripheral vertigo?"}
    )
    nodes = MedicalRAGNodes(
        rewrite_chain=rewrite_chain,
        router_chain=StubAsyncChain(),
        generation_chain=StubAsyncChain(),
        retrieval_client=StubRetrievalClient(),
    )

    update = await nodes.rewrite_node({"original_query": original_query})

    assert update["rewrite_output"].normalized_query == (
        "What causes peripheral vertigo?"
    )
    assert update["query"].original_query == original_query
    assert update["query"].normalized_query == (
        "What causes peripheral vertigo?"
    )
    assert rewrite_chain.calls == [{"original_query": original_query}]


@pytest.mark.asyncio
async def test_rewrite_structured_output_failure_sets_internal_error() -> None:
    """Malformed structured output must be captured at the node boundary."""
    nodes = MedicalRAGNodes(
        rewrite_chain=StubAsyncChain(result={"normalized_query": ""}),
        router_chain=StubAsyncChain(),
        generation_chain=StubAsyncChain(),
        retrieval_client=StubRetrievalClient(),
    )

    update = await nodes.rewrite_node({"original_query": "Medical question"})

    assert update["error_node"] == "rewrite_node"
    assert "ValidationError" in update["error"]
    assert "query" not in update

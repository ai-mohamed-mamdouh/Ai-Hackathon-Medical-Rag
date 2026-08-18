"""Tests for medical-domain routing."""

from __future__ import annotations

import pytest

from src.generation.orchestration.medical_rag.nodes import MedicalRAGNodes
from src.generation.orchestration.medical_rag.routing import (
    OUT_OF_DOMAIN_NODE,
    RETRIEVE_NODE,
    route_after_router,
)
from src.retrieval.query.query import Query
from tests.helpers import StubAsyncChain, StubRetrievalClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("normalized_query", "router_result"),
    [
        (
            "What are common causes of vertigo?",
            {"is_medical": True, "decomposition": False},
        ),
        (
            "How can a hospital reduce emergency department wait times?",
            {"is_medical": True, "decomposition": True},
        ),
    ],
)
async def test_medical_and_healthcare_queries_route_to_retrieval(
    normalized_query: str,
    router_result: dict[str, bool],
) -> None:
    """Clinical and broader healthcare requests are routed to retrieval."""
    router_chain = StubAsyncChain(result=router_result)
    nodes = MedicalRAGNodes(
        rewrite_chain=StubAsyncChain(),
        router_chain=router_chain,
        generation_chain=StubAsyncChain(),
        retrieval_client=StubRetrievalClient(),
    )
    state = {
        "query": Query(
            original_query=normalized_query,
            normalized_query=normalized_query,
        )
    }

    update = await nodes.router_node(state)

    assert update["router_output"].is_medical is True
    assert route_after_router(update) == RETRIEVE_NODE
    assert router_chain.calls == [{"normalized_query": normalized_query}]


@pytest.mark.asyncio
async def test_non_medical_query_routes_out_of_domain() -> None:
    """A non-medical request must skip the retrieval branch."""
    nodes = MedicalRAGNodes(
        rewrite_chain=StubAsyncChain(),
        router_chain=StubAsyncChain(
            result={"is_medical": False, "decomposition": False}
        ),
        generation_chain=StubAsyncChain(),
        retrieval_client=StubRetrievalClient(),
    )
    query = Query(
        original_query="How do I sort a Python list?",
        normalized_query="How do I sort a Python list?",
    )

    update = await nodes.router_node({"query": query})

    assert update["router_output"].is_medical is False
    assert update["router_output"].decomposition is False
    assert route_after_router(update) == OUT_OF_DOMAIN_NODE


@pytest.mark.asyncio
async def test_non_medical_decomposition_true_is_rejected() -> None:
    """Invalid router output must be treated as a structured-output failure."""
    nodes = MedicalRAGNodes(
        rewrite_chain=StubAsyncChain(),
        router_chain=StubAsyncChain(
            result={"is_medical": False, "decomposition": True}
        ),
        generation_chain=StubAsyncChain(),
        retrieval_client=StubRetrievalClient(),
    )
    query = Query(
        original_query="Write a poem.",
        normalized_query="Write a poem.",
    )

    update = await nodes.router_node({"query": query})

    assert update["error_node"] == "router_node"
    assert "ValidationError" in update["error"]

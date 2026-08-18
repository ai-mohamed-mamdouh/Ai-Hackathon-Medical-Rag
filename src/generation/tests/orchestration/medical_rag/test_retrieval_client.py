"""Tests for the asynchronous retrieval-service client."""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.documents import Document

from src.generation.orchestration.medical_rag.retrieval_client import (
    RetrievalClient,
    RetrievalConnectionError,
    RetrievalHTTPStatusError,
    RetrievalInvalidJSONError,
    RetrievalResponseValidationError,
    RetrievalTimeoutError,
)
from src.retrieval.query.query import Query


def retrieval_payload() -> dict[str, object]:
    """Return a valid nested retrieval payload."""
    return {
        "documents": [
            [
                {
                    "metadata": {
                        "file_id": "file-1",
                        "chunk_id": "chunk-1",
                    },
                    "page_content": "First medical passage.",
                    "type": "Document",
                }
            ],
            [
                {
                    "metadata": {
                        "file_id": "file-2",
                        "chunk_id": "chunk-2",
                    },
                    "page_content": "Second medical passage.",
                    "type": "Document",
                }
            ],
        ],
        "queries": [
            {
                "original_query": "Original question",
                "normalized_query": "First subquestion",
            },
            {
                "original_query": "Original question",
                "normalized_query": "Second subquestion",
            },
        ],
    }


@pytest.mark.asyncio
async def test_retrieval_request_and_nested_response_normalization() -> None:
    """The client sends the required request and returns runtime objects."""
    captured_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json=retrieval_payload())

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = RetrievalClient(
            http_client=http_client,
            base_url="http://127.0.0.1:8000/",
            timeout=7.5,
        )
        query = Query(
            original_query="Original question",
            normalized_query="Normalized question",
        )

        response = await client.retrieve(query=query, decomposition=True)

    assert captured_request is not None
    assert captured_request.method == "POST"
    assert str(captured_request.url.copy_with(query=None)) == (
        "http://127.0.0.1:8000/retrieval/retrieve"
    )
    assert captured_request.url.params["decomposition"] == "true"
    assert json.loads(captured_request.content) == {
        "original_query": "Original question",
        "normalized_query": "Normalized question",
    }
    assert len(response.queries) == 2
    assert response.queries[0].normalized_query == "First subquestion"
    assert len(response.documents) == 2
    assert len(response.documents[0]) == 1
    assert isinstance(response.documents[0][0], Document)
    assert response.documents[1][0].metadata["chunk_id"] == "chunk-2"


@pytest.mark.asyncio
async def test_retrieval_sends_false_decomposition_parameter() -> None:
    """False decomposition is serialized as the lowercase query value."""
    seen_value: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_value
        seen_value = request.url.params["decomposition"]
        payload = retrieval_payload()
        payload["queries"] = [payload["queries"][0]]
        payload["documents"] = [payload["documents"][0]]
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = RetrievalClient(http_client, "http://service", 5.0)
        await client.retrieve(
            Query(
                original_query="Original question",
                normalized_query="Normalized question",
            ),
            decomposition=False,
        )

    assert seen_value == "false"


@pytest.mark.asyncio
async def test_retrieval_timeout_is_wrapped() -> None:
    """HTTPX timeout errors become meaningful retrieval exceptions."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = RetrievalClient(http_client, "http://service", 1.0)
        with pytest.raises(RetrievalTimeoutError):
            await client.retrieve(
                Query(
                    original_query="Question",
                    normalized_query="Question",
                ),
                decomposition=False,
            )


@pytest.mark.asyncio
async def test_retrieval_connection_failure_is_wrapped() -> None:
    """Connection failures become meaningful retrieval exceptions."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = RetrievalClient(http_client, "http://service", 1.0)
        with pytest.raises(RetrievalConnectionError):
            await client.retrieve(
                Query(
                    original_query="Question",
                    normalized_query="Question",
                ),
                decomposition=False,
            )


@pytest.mark.asyncio
async def test_non_success_status_is_wrapped() -> None:
    """Non-success responses are raised after response.raise_for_status()."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "unavailable"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = RetrievalClient(http_client, "http://service", 5.0)
        with pytest.raises(RetrievalHTTPStatusError) as exc_info:
            await client.retrieve(
                Query(
                    original_query="Question",
                    normalized_query="Question",
                ),
                decomposition=False,
            )

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_invalid_json_is_rejected() -> None:
    """Invalid JSON must not pass into graph state."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not-json",
            headers={"content-type": "application/json"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = RetrievalClient(http_client, "http://service", 5.0)
        with pytest.raises(RetrievalInvalidJSONError):
            await client.retrieve(
                Query(
                    original_query="Question",
                    normalized_query="Question",
                ),
                decomposition=False,
            )


@pytest.mark.asyncio
async def test_flat_document_list_is_not_silently_coerced() -> None:
    """A flattened document payload is invalid because grouping is required."""
    payload = retrieval_payload()
    payload["documents"] = [payload["documents"][0][0]]
    payload["queries"] = [payload["queries"][0]]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = RetrievalClient(http_client, "http://service", 5.0)
        with pytest.raises(RetrievalResponseValidationError):
            await client.retrieve(
                Query(
                    original_query="Question",
                    normalized_query="Question",
                ),
                decomposition=False,
            )


@pytest.mark.asyncio
async def test_query_document_group_count_mismatch_is_rejected() -> None:
    """Query and document-group cardinalities must remain aligned."""
    payload = retrieval_payload()
    payload["documents"] = [payload["documents"][0]]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = RetrievalClient(http_client, "http://service", 5.0)
        with pytest.raises(RetrievalResponseValidationError):
            await client.retrieve(
                Query(
                    original_query="Question",
                    normalized_query="Question",
                ),
                decomposition=True,
            )


@pytest.mark.asyncio
async def test_invalid_document_schema_is_rejected() -> None:
    """Each nested document must contain string content and mapping metadata."""
    payload = retrieval_payload()
    payload["documents"] = [[{"page_content": None, "metadata": []}]]
    payload["queries"] = [payload["queries"][0]]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        client = RetrievalClient(http_client, "http://service", 5.0)
        with pytest.raises(RetrievalResponseValidationError):
            await client.retrieve(
                Query(
                    original_query="Question",
                    normalized_query="Question",
                ),
                decomposition=False,
            )

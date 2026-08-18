"""Test doubles for medical RAG orchestration tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from src.generation.orchestration.medical_rag.schemas import RetrieveResponse
from src.retrieval.query.query import Query


class StubAsyncChain:
    """Minimal async chain double with call tracking."""

    def __init__(
        self,
        result: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, inputs: dict[str, Any]) -> Any:
        """Return the configured result or raise the configured error."""
        self.calls.append(inputs)
        if self.error is not None:
            raise self.error
        if isinstance(self.result, Callable):
            return self.result(inputs)
        return self.result


class StubRetrievalClient:
    """Minimal retrieval-client double with call tracking."""

    def __init__(
        self,
        result: RetrieveResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[Query, bool]] = []

    async def retrieve(
        self,
        query: Query,
        decomposition: bool,
    ) -> RetrieveResponse:
        """Return the configured retrieval result or raise an error."""
        self.calls.append((query, decomposition))
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise AssertionError("Stub retrieval result was not configured.")
        return self.result


def make_document(
    file_id: str,
    chunk_id: str,
    content: str = "Retrieved medical content.",
    **metadata: Any,
) -> Document:
    """Create a test document with required source identifiers."""
    return Document(
        page_content=content,
        metadata={
            "file_id": file_id,
            "chunk_id": chunk_id,
            **metadata,
        },
    )

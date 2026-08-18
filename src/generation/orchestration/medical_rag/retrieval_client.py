"""Asynchronous client for the external retrieval service."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import httpx
from langchain_core.documents import Document
from pydantic import ValidationError

from src.generation.orchestration.medical_rag.schemas import RetrieveResponse
from src.retrieval.query.query import Query

logger = logging.getLogger(__name__)


class RetrievalClientError(RuntimeError):
    """Base exception for retrieval-client failures."""


class RetrievalTimeoutError(RetrievalClientError):
    """Raised when the retrieval service times out."""


class RetrievalConnectionError(RetrievalClientError):
    """Raised when the retrieval service cannot be reached."""


class RetrievalHTTPStatusError(RetrievalClientError):
    """Raised when the retrieval service returns a non-success status."""

    def __init__(self, status_code: int) -> None:
        super().__init__(
            f"Retrieval service returned HTTP status {status_code}."
        )
        self.status_code = status_code


class RetrievalInvalidJSONError(RetrievalClientError):
    """Raised when the retrieval service response is not valid JSON."""


class RetrievalResponseValidationError(RetrievalClientError):
    """Raised when the retrieval response does not match the expected schema."""


class RetrievalClient:
    """Reusable asynchronous client for POST /retrieval/retrieve."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        timeout: float | httpx.Timeout,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string.")

        self._http_client = http_client
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def endpoint_url(self) -> str:
        """Return the complete retrieval endpoint URL."""
        return f"{self._base_url}/retrieval/retrieve"

    async def retrieve(
        self,
        query: Query,
        decomposition: bool,
    ) -> RetrieveResponse:
        """Retrieve and normalize grouped documents for a rewritten query."""
        payload = _query_to_payload(query)
        params = {"decomposition": str(bool(decomposition)).lower()}

        try:
            response = await self._http_client.post(
                self.endpoint_url,
                params=params,
                json=payload,
                timeout=self._timeout,
                headers={
                    "accept": "application/json",
                    "content-type": "application/json",
                },
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            # raise RetrievalTimeoutError(
            #     "Retrieval request timed out."
            # ) from None
            print("TIMEOUT TYPE:", type(exc).__name__)
            print("URL:", self.endpoint_url)
            print("TIMEOUT:", self._timeout)
            raise
        except httpx.HTTPStatusError as exc:
            raise RetrievalHTTPStatusError(exc.response.status_code) from None
        except httpx.RequestError as exc:
            raise RetrievalConnectionError(
                "Retrieval service connection failed."
            ) from None

        try:
            raw_response = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            raise RetrievalInvalidJSONError(
                "Retrieval service returned invalid JSON."
            ) from None

        normalized_response = _normalize_response(raw_response)
        total_documents = sum(
            len(group) for group in normalized_response.documents
        )
        logger.info(
            "Retrieval completed.",
            extra={
                "status_code": response.status_code,
                "query_count": len(normalized_response.queries),
                "document_group_count": len(normalized_response.documents),
                "document_count": total_documents,
            },
        )
        return normalized_response


def create_retrieval_client(
    http_client: httpx.AsyncClient,
) -> RetrievalClient:
    """Create a retrieval client from application settings."""
    from src.config.settings import settings

    return RetrievalClient(
        http_client=http_client,
        base_url=settings.RETRIEVAL_BASE_URL,
        timeout=float(settings.RETRIEVAL_TIMEOUT),
    )


def _query_to_payload(query: Query) -> dict[str, Any]:
    """Serialize the complete Query object while validating required fields."""
    if hasattr(query, "model_dump"):
        payload = query.model_dump(mode="json")
    elif hasattr(query, "dict") and callable(query.dict):
        payload = query.dict()
    elif is_dataclass(query):
        payload = asdict(query)
    else:
        payload = {
            "original_query": getattr(query, "original_query", None),
            "normalized_query": getattr(query, "normalized_query", None),
        }

    if not isinstance(payload, dict):
        raise RetrievalResponseValidationError(
            "Query could not be serialized as a JSON object."
        )

    original_query = payload.get("original_query")
    normalized_query = payload.get("normalized_query")
    if not isinstance(original_query, str) or not original_query.strip():
        raise RetrievalResponseValidationError(
            "Serialized query is missing a valid original_query."
        )
    if not isinstance(normalized_query, str) or not normalized_query.strip():
        raise RetrievalResponseValidationError(
            "Serialized query is missing a valid normalized_query."
        )

    return payload


def _normalize_response(raw_response: Any) -> RetrieveResponse:
    """Convert the HTTP payload into validated runtime Query and Document objects."""
    if not isinstance(raw_response, Mapping):
        raise RetrievalResponseValidationError(
            "Retrieval response must be a JSON object."
        )

    raw_queries = raw_response.get("queries")
    raw_document_groups = raw_response.get("documents")

    if not isinstance(raw_queries, list):
        raise RetrievalResponseValidationError(
            "Retrieval response queries must be a list."
        )
    if not isinstance(raw_document_groups, list):
        raise RetrievalResponseValidationError(
            "Retrieval response documents must be a nested list."
        )

    queries = [
        _normalize_query_item(item, index)
        for index, item in enumerate(raw_queries)
    ]

    document_groups: list[list[Document]] = []
    for group_index, raw_group in enumerate(raw_document_groups):
        if not isinstance(raw_group, list):
            raise RetrievalResponseValidationError(
                "Each retrieval document group must be a list."
            )
        document_groups.append(
            [
                _normalize_document_item(item, group_index, document_index)
                for document_index, item in enumerate(raw_group)
            ]
        )

    if len(queries) != len(document_groups):
        raise RetrievalResponseValidationError(
            "Number of queries must match number of document groups."
        )

    try:
        return RetrieveResponse(
            queries=queries,
            documents=document_groups,
        )
    except ValidationError:
        raise RetrievalResponseValidationError(
            "Retrieval response failed internal schema validation."
        ) from None


def _normalize_query_item(item: Any, index: int) -> Query:
    """Convert one retrieval query payload into the repository Query type."""
    if isinstance(item, Query):
        return item
    if not isinstance(item, Mapping):
        raise RetrievalResponseValidationError(
            f"Query item at index {index} must be an object."
        )

    try:
        return Query(**dict(item))
    except (TypeError, ValueError, ValidationError):
        raise RetrievalResponseValidationError(
            f"Query item at index {index} could not be converted."
        ) from None


def _normalize_document_item(
    item: Any,
    group_index: int,
    document_index: int,
) -> Document:
    """Convert one retrieval document payload into a LangChain Document."""
    if isinstance(item, Document):
        return item
    if not isinstance(item, Mapping):
        raise RetrievalResponseValidationError(
            "Document items must be objects inside nested document groups."
        )

    page_content = item.get("page_content")
    metadata = item.get("metadata", {})

    if not isinstance(page_content, str):
        raise RetrievalResponseValidationError(
            "Document page_content must be a string."
        )
    if not isinstance(metadata, Mapping):
        raise RetrievalResponseValidationError(
            "Document metadata must be an object."
        )

    try:
        return Document(
            page_content=page_content,
            metadata=dict(metadata),
        )
    except (TypeError, ValueError, ValidationError):
        raise RetrievalResponseValidationError(
            "Document at group "
            f"{group_index}, index {document_index} could not be converted."
        ) from None

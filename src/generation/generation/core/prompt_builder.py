"""Build the final generation prompt from grouped retrieval results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from langchain_core.documents import Document

from src.generation.generation.core.prompts import RAG_SYSTEM_PROMPT
from src.generation.retrieval.query.query import Query


def build_prompt_template(
    queries: list[Query],
    documents: list[list[Document]],
    system_prompt: str = RAG_SYSTEM_PROMPT,
) -> str:
    """Build a prompt while preserving query-to-document-group alignment."""
    if len(queries) != len(documents):
        raise ValueError("Number of queries must match number of document groups.")

    if not queries:
        raise ValueError("Queries list cannot be empty.")

    original_query = queries[0].original_query
    if not isinstance(original_query, str) or not original_query.strip():
        raise ValueError("The original query must be a non-empty string.")

    prompt_parts = [
        f"""SYSTEM PROMPT:
{system_prompt}

ORIGINAL QUERY:
{original_query}
"""
    ]

    if len(queries) == 1:
        prompt_parts.append("\nCONTEXT:\n")
        add_documents(prompt_parts=prompt_parts, documents=documents[0])
        return "\n".join(prompt_parts)

    for query_index, (query, query_docs) in enumerate(
        zip(queries, documents, strict=True),
        start=1,
    ):
        question = query.normalized_query or query.original_query
        prompt_parts.append(
            f"""
QUESTION {query_index}:
{question}

CONTEXT:
"""
        )
        add_documents(prompt_parts=prompt_parts, documents=query_docs)

    return "\n".join(prompt_parts)


def add_documents(
    prompt_parts: list[str],
    documents: list[Document],
) -> None:
    """Append selected document fields to a generation prompt."""
    if not documents:
        prompt_parts.append("No relevant documents found.")
        return

    for doc_index, document in enumerate(documents, start=1):
        metadata = _metadata_mapping(document.metadata)
        metadata_lines = [
            f"File ID: {_safe_text(metadata.get('file_id'))}",
            f"Chunk ID: {_safe_text(metadata.get('chunk_id'))}",
            f"File: {_safe_text(metadata.get('file_name'))}",
            f"Page: {_safe_text(metadata.get('page_number'))}",
            f"Relevance Score: {_format_rerank_score(metadata.get('rerank_score'))}",
        ]

        section = _optional_text(metadata.get("section"))
        if section is not None:
            metadata_lines.append(f"Section: {section}")

        page_content = document.page_content
        if not isinstance(page_content, str):
            page_content = str(page_content)

        prompt_parts.append(
            f"""
[Document {doc_index}]
{chr(10).join(metadata_lines)}

Content:
{page_content}
"""
        )


def _metadata_mapping(value: Any) -> Mapping[str, Any]:
    """Return metadata as a mapping without exposing unexpected objects."""
    if isinstance(value, Mapping):
        return value
    return {}


def _safe_text(value: Any, default: str = "unknown") -> str:
    """Convert a metadata value to readable text without raising."""
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _optional_text(value: Any) -> str | None:
    """Return optional metadata text when it is meaningful."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_rerank_score(value: Any) -> str:
    """Format numeric scores while tolerating missing or unexpected values."""
    if value is None:
        return "unknown"

    try:
        score = float(value)
    except (TypeError, ValueError):
        return _safe_text(value)

    if not math.isfinite(score):
        return _safe_text(value)

    return f"{score:.4f}"

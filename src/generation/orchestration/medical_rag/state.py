"""Typed LangGraph state for the medical RAG workflow."""

from __future__ import annotations

from typing import Any, TypedDict

from src.generation.orchestration.medical_rag.schemas import (
    FinalResponse,
    GenerationOutput,
    QueryNormalizationOutput,
    RetrieveResponse,
    RouterOutput,
)
from src.retrieval.query.query import Query


class MedicalRAGState(TypedDict, total=False):
    """Complete internal state with partial node-update semantics."""

    original_query: str

    rewrite_output: QueryNormalizationOutput
    query: Query

    router_output: RouterOutput

    retrieval_response: RetrieveResponse

    generation_prompt: str

    generation_output: GenerationOutput
    answer: str

    sources: list[dict[str, Any]]
    warnings: list[str]

    final_response: FinalResponse

    error: str | None
    error_node: str | None

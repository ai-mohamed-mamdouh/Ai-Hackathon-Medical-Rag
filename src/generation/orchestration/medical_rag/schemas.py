"""Pydantic models used by the medical RAG orchestration layer."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.retrieval.query.query import Query


class QueryNormalizationOutput(BaseModel):
    """Structured query-rewrite output."""

    model_config = ConfigDict(extra="forbid")

    normalized_query: str

    @field_validator("normalized_query")
    @classmethod
    def validate_normalized_query(cls, value: str) -> str:
        """Require a meaningful normalized query."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("normalized_query must be a non-empty string.")
        return value.strip()


class RouterOutput(BaseModel):
    """Structured medical-domain and decomposition routing output."""

    model_config = ConfigDict(extra="forbid")

    is_medical: bool
    decomposition: bool

    @model_validator(mode="after")
    def validate_non_medical_decomposition(self) -> "RouterOutput":
        """Disallow decomposition for out-of-domain requests."""
        if not self.is_medical and self.decomposition:
            raise ValueError(
                "decomposition must be false when is_medical is false."
            )
        return self


class UsedChunk(BaseModel):
    """A retrieved chunk identifier used by the generation model."""

    model_config = ConfigDict(extra="forbid")

    file_id: str
    chunk_id: str

    @field_validator("file_id", "chunk_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        """Require a non-empty source identifier."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Source identifiers must be non-empty strings.")
        return value.strip()


class GenerationOutput(BaseModel):
    """Structured final-answer output from the generation model."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    used_chunks: list[UsedChunk] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        """Require a non-empty generated answer."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("answer must be a non-empty string.")
        return value.strip()


class FinalResponse(BaseModel):
    """External API-facing response."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def validate_final_answer(cls, value: str) -> str:
        """Require the final response to contain an answer."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("FinalResponse.answer must be non-empty.")
        return value


class RetrieveResponse(BaseModel):
    """Validated internal retrieval response with grouped documents."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    documents: list[list[Document]] = Field(default_factory=list)
    queries: list[Query] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_group_alignment(self) -> "RetrieveResponse":
        """Require at least one query and one aligned document group per query."""
        if not self.queries:
            raise ValueError("Retrieval response queries cannot be empty.")
        if len(self.queries) != len(self.documents):
            raise ValueError(
                "Number of queries must match number of document groups."
            )
        return self

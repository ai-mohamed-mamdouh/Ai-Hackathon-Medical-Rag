"""LLM chain factories for the medical RAG workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_groq import ChatGroq

from src.config.settings import settings
from src.generation.generation.core.prompts import (
    REWRITE_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)
from src.generation.orchestration.medical_rag.schemas import (
    GenerationOutput,
    QueryNormalizationOutput,
    RouterOutput,
)


@dataclass(frozen=True, slots=True)
class MedicalRAGChains:
    """Dependency-injectable LLM chains used by graph nodes."""

    chain_rewrite: Runnable[Any, QueryNormalizationOutput]
    chain_router: Runnable[Any, RouterOutput]
    chain_generation: Runnable[Any, GenerationOutput]


def create_medical_rag_chains() -> MedicalRAGChains:
    """Create deterministic Groq-backed structured-output chains once."""
    _validate_model_settings()

    small_model = ChatGroq(
        model=settings.SMALL_GROQ_MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=0,
    )
    generation_model = ChatGroq(
        model=settings.GROQ_MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=0,
    )

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REWRITE_SYSTEM_PROMPT),
            ("human", "{original_query}"),
        ]
    )
    router_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", "{normalized_query}"),
        ]
    )

    chain_rewrite = (
        rewrite_prompt
        | small_model.with_structured_output(QueryNormalizationOutput , method="json_schema")
    ).with_config({"run_name": "medical_rag_query_rewrite"})

    chain_router = (
        router_prompt
        | small_model.with_structured_output(RouterOutput , method="json_schema")
    ).with_config({"run_name": "medical_rag_router"})

    chain_generation = (
        RunnableLambda(_extract_generation_prompt)
        | generation_model.with_structured_output(GenerationOutput , method="json_schema")
    ).with_config({"run_name": "medical_rag_generation"})

    return MedicalRAGChains(
        chain_rewrite=chain_rewrite,
        chain_router=chain_router,
        chain_generation=chain_generation,
    )


def _extract_generation_prompt(inputs: Mapping[str, Any]) -> str:
    """Extract and validate the already-built generation prompt."""
    generation_prompt = inputs.get("generation_prompt")
    if not isinstance(generation_prompt, str) or not generation_prompt.strip():
        raise ValueError("generation_prompt must be a non-empty string.")
    return generation_prompt


def _validate_model_settings() -> None:
    """Fail early when required model configuration is missing."""
    if not str(settings.GROQ_MODEL_NAME).strip():
        raise ValueError("settings.GROQ_MODEL_NAME must be configured.")
    if not str(settings.SMALL_GROQ_MODEL_NAME).strip():
        raise ValueError("settings.SMALL_GROQ_MODEL_NAME must be configured.")

    api_key = settings.GROQ_API_KEY.get_secret_value()
    if not api_key.strip():
        raise ValueError("settings.GROQ_API_KEY must be configured.")

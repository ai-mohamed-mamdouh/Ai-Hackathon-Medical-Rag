"""Focused graph nodes for the medical RAG orchestration workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import Runnable

from src.generation.generation.core.prompt_builder import build_prompt_template
from src.generation.orchestration.medical_rag.retrieval_client import RetrievalClient
from src.generation.orchestration.medical_rag.schemas import (
    FinalResponse,
    GenerationOutput,
    QueryNormalizationOutput,
    RetrieveResponse,
    RouterOutput,
)
from src.generation.orchestration.medical_rag.source_resolver import (
    resolve_source_metadata,
)
from src.generation.orchestration.medical_rag.state import MedicalRAGState
from src.retrieval.query.query import Query

logger = logging.getLogger(__name__)

OUT_OF_DOMAIN_RESPONSE = (
    "I can only answer questions related to the medical domain."
)
SAFE_ERROR_RESPONSE = (
    "An error occurred while processing your request. Please try again."
)


class MissingStateValueError(ValueError):
    """Raised when a graph node cannot find a required state value."""


@dataclass(slots=True)
class MedicalRAGNodes:
    """Dependency-injected node implementations used by the compiled graph."""

    rewrite_chain: Runnable[Any, Any]
    router_chain: Runnable[Any, Any]
    generation_chain: Runnable[Any, Any]
    retrieval_client: RetrievalClient

    async def rewrite_node(
        self,
        state: MedicalRAGState,
    ) -> dict[str, Any]:
        """Normalize the original query while preserving its exact text."""
        node_name = "rewrite_node"
        try:
            original_query = _require_non_empty_string(
                state.get("original_query"),
                "original_query",
            )
            raw_output = await self.rewrite_chain.ainvoke(
                {"original_query": original_query}
            )
            rewrite_output = QueryNormalizationOutput.model_validate(raw_output)
            query = Query(
                original_query=original_query,
                normalized_query=rewrite_output.normalized_query,
            )
            logger.debug("Medical RAG rewrite node completed.")
            return {
                "rewrite_output": rewrite_output,
                "query": query,
            }
        # except Exception as exc:
        #     return _node_error_update(node_name, exc)
        except Exception:
            logger.exception("Medical RAG node failed.")
            raise

    async def router_node(
        self,
        state: MedicalRAGState,
    ) -> dict[str, Any]:
        """Classify medical scope and decide retrieval decomposition."""
        node_name = "router_node"
        try:
            query = _require_query(state.get("query"))
            normalized_query = _require_non_empty_string(
                query.normalized_query,
                "query.normalized_query",
            )
            raw_output = await self.router_chain.ainvoke(
                {"normalized_query": normalized_query}
            )
            router_output = RouterOutput.model_validate(raw_output)
            logger.debug("Medical RAG router node completed.")
            return {"router_output": router_output}
        except Exception as exc:
            return _node_error_update(node_name, exc)

    def out_of_domain_node(
        self,
        state: MedicalRAGState,
    ) -> dict[str, Any]:
        """Return the fixed response for non-medical requests."""
        del state
        return {
            "answer": OUT_OF_DOMAIN_RESPONSE,
            "sources": [],
        }

    async def retrieve_node(
        self,
        state: MedicalRAGState,
    ) -> dict[str, Any]:
        """Call the external retrieval service asynchronously."""
        node_name = "retrieve_node"
        try:
            query = _require_query(state.get("query"))
            router_output = _require_router_output(state.get("router_output"))
            if not router_output.is_medical:
                raise MissingStateValueError(
                    "retrieve_node cannot run for an out-of-domain query."
                )

            retrieval_response = await self.retrieval_client.retrieve(
                query=query,
                decomposition=router_output.decomposition,
            )
            validated_response = RetrieveResponse.model_validate(
                retrieval_response
            )
            logger.debug("Medical RAG retrieval node completed.")
            return {"retrieval_response": validated_response}
        except Exception as exc:
            return _node_error_update(node_name, exc)

    def build_prompt_node(
        self,
        state: MedicalRAGState,
    ) -> dict[str, Any]:
        """Build the generation prompt from the validated retrieval response."""
        node_name = "build_prompt_node"
        try:
            retrieval_response = _require_retrieval_response(
                state.get("retrieval_response")
            )
            generation_prompt = build_prompt_template(
                queries=retrieval_response.queries,
                documents=retrieval_response.documents,
            )
            _require_non_empty_string(
                generation_prompt,
                "generation_prompt",
            )
            logger.debug("Medical RAG prompt-building node completed.")
            return {"generation_prompt": generation_prompt}
        except Exception as exc:
            return _node_error_update(node_name, exc)

    async def generation_node(
        self,
        state: MedicalRAGState,
    ) -> dict[str, Any]:
        """Generate a structured answer and used source identifiers."""
        node_name = "generation_node"
        try:
            generation_prompt = _require_non_empty_string(
                state.get("generation_prompt"),
                "generation_prompt",
            )
            raw_output = await self.generation_chain.ainvoke(
                {"generation_prompt": generation_prompt}
            )
            generation_output = GenerationOutput.model_validate(raw_output)
            logger.debug("Medical RAG generation node completed.")
            return {
                "generation_output": generation_output,
                "answer": generation_output.answer,
            }
        except Exception as exc:
            return _node_error_update(node_name, exc)

    def source_metadata_node(
        self,
        state: MedicalRAGState,
    ) -> dict[str, Any]:
        """Resolve source metadata deterministically without calling an LLM."""
        node_name = "source_metadata_node"
        try:
            generation_output = _require_generation_output(
                state.get("generation_output")
            )
            retrieval_response = _require_retrieval_response(
                state.get("retrieval_response")
            )
            resolution = resolve_source_metadata(
                used_chunks=generation_output.used_chunks,
                document_groups=retrieval_response.documents,
            )
            prior_warnings = list(state.get("warnings", []))
            return {
                "sources": resolution.sources,
                "warnings": prior_warnings + resolution.warnings,
            }
        except Exception as exc:
            return _node_error_update(node_name, exc)

    def finalize_node(
        self,
        state: MedicalRAGState,
    ) -> dict[str, Any]:
        """Create the external API-facing FinalResponse."""
        node_name = "finalize_node"
        try:
            answer = _require_non_empty_string(
                state.get("answer"),
                "answer",
            )
            sources = state.get("sources", [])
            if not isinstance(sources, list):
                raise MissingStateValueError("sources must be a list.")

            final_response = FinalResponse(
                answer=answer,
                sources=sources,
            )
            return {"final_response": final_response}
        except Exception as exc:
            update = _node_error_update(node_name, exc)
            update.update(_safe_error_fields())
            return update

    def error_handler_node(
        self,
        state: MedicalRAGState,
    ) -> dict[str, Any]:
        """Return a safe response while preserving internal error fields."""
        logger.warning(
            "Medical RAG workflow entered the error handler.",
            extra={"error_node": state.get("error_node")},
        )
        return _safe_error_fields()


def _safe_error_fields() -> dict[str, Any]:
    """Build the fixed safe error response fields."""
    final_response = FinalResponse(
        answer=SAFE_ERROR_RESPONSE,
        sources=[],
    )
    return {
        "answer": SAFE_ERROR_RESPONSE,
        "sources": [],
        "final_response": final_response,
    }


def _node_error_update(
    node_name: str,
    exc: Exception,
) -> dict[str, Any]:
    """Log a sanitized stack trace and preserve details in internal state."""
    sanitized_exception = RuntimeError(
        f"{type(exc).__name__} raised in {node_name}."
    )
    logger.error(
        "Medical RAG node failed.",
        extra={
            "node": node_name,
            "error_type": type(exc).__name__,
        },
        exc_info=(
            type(sanitized_exception),
            sanitized_exception,
            exc.__traceback__,
        ),
    )
    return {
        "error": f"{type(exc).__name__}: {exc}",
        "error_node": node_name,
    }


def _require_non_empty_string(value: Any, field_name: str) -> str:
    """Return a required string without changing its original contents."""
    if not isinstance(value, str) or not value.strip():
        raise MissingStateValueError(
            f"Required state field '{field_name}' must be a non-empty string."
        )
    return value


def _require_query(value: Any) -> Query:
    """Return a required Query instance."""
    if not isinstance(value, Query):
        raise MissingStateValueError(
            "Required state field 'query' must contain a Query instance."
        )
    return value


def _require_router_output(value: Any) -> RouterOutput:
    """Return a validated RouterOutput instance."""
    try:
        return RouterOutput.model_validate(value)
    except Exception:
        raise MissingStateValueError(
            "Required state field 'router_output' is missing or invalid."
        ) from None


def _require_retrieval_response(value: Any) -> RetrieveResponse:
    """Return a validated RetrieveResponse instance."""
    try:
        return RetrieveResponse.model_validate(value)
    except Exception:
        raise MissingStateValueError(
            "Required state field 'retrieval_response' is missing or invalid."
        ) from None


def _require_generation_output(value: Any) -> GenerationOutput:
    """Return a validated GenerationOutput instance."""
    try:
        return GenerationOutput.model_validate(value)
    except Exception:
        raise MissingStateValueError(
            "Required state field 'generation_output' is missing or invalid."
        ) from None

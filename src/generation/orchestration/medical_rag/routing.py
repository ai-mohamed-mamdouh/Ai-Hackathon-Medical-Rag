"""Explicit conditional routing functions for the medical RAG graph."""

from __future__ import annotations

import logging
from typing import Literal

from src.generation.orchestration.medical_rag.schemas import RouterOutput
from src.generation.orchestration.medical_rag.state import MedicalRAGState

logger = logging.getLogger(__name__)

REWRITE_NODE = "rewrite_node"
ROUTER_NODE = "router_node"
OUT_OF_DOMAIN_NODE = "out_of_domain_node"
RETRIEVE_NODE = "retrieve_node"
BUILD_PROMPT_NODE = "build_prompt_node"
GENERATION_NODE = "generation_node"
SOURCE_METADATA_NODE = "source_metadata_node"
FINALIZE_NODE = "finalize_node"
ERROR_HANDLER_NODE = "error_handler_node"


def route_after_rewrite(
    state: MedicalRAGState,
) -> Literal["router_node", "error_handler_node"]:
    """Route rewrite success to the router and failures to the error handler."""
    if state.get("error"):
        return ERROR_HANDLER_NODE
    return ROUTER_NODE


def route_after_router(
    state: MedicalRAGState,
) -> Literal[
    "out_of_domain_node",
    "retrieve_node",
    "error_handler_node",
]:
    """Route by error state and medical-domain classification."""
    if state.get("error"):
        return ERROR_HANDLER_NODE

    router_output = state.get("router_output")
    if not isinstance(router_output, RouterOutput):
        logger.error("Router output was missing during conditional routing.")
        return ERROR_HANDLER_NODE

    if router_output.is_medical:
        return RETRIEVE_NODE
    return OUT_OF_DOMAIN_NODE


def route_after_retrieve(
    state: MedicalRAGState,
) -> Literal["build_prompt_node", "error_handler_node"]:
    """Route retrieval success to prompt construction."""
    if state.get("error"):
        return ERROR_HANDLER_NODE
    return BUILD_PROMPT_NODE


def route_after_prompt_build(
    state: MedicalRAGState,
) -> Literal["generation_node", "error_handler_node"]:
    """Route prompt-building success to generation."""
    if state.get("error"):
        return ERROR_HANDLER_NODE
    return GENERATION_NODE


def route_after_generation(
    state: MedicalRAGState,
) -> Literal["source_metadata_node", "error_handler_node"]:
    """Route generation success to deterministic source resolution."""
    if state.get("error"):
        return ERROR_HANDLER_NODE
    return SOURCE_METADATA_NODE


def route_after_source_metadata(
    state: MedicalRAGState,
) -> Literal["finalize_node", "error_handler_node"]:
    """Route source-resolution success to finalization."""
    if state.get("error"):
        return ERROR_HANDLER_NODE
    return FINALIZE_NODE

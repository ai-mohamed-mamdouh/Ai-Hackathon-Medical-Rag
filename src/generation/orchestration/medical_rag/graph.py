"""LangGraph construction for the medical RAG orchestration workflow."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.generation.orchestration.medical_rag.nodes import MedicalRAGNodes
from src.generation.orchestration.medical_rag.retrieval_client import RetrievalClient
from src.generation.orchestration.medical_rag.routing import (
    BUILD_PROMPT_NODE,
    ERROR_HANDLER_NODE,
    FINALIZE_NODE,
    GENERATION_NODE,
    OUT_OF_DOMAIN_NODE,
    RETRIEVE_NODE,
    REWRITE_NODE,
    ROUTER_NODE,
    SOURCE_METADATA_NODE,
    route_after_generation,
    route_after_prompt_build,
    route_after_retrieve,
    route_after_rewrite,
    route_after_router,
    route_after_source_metadata,
)
from src.generation.orchestration.medical_rag.state import MedicalRAGState


def create_medical_rag_graph(
    rewrite_chain: Runnable[Any, Any],
    router_chain: Runnable[Any, Any],
    generation_chain: Runnable[Any, Any],
    retrieval_client: RetrievalClient,
) -> CompiledStateGraph:
    """Compile the medical RAG graph with dependency-injected services."""
    nodes = MedicalRAGNodes(
        rewrite_chain=rewrite_chain,
        router_chain=router_chain,
        generation_chain=generation_chain,
        retrieval_client=retrieval_client,
    )

    graph_builder = StateGraph(MedicalRAGState)

    graph_builder.add_node(REWRITE_NODE, nodes.rewrite_node)
    graph_builder.add_node(ROUTER_NODE, nodes.router_node)
    graph_builder.add_node(OUT_OF_DOMAIN_NODE, nodes.out_of_domain_node)
    graph_builder.add_node(RETRIEVE_NODE, nodes.retrieve_node)
    graph_builder.add_node(BUILD_PROMPT_NODE, nodes.build_prompt_node)
    graph_builder.add_node(GENERATION_NODE, nodes.generation_node)
    graph_builder.add_node(SOURCE_METADATA_NODE, nodes.source_metadata_node)
    graph_builder.add_node(FINALIZE_NODE, nodes.finalize_node)
    graph_builder.add_node(ERROR_HANDLER_NODE, nodes.error_handler_node)

    graph_builder.add_edge(START, REWRITE_NODE)

    graph_builder.add_conditional_edges(
        REWRITE_NODE,
        route_after_rewrite,
        {
            ROUTER_NODE: ROUTER_NODE,
            ERROR_HANDLER_NODE: ERROR_HANDLER_NODE,
        },
    )
    graph_builder.add_conditional_edges(
        ROUTER_NODE,
        route_after_router,
        {
            OUT_OF_DOMAIN_NODE: OUT_OF_DOMAIN_NODE,
            RETRIEVE_NODE: RETRIEVE_NODE,
            ERROR_HANDLER_NODE: ERROR_HANDLER_NODE,
        },
    )
    graph_builder.add_conditional_edges(
        RETRIEVE_NODE,
        route_after_retrieve,
        {
            BUILD_PROMPT_NODE: BUILD_PROMPT_NODE,
            ERROR_HANDLER_NODE: ERROR_HANDLER_NODE,
        },
    )
    graph_builder.add_conditional_edges(
        BUILD_PROMPT_NODE,
        route_after_prompt_build,
        {
            GENERATION_NODE: GENERATION_NODE,
            ERROR_HANDLER_NODE: ERROR_HANDLER_NODE,
        },
    )
    graph_builder.add_conditional_edges(
        GENERATION_NODE,
        route_after_generation,
        {
            SOURCE_METADATA_NODE: SOURCE_METADATA_NODE,
            ERROR_HANDLER_NODE: ERROR_HANDLER_NODE,
        },
    )
    graph_builder.add_conditional_edges(
        SOURCE_METADATA_NODE,
        route_after_source_metadata,
        {
            FINALIZE_NODE: FINALIZE_NODE,
            ERROR_HANDLER_NODE: ERROR_HANDLER_NODE,
        },
    )

    graph_builder.add_edge(OUT_OF_DOMAIN_NODE, FINALIZE_NODE)
    graph_builder.add_edge(FINALIZE_NODE, END)
    graph_builder.add_edge(ERROR_HANDLER_NODE, END)

    return graph_builder.compile()

"""Minimal asynchronous invocation example for the medical RAG graph."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from src.generation.orchestration.medical_rag.chains import create_medical_rag_chains
from src.generation.orchestration.medical_rag.graph import create_medical_rag_graph
from src.generation.orchestration.medical_rag.retrieval_client import (
    create_retrieval_client,
)
from src.generation.orchestration.medical_rag.schemas import FinalResponse
from src.generation.orchestration.medical_rag.state import MedicalRAGState


async def invoke_medical_rag() -> tuple[FinalResponse, MedicalRAGState]:
    """Invoke the graph and return both external and internal representations."""
    async with httpx.AsyncClient() as http_client:
        chains = create_medical_rag_chains()
        retrieval_client = create_retrieval_client(http_client)
        graph = create_medical_rag_graph(
            rewrite_chain=chains.chain_rewrite,
            router_chain=chains.chain_router,
            generation_chain=chains.chain_generation,
            retrieval_client=retrieval_client,
        )

        initial_state: MedicalRAGState = {
            "original_query": (
                "What are the common causes of peripheral vertigo?"
            )
        }
        result: MedicalRAGState = await graph.ainvoke(initial_state)
        final_response = result["final_response"]

        external_payload: dict[str, Any] = final_response.model_dump(
            mode="json"
        )
        _ = external_payload

        return final_response, result


if __name__ == "__main__":
    final_response, state = asyncio.run(invoke_medical_rag())

    print("FINAL RESPONSE:")
    print(final_response)

    print("\nSTATE:")
    print(state)
    # async def invoke_medical_rag():
    #     async with httpx.AsyncClient() as http_client:
    #         chains = create_medical_rag_chains()
    #         retrieval_client = create_retrieval_client(http_client)

    #         graph = create_medical_rag_graph(
    #             rewrite_chain=chains.chain_rewrite,
    #             router_chain=chains.chain_router,
    #             generation_chain=chains.chain_generation,
    #             retrieval_client=retrieval_client,
    #         )

    #         graph.get_graph(xray=True).draw_mermaid_png(
    #             output_file_path="medical_rag_graph.png"
    #         )

    #         print("Graph saved to medical_rag_graph.png")

    # asyncio.run(invoke_medical_rag())
"""Tests for structured answer generation."""

from __future__ import annotations

import pytest

from src.generation.orchestration.medical_rag.nodes import MedicalRAGNodes
from tests.helpers import StubAsyncChain, StubRetrievalClient


@pytest.mark.asyncio
async def test_generation_stores_answer_and_used_chunk_ids() -> None:
    """Generation output and answer must both remain available in state."""
    generation_chain = StubAsyncChain(
        result={
            "answer": "Peripheral vertigo commonly has vestibular causes.",
            "used_chunks": [
                {"file_id": "file-1", "chunk_id": "chunk-1"}
            ],
        }
    )
    nodes = MedicalRAGNodes(
        rewrite_chain=StubAsyncChain(),
        router_chain=StubAsyncChain(),
        generation_chain=generation_chain,
        retrieval_client=StubRetrievalClient(),
    )

    update = await nodes.generation_node(
        {"generation_prompt": "A complete generation prompt"}
    )

    assert update["answer"] == (
        "Peripheral vertigo commonly has vestibular causes."
    )
    assert update["generation_output"].used_chunks[0].file_id == "file-1"
    assert update["generation_output"].used_chunks[0].chunk_id == "chunk-1"
    assert generation_chain.calls == [
        {"generation_prompt": "A complete generation prompt"}
    ]

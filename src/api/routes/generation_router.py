from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from src.generation.invoke_graph import invoke_medical_rag
from src.generation.orchestration.medical_rag.state import MedicalRAGState


generation_router = APIRouter(
    prefix="/Generation",
    tags=["Generation"],
)


class GenerateRequest(BaseModel):
    """Request body for medical RAG generation."""

    original_query: str = Field(
        ...,
        min_length=1,
        description="The user's medical question.",
    )


class GenerateResponse(BaseModel):
    """Response returned by the medical RAG API."""

    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)


@generation_router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
)
async def generate(
    payload: GenerateRequest,
) -> GenerateResponse:
    """Run the medical RAG workflow and return the final answer."""

    initial_state: MedicalRAGState = {
        "original_query": payload.original_query,
    }

    final_response, _ = await invoke_medical_rag(
        initial_state=initial_state,
    )

    return GenerateResponse(
        answer=final_response.answer,
        sources=final_response.sources,
    )

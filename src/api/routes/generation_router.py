from typing import Any
from src.retrieval.query.query import Query
from pydantic import BaseModel, Field
from src.retrieval import Retriever
from fastapi import APIRouter, Request, status
from src.generation.core.build_prompt_template import build_prompt_template

generation_router = APIRouter(
    prefix="/generation",
    tags=["Generation"],
)


class RetrieveResponse(BaseModel):
    documents: list[Any] = Field(default_factory=list)
    queries: list[Any] = Field(default_factory=list)


@generation_router.post(
    "/generation",
    # response_model=RetrieveResponse,
    status_code=status.HTTP_200_OK,
)
def generation(
    query: Query,
    request: Request,
    decomposition:bool,
) -> str:
    
    reriever = Retriever(request.app.state.vector_store)

    documents, queries = reriever.retrieval_pipeline(
        query=query,
        reranker_model=request.app.state.reRanker_model,
        decomposition=decomposition
        )
    prompt = build_prompt_template(
        queries=queries,
        documents=documents
    )


    return prompt

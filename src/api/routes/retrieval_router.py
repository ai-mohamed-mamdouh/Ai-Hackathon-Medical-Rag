from typing import Any
from src.retrieval.query.query import Query
from pydantic import BaseModel, Field
from src.retrieval import Retriever
from fastapi import APIRouter, Request, status


retrieval_router = APIRouter(
    prefix="/retrieval",
    tags=["Retrieval"],
)


class RetrieveResponse(BaseModel):
    documents: list[Any] = Field(default_factory=list)


@retrieval_router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    status_code=status.HTTP_200_OK,
)
def retrieve(
    query: Query,
    request: Request,
) -> RetrieveResponse:
    
    reriever = Retriever(request.app.state.vector_store)

    documents = reriever.retrieval_pipeline(
        query=query,
        reRanker_model=request.app.state.reRanker_model
        )

    return RetrieveResponse(documents=documents)

from contextlib import asynccontextmanager
import logging
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI

from src.api.routes.indexing_router import indexing_router
from src.api.routes.retrieval_router import retrieval_router
from src.indexing.embeddings import Embedding
from src.indexing.vector_store import VectorStore
from src.retrieval.reranker import RerankerModel
# from src.api.routes.generation_router import generation_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading embedding model...")

    app.state.vector_store = VectorStore( Embedding().get_embedding_model() )
    app.state.reRanker_model=RerankerModel().get_reranker_model()

    logger.info("Embedding model loaded successfully.")

    yield

    # Optional cleanup on application shutdown
    app.state.vector_store = None

    logger.info("Application shutdown complete.")

app = FastAPI(
    title="Medical RAG API",
    description="API for document ingestion and medical RAG operations.",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(indexing_router)
app.include_router(retrieval_router)
# app.include_router(generation_router)


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "Medical RAG API is running",
    }
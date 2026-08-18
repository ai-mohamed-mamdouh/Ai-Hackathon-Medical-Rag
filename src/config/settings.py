from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    BASE_DIR = Path(__file__).resolve().parents[2]

    DATA_DIR = BASE_DIR / "docs"
    ALLOWED_EXTENSIONS = {"pdf"}

    COLLECTION_NAME="medical_giddiness"
    CHROMA_PERSIST_DIRECTORY="./chroma_db"

    TOP_K=4
    RRF_K=60
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100

    EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"
    EMBEDDING_DEVICE = "cpu"
    EMBEDDING_BATCH_SIZE = 8

    GROQ_MODEL_NAME=''
    QUERY_MODEL_NAME='openai/gpt-oss-20b'

    COLLECTION_NAME="medical_giddiness"

    TOP_K=4
    RRF_K=60
    RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L2-v2"  # "cross-encoder/ms-marco-MiniLM-L6-v2"

    RELEVANCE_THRESHOLD=0.8
settings = Settings()
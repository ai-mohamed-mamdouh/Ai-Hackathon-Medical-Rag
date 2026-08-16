from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    BASE_DIR = Path(__file__).resolve().parents[2]

    DATA_DIR = BASE_DIR / "docs"
    ALLOWED_EXTENSIONS = {"pdf"}

    # GROQ_API_KEY = os.getenv("OPENAI_API_KEY")

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
    GROQ_MODEL_NAME=''

    COLLECTION_NAME="medical_giddiness"

    TOP_K=10
    RRF_K=60
    RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L2-v2"  # "cross-encoder/ms-marco-MiniLM-L6-v2"

    RELEVANCE_THRESHOLD=0.8
settings = Settings()
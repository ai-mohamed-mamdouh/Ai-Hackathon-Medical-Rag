from pathlib import Path

from pydantic import Field, PositiveFloat, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Secrets
    GROQ_API_KEY: SecretStr = Field(repr=False)

    # Retrieval API
    RETRIEVAL_BASE_URL: str = "http://127.0.0.1:8000"
    RETRIEVAL_TIMEOUT: PositiveFloat = 100.0

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = BASE_DIR / "docs"

    ALLOWED_EXTENSIONS: set[str] = {"pdf"}

    # Vector DB
    COLLECTION_NAME: str = "medical_giddiness"
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"

    # Retrieval
    TOP_K: int = 10  # Legacy fallback only
    VECTOR_CANDIDATE_K: int = 30
    BM25_CANDIDATE_K: int = 30

    RRF_K: int = 60

    RERANKER_INPUT_K: int = 30
    RERANKER_OUTPUT_K: int = 15

    RELEVANCE_THRESHOLD: float | None = 5.0
    LEXICAL_DEDUP_THRESHOLD: float = 0.60
    SEMANTIC_DEDUP_THRESHOLD: float = 0.88

    FINAL_TOP_K: int = 5

    BM25_MIN_SCORE: float = 0.0
    PARALLEL_RETRIEVAL: bool = False

    # Chunking
    CHUNK_SIZE: int = 450
    CHUNK_OVERLAP: int = 50

    # Embedding
    EMBEDDING_MODEL_NAME: str = "NeuML/pubmedbert-base-embeddings"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 8

    # Reranker
    RERANKER_MODEL_NAME: str = "ncbi/MedCPT-Cross-Encoder"
    RERANKER_DEVICE: str = "cpu"
    RERANKER_BATCH_SIZE: int = 8
    RERANKER_MAX_LENGTH: int | None = None

    # LLM
    QUERY_MODEL_NAME: str = "openai/gpt-oss-20b"
    GROQ_MODEL_NAME: str = "openai/gpt-oss-120b"
    SMALL_GROQ_MODEL_NAME: str = "openai/gpt-oss-20b"


settings = Settings()
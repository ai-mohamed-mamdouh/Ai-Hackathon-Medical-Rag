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
    RETRIEVAL_TIMEOUT: PositiveFloat = 30.0

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = BASE_DIR / "docs"

    ALLOWED_EXTENSIONS: set[str] = {"pdf"}

    # Vector DB
    COLLECTION_NAME: str = "medical_giddiness"
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"

    # Retrieval
    TOP_K: int = 4
    RRF_K: int = 60
    RELEVANCE_THRESHOLD: float = 0.8

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100

    # Embedding
    EMBEDDING_MODEL_NAME: str = "jinaai/jina-embeddings-v2-small-en"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 8

    # LLM
    QUERY_MODEL_NAME: str = "openai/gpt-oss-20b"
    GROQ_MODEL_NAME: str = "openai/gpt-oss-120b"
    SMALL_GROQ_MODEL_NAME: str = "openai/gpt-oss-20b"

    # Reranker
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L2-v2"


settings = Settings()
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Application
    app_name: str = "Evidence Grounded Hybrid RAG"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # LLM
    llm_provider: str = "openai"
    llm_model: str = ""
    openai_api_key: SecretStr | None = None

    # Embeddings
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = ""
    embedding_device: str = "cuda"

    # Vector database
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"

    # Retrieval
    dense_top_k: int = 20
    bm25_top_k: int = 20
    hybrid_top_k: int = 30
    rerank_top_k: int = 8

    # Chunking
    chunk_size: int = 600
    chunk_overlap: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()
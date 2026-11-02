from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = "Esports Intelligence"
    environment: str = "development"
    debug: bool = False
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_persist_directory: str = "data/index/chroma"
    chroma_collection_name: str = "bgmi_rulebook_chunks"
    retrieval_top_k: int = 5
    retrieval_max_distance: float = 0.65
    llm_timeout_seconds: float = 45.0


settings = Settings()
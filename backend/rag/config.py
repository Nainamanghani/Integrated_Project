from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    tavily_api_key: str | None = None 
    # Vector DB
    vector_db_provider: str = "chroma"
    chroma_persist_directory: Path = Path("./storage/chroma")

    # Embeddings
    embedding_model: str | None = None
    llm_model: str | None = None

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Project
    default_project: str = "energy-intelligence"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow" 


settings = Settings()

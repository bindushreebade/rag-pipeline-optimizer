from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).parent.resolve()


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    ANSWER_MODEL: str = "llama-3.1-8b-instant"
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 120
    RETRIEVAL_K: int = 5
    
    # MySQL Configuration
    MYSQL_HOST: str = "localhost"
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "rag_optimizer"

    class Config:
        env_file = str(BASE_DIR.parent / ".env")
        env_file_encoding = "utf-8"


settings = Settings()

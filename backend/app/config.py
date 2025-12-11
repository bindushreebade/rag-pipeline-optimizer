from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

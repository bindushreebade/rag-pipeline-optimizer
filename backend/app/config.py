from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).parent.resolve()


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""

    # ── Models ────────────────────────────────────────────────────────────
    # BGE-base is a much stronger retriever than MiniLM / mpnet on
    # scientific text, and its query instruction is applied in retrievers.py.
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    ANSWER_MODEL: str = "llama-3.3-70b-versatile"
    JUDGE_MODEL: str = "llama-3.1-8b-instant"

    # ── Chunking (characters) ─────────────────────────────────────────────
    # Starting point for a research paper: big enough to hold a whole
    # definition / theorem statement, small enough to stay on one topic.
    # Run `python -m app.sweep` to confirm the best value on your data.
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ── Retrieval ─────────────────────────────────────────────────────────
    CANDIDATE_K: int = 30   # candidates pulled from each retriever (dense, BM25)
    RETRIEVAL_K: int = 5    # chunks kept for scoring / LLM context
    RRF_K: int = 60         # reciprocal-rank-fusion constant
    DENSE_WEIGHT: float = 1.0
    BM25_WEIGHT: float = 1.0

    # MySQL Configuration
    MYSQL_HOST: str = "localhost"
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "rag_optimizer"

    class Config:
        env_file = str(BASE_DIR.parent / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
"""Embeddings, vector stores, and the sparse (BM25) index.

Layout on disk, per collection ``<name>``:

    data/vectorstores/<name>/            Chroma persistence (dense vectors)
    data/vectorstores/<name>/chunks.json Every chunk (id, text, metadata) so
                                         BM25 can be built without Chroma.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from .config import settings

VECTOR_DIR = Path(__file__).parent / "data" / "vectorstores"
STUDY_COLLECTION = "study_documents"   # user uploads from the UI
EVAL_COLLECTION = "eval_documents"     # benchmark PDF used for metrics
CHUNKS_FILE = "chunks.json"

# BGE models are trained with an instruction on the *query* side only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_models: dict[str, SentenceTransformer] = {}


# ─── Embeddings ───────────────────────────────────────────────────────────────

def get_embedding_model(name: str | None = None) -> SentenceTransformer:
    name = name or settings.EMBEDDING_MODEL
    if name not in _models:
        try:
            _models[name] = SentenceTransformer(name, local_files_only=True)
        except Exception:
            _models[name] = SentenceTransformer(name)
    return _models[name]


class SentenceTransformerEmbeddings:
    """LangChain-compatible wrapper with L2-normalised vectors (cosine)."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._loaded = None

    @property
    def model(self) -> SentenceTransformer:
        if self._loaded is None:
            self._loaded = get_embedding_model(self.model_name)
        return self._loaded

    @property
    def _query_prefix(self) -> str:
        return BGE_QUERY_INSTRUCTION if "bge" in self.model_name.lower() else ""

    def embed_documents(self, texts):
        return self.model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [self._query_prefix + text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()[0]


# ─── Dense store ──────────────────────────────────────────────────────────────

def collection_path(name: str) -> Path:
    return VECTOR_DIR / name


def open_vector_store(name: str) -> Chroma:
    """Open (creating if needed) a cosine-space Chroma collection."""
    path = collection_path(name)
    path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=name,
        persist_directory=str(path),
        embedding_function=SentenceTransformerEmbeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_vector_store(name: str) -> Chroma | None:
    if not (collection_path(name) / CHUNKS_FILE).exists():
        return None
    return open_vector_store(name)


def load_study_vector_store() -> Chroma | None:
    return load_vector_store(STUDY_COLLECTION)


# ─── Chunk store + BM25 ───────────────────────────────────────────────────────

def _chunks_path(name: str) -> Path:
    return collection_path(name) / CHUNKS_FILE


def read_chunk_records(name: str) -> list[dict[str, Any]]:
    path = _chunks_path(name)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_chunk_records(name: str, records: list[dict[str, Any]]) -> None:
    path = _chunks_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    _bm25_cache.pop(name, None)


def load_chunks(name: str) -> list[Document]:
    return [
        Document(page_content=r["text"], metadata=r["metadata"])
        for r in read_chunk_records(name)
    ]


_STOPWORDS = frozenset(
    "a an and are as at be been by do does for from has have how in is it its "
    "of on or that the their this to was were what when where which who why "
    "with can may not under than then these those into also such".split()
)
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s")


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(t) for t in tokens if t not in _STOPWORDS]


_bm25_cache: dict[str, tuple[float, list[Document], Any]] = {}


def get_bm25(name: str):
    """Return (chunks, BM25Okapi) for a collection, cached by file mtime."""
    from rank_bm25 import BM25Okapi

    path = _chunks_path(name)
    if not path.exists():
        return None

    mtime = path.stat().st_mtime
    cached = _bm25_cache.get(name)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]

    chunks = load_chunks(name)
    bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])
    _bm25_cache[name] = (mtime, chunks, bm25)
    return chunks, bm25
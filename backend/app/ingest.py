"""PDF loading, cleaning, chunking, and indexing (dense + BM25 chunk store)."""

from __future__ import annotations

import bisect
import hashlib
import os
import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import settings
from .retrievers import (
    STUDY_COLLECTION,
    VECTOR_DIR,
    _bm25_cache,
    collection_path,
    open_vector_store,
    read_chunk_records,
    write_chunk_records,
)

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
MIN_CHUNK_CHARS = 40


# ─── Loading & cleaning ───────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Undo PDF line-wrapping artefacts without destroying real structure."""
    text = text.replace("\u00ad", "")                   # soft hyphens
    text = re.sub(r"-\n(?=\w)", "-", text)              # keep hyphen at wraps
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)        # single \n -> space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pages(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append((number, text))
    return pages


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_pdf(
    path: Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Chunk a PDF as one continuous text so definitions and theorem
    statements are not cut at page boundaries. Each chunk records the page
    it starts on (``page``) and ends on (``page_end``)."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = (
        settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    )

    pages = load_pdf_pages(path)
    if not pages:
        return []

    full_text = ""
    page_starts: list[int] = []
    page_numbers: list[int] = []
    for number, text in pages:
        page_starts.append(len(full_text))
        page_numbers.append(number)
        full_text += text + "\n\n"

    def page_at(offset: int) -> int:
        index = bisect.bisect_right(page_starts, max(offset, 0)) - 1
        return page_numbers[max(index, 0)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        add_start_index=True,
    )
    raw_chunks = splitter.create_documents([full_text])

    chunks: list[Document] = []
    for raw in raw_chunks:
        content = raw.page_content.strip()
        if len(content) < MIN_CHUNK_CHARS:
            continue
        start = raw.metadata.get("start_index", 0)
        index = len(chunks)
        page = page_at(start)
        page_end = page_at(start + len(raw.page_content) - 1)
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:8]
        chunks.append(
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                    "page": page,
                    "page_end": page_end,
                    "chunk_index": index,
                    "chunk_id": f"{path.stem}-p{page}-c{index}-{digest}",
                },
            )
        )
    return chunks


# ─── Indexing ─────────────────────────────────────────────────────────────────

def index_chunks(
    chunks: list[Document],
    collection: str,
    reset: bool = False,
) -> None:
    """Write chunks to the dense store and the chunk file used by BM25.
    Re-indexing a source replaces its previous chunks."""
    store = open_vector_store(collection)

    if reset:
        store.delete_collection()
        store = open_vector_store(collection)
        records: list[dict] = []
    else:
        records = read_chunk_records(collection)
        incoming = {c.metadata["source"] for c in chunks}
        stale = [r for r in records if r["metadata"]["source"] in incoming]
        if stale:
            store.delete(ids=[r["metadata"]["chunk_id"] for r in stale])
        records = [r for r in records if r["metadata"]["source"] not in incoming]

    batch = 128
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        store.add_documents(part, ids=[c.metadata["chunk_id"] for c in part])

    records.extend(
        {"text": c.page_content, "metadata": c.metadata} for c in chunks
    )
    write_chunk_records(collection, records)
    _bm25_cache.pop(collection, None)


def ingest_files(
    paths: Iterable[Path],
    collection: str = STUDY_COLLECTION,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    """Add PDFs to a collection (default: the shared study collection)."""
    paths = [Path(p) for p in paths]
    chunks: list[Document] = []
    for path in paths:
        if path.suffix.lower() == ".pdf":
            chunks.extend(chunk_pdf(path, chunk_size, chunk_overlap))

    if chunks:
        index_chunks(chunks, collection)

    return {"files": [p.name for p in paths], "chunks": len(chunks)}


def ingest_document():
    print("Starting ingestion...")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = [
        UPLOAD_DIR / name
        for name in os.listdir(UPLOAD_DIR)
        if name.lower().endswith(".pdf")
    ]
    result = ingest_files(pdfs)
    print(
        f"Ingestion complete: {result['chunks']} chunks from "
        f"{len(result['files'])} file(s)."
    )


if __name__ == "__main__":
    ingest_document()
"""Build the benchmark index (``eval_documents``) from the benchmark PDF.

    python -m app.rebuild_comparable_collections
    python -m app.rebuild_comparable_collections --chunk-size 800 --overlap 160

All four pipelines (A-D) read this single collection, so differences in the
metrics come from the retrieval strategy alone, not from different indexes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import settings
from .ingest import chunk_pdf, index_chunks
from .retrievers import EVAL_COLLECTION

PDF_PATH = Path(__file__).parent / "uploads" / "Sample_pdf.pdf"


def rebuild(
    collection: str = EVAL_COLLECTION,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> int:
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Benchmark PDF not found: {PDF_PATH}")

    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP if overlap is None else overlap

    chunks = chunk_pdf(PDF_PATH, chunk_size, overlap)
    index_chunks(chunks, collection, reset=True)
    print(
        f"{collection}: {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={overlap}, model={settings.EMBEDDING_MODEL})"
    )
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    args = parser.parse_args()
    rebuild(chunk_size=args.chunk_size, overlap=args.overlap)
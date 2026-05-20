import os
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

from .config import settings
from .retrievers import (
    STUDY_COLLECTION,
    VECTOR_DIR,
    SentenceTransformerEmbeddings,
    get_embedding_model,
)

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
STUDY_VECTOR_DIR = VECTOR_DIR / STUDY_COLLECTION


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": page_number},
                )
            )

    return documents


def _load_documents(paths: Iterable[Path]) -> list[Document]:
    documents = []
    for path in paths:
        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path))
    return documents


def ingest_files(paths: Iterable[Path]) -> dict:
    """Add uploaded PDFs to the shared study vectorstore."""
    paths = [Path(path) for path in paths]

    if SentenceTransformer is None:
        raise ImportError("sentence_transformers is required for document ingestion.")

    documents = _load_documents(paths)
    if not documents:
        return {"files": [path.name for path in paths], "chunks": 0}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embedding_fn = SentenceTransformerEmbeddings(get_embedding_model())

    STUDY_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        collection_name=STUDY_COLLECTION,
        persist_directory=str(STUDY_VECTOR_DIR),
        embedding_function=embedding_fn,
    )
    vectorstore.add_documents(chunks)

    return {"files": [path.name for path in paths], "chunks": len(chunks)}


def ingest_document():
    print("Starting ingestion...")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = [
        UPLOAD_DIR / fname
        for fname in os.listdir(UPLOAD_DIR)
        if fname.lower().endswith(".pdf")
    ]
    result = ingest_files(pdfs)
    print(
        f"Ingestion complete: {result['chunks']} chunks from "
        f"{len(result['files'])} file(s)."
    )


if __name__ == "__main__":
    ingest_document()

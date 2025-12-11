from dotenv import load_dotenv
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DIR = BASE_DIR / "data" / "vectorstores"

PIPELINES = ["pipeline_a", "pipeline_b", "pipeline_c", "pipeline_d"]

# Embedding wrapper
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True).tolist()[0]

def ingest_document():
    print("🔄 Starting ingestion...")

    # Load PDFs
    documents = []
    for fname in os.listdir(UPLOAD_DIR):
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(str(UPLOAD_DIR / fname))
            documents.extend(loader.load())

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # Load embeddings
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embedding_fn = SentenceTransformerEmbeddings(embedder)

    # Build vectorstores
    for name in PIPELINES:
        print(f"📦 Building {name}")
        pipeline_path = VECTOR_DIR / name
        pipeline_path.mkdir(parents=True, exist_ok=True)

        vectorstore = Chroma(
            collection_name=name,
            persist_directory=str(pipeline_path),
            embedding_function=embedding_fn,
        )

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        vectorstore.add_texts(texts=texts, metadatas=metadatas)
        print(f"✅ Saved at {pipeline_path}")

    print("🎉 Ingestion complete!")

if __name__ == "__main__":
    ingest_document()

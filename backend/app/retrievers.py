from pathlib import Path
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from .config import settings

VECTOR_DIR = Path(__file__).parent / "data" / "vectorstores"

# Embedding wrapper
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True).tolist()[0]

# Load embedding model once
embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
embedding_fn = SentenceTransformerEmbeddings(embedder)

def load_vector_store(name: str):
    path = VECTOR_DIR / name
    if not path.exists():
        print(f"❌ Vector store not found: {path}")
        return None

    return Chroma(
        collection_name=name,
        persist_directory=str(path),
        embedding_function=embedding_fn,
    )

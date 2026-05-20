from pathlib import Path
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

from .config import settings

VECTOR_DIR = Path(__file__).parent / "data" / "vectorstores"
STUDY_COLLECTION = "study_documents"
_embedder = None


class SentenceTransformerEmbeddings:
    def __init__(self, model=None):
        self.model = model

    def embed_documents(self, texts):
        return self._model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text):
        return self._model.encode([text], convert_to_numpy=True).tolist()[0]

    @property
    def _model(self):
        if self.model is None:
            self.model = get_embedding_model()
        return self.model


def get_embedding_model():
    global _embedder

    if _embedder is None:
        try:
            _embedder = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                local_files_only=True,
            )
        except Exception:
            _embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    return _embedder


embedding_fn = SentenceTransformerEmbeddings()


def load_vector_store(name: str):
    path = VECTOR_DIR / name
    if not path.exists():
        return None

    return Chroma(
        collection_name=name,
        persist_directory=str(path),
        embedding_function=embedding_fn,
    )


def load_study_vector_store():
    return load_vector_store(STUDY_COLLECTION)

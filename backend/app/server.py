from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .ingest import UPLOAD_DIR, ingest_files
from .pipelines import answer_from_study_documents, run_all_pipelines
from .database import db

app = FastAPI(title="Study Research Q&A")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    try:
        db.init_db()
    except Exception as e:
        print(f"Warning: Could not initialize database: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


class ChatEntry(BaseModel):
    question: str
    answer: str
    pipeline: Optional[str] = None
    document_name: Optional[str] = None
    retrieved_documents: Optional[str] = None


class PipelineComparisonEntry(BaseModel):
    question: str
    pipeline_a_answer: str
    pipeline_b_answer: str
    pipeline_c_answer: str
    pipeline_d_answer: str
    best_pipeline: str
    document_name: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(path.name for path in UPLOAD_DIR.glob("*.pdf"))
    return {"documents": files}


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        destination = UPLOAD_DIR / Path(file.filename).name
        destination.write_bytes(await file.read())
        saved_paths.append(destination)

    result = ingest_files(saved_paths)
    return {"message": "Documents processed successfully.", **result}


@app.post("/ask")
def ask_question(data: Question):
    return answer_from_study_documents(data.question)


@app.post("/compare")
def compare_pipelines(data: Question):
    return run_all_pipelines(data.question)


# ─── Chat History Endpoints ───────────────────────────────────────────

@app.post("/history/save")
async def save_chat_history(entry: ChatEntry):
    """Save a chat interaction to MySQL history."""
    try:
        chat_id = db.save_chat(
            question=entry.question,
            answer=entry.answer,
            pipeline=entry.pipeline,
            document_name=entry.document_name,
            retrieved_documents=entry.retrieved_documents
        )
        return {"id": chat_id, "message": "Chat saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving chat: {str(e)}")


@app.post("/history/save-comparison")
async def save_pipeline_comparison(entry: PipelineComparisonEntry):
    """Save multi-pipeline comparison results to history."""
    try:
        results = {
            "A": entry.pipeline_a_answer,
            "B": entry.pipeline_b_answer,
            "C": entry.pipeline_c_answer,
            "D": entry.pipeline_d_answer
        }
        comparison_id = db.save_pipeline_comparison(
            question=entry.question,
            results=results,
            best_pipeline=entry.best_pipeline,
            document_name=entry.document_name
        )
        return {"id": comparison_id, "message": "Comparison saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving comparison: {str(e)}")


@app.get("/history")
def get_chat_history(limit: int = 50, document_name: Optional[str] = None):
    """Retrieve chat history with optional filtering."""
    try:
        history = db.get_chat_history(limit=limit, document_name=document_name)
        return {"count": len(history), "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


@app.get("/history/comparisons")
def get_pipeline_comparisons(limit: int = 50, document_name: Optional[str] = None):
    """Retrieve pipeline comparison history."""
    try:
        comparisons = db.get_pipeline_comparisons(limit=limit, document_name=document_name)
        return {"count": len(comparisons), "comparisons": comparisons}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving comparisons: {str(e)}")


@app.get("/history/stats")
def get_history_statistics(document_name: Optional[str] = None):
    """Get statistics about chat history."""
    try:
        stats = db.get_statistics(document_name=document_name)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")


@app.delete("/history/{chat_id}")
def delete_chat_entry(chat_id: int):
    """Delete a specific chat history entry."""
    try:
        success = db.delete_chat_history(chat_id)
        if success:
            return {"message": "Chat entry deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Chat entry not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chat: {str(e)}")

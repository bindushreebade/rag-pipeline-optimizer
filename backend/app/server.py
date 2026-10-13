from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .database import db
from .ingest import UPLOAD_DIR, ingest_files
from .metrics import calculate_all_pipeline_metrics
from .pipelines import answer_from_study_documents, run_all_pipelines

app = FastAPI(title="Study Research Q&A")


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


@app.get("/metrics/retrieval")
def retrieval_metrics(
    retrieve_k: int = Query(30, ge=1, description="Candidate chunks retrieved before ranking"),
    final_k: int = Query(5, description="Chunks kept for scoring: 1, 3, 5 or 10"),
    split: str = Query("all", description="all, dev or test"),
):
    if final_k not in (1, 3, 5, 10):
        raise HTTPException(status_code=422, detail="final_k must be 1, 3, 5 or 10.")
    if split not in ("all", "dev", "test"):
        raise HTTPException(status_code=422, detail="split must be all, dev or test.")
    try:
        return calculate_all_pipeline_metrics(
            retrieve_k=retrieve_k,
            final_k=final_k,
            split=split,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error calculating metrics: {error}")


# ─── Chat History Endpoints ───────────────────────────────────────────

@app.post("/history/save")
async def save_chat_history(entry: ChatEntry):
    try:
        chat_id = db.save_chat(
            question=entry.question,
            answer=entry.answer,
            pipeline=entry.pipeline,
            document_name=entry.document_name,
            retrieved_documents=entry.retrieved_documents,
        )
        return {"id": chat_id, "message": "Chat saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving chat: {str(e)}")


@app.post("/history/save-comparison")
async def save_pipeline_comparison(entry: PipelineComparisonEntry):
    try:
        results = {
            "A": entry.pipeline_a_answer,
            "B": entry.pipeline_b_answer,
            "C": entry.pipeline_c_answer,
            "D": entry.pipeline_d_answer,
        }
        comparison_id = db.save_pipeline_comparison(
            question=entry.question,
            results=results,
            best_pipeline=entry.best_pipeline,
            document_name=entry.document_name,
        )
        return {"id": comparison_id, "message": "Comparison saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving comparison: {str(e)}")


@app.get("/history")
def get_chat_history(limit: int = 50, document_name: Optional[str] = None):
    try:
        history = db.get_chat_history(limit=limit, document_name=document_name)
        return {"count": len(history), "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


@app.get("/history/comparisons")
def get_pipeline_comparisons(limit: int = 50, document_name: Optional[str] = None):
    try:
        comparisons = db.get_pipeline_comparisons(limit=limit, document_name=document_name)
        return {"count": len(comparisons), "comparisons": comparisons}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving comparisons: {str(e)}")


@app.get("/history/stats")
def get_history_statistics(document_name: Optional[str] = None):
    try:
        return db.get_statistics(document_name=document_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")


@app.delete("/history/{chat_id}")
def delete_chat_entry(chat_id: int):
    try:
        success = db.delete_chat_history(chat_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chat: {str(e)}")
    if not success:
        raise HTTPException(status_code=404, detail="Chat entry not found")
    return {"message": "Chat entry deleted successfully"}
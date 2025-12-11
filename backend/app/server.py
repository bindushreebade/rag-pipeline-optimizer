from fastapi import FastAPI
from pydantic import BaseModel
from .pipelines import run_all_pipelines

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(data: Question):
    result = run_all_pipelines(data.question)
    return result

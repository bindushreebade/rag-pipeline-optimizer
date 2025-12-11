from groq import Groq
from sentence_transformers import SentenceTransformer
from .config import settings
from .retrievers import load_vector_store

client = Groq(api_key=settings.GROQ_API_KEY)

# Load embedder once
embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

def ask_llm(question, context, model="llama-3.3-70b-versatile"):
    prompt = f"""
You are a RAG assistant. Use ONLY the provided context.

Question:
{question}

Context:
{context}

Give a short and accurate answer.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def pipeline_a(question: str):
    vs = load_vector_store("pipeline_a")
    docs = vs.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])
    answer = ask_llm(question, context)
    return {"pipeline": "A", "answer": answer, "docs": [d.page_content for d in docs]}


def pipeline_b(question: str):
    vs = load_vector_store("pipeline_b")
    docs = vs.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])
    answer = ask_llm(question, context)
    return {"pipeline": "B", "answer": answer, "docs": [d.page_content for d in docs]}


def pipeline_c(question: str):
    vs = load_vector_store("pipeline_c")
    docs = vs.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])
    answer = ask_llm(question, context, model="llama-3.3-70b-versatile")
    return {"pipeline": "C", "answer": answer, "docs": [d.page_content for d in docs]}


def pipeline_d(question: str):
    vs = load_vector_store("pipeline_d")
    docs = vs.similarity_search(question, k=5)
    reranked = sorted(docs, key=lambda x: len(x.page_content), reverse=True)[:3]
    context = "\n".join([d.page_content for d in reranked])
    answer = ask_llm(question, context)
    return {"pipeline": "D", "answer": answer, "docs": [d.page_content for d in reranked]}


def run_all_pipelines(question: str):
    results = {
        "A": pipeline_a(question),
        "B": pipeline_b(question),
        "C": pipeline_c(question),
        "D": pipeline_d(question),
    }

    best = judge_best_pipeline(question, results)
    results["best_pipeline"] = best

    return results

def judge_best_pipeline(question, pipeline_outputs):
    """
    Uses Groq LLM to decide which pipeline answer is best.
    """

    prompt = f"""
You are an evaluation LLM. Your job is to judge which pipeline produced the BEST answer.

Question:
{question}

Here are the answers:

Pipeline A:
{pipeline_outputs['A']['answer']}

Pipeline B:
{pipeline_outputs['B']['answer']}

Pipeline C:
{pipeline_outputs['C']['answer']}

Pipeline D:
{pipeline_outputs['D']['answer']}

Pick the best pipeline by writing ONLY a single letter: A, B, C, or D.
No explanation.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # fast judge model
        messages=[{"role": "user", "content": prompt}]
    )

    best = response.choices[0].message.content.strip()

    # clean result just in case
    best = best.replace(".", "").upper()

    if best not in ["A", "B", "C", "D"]:
        best = "A"  # fallback

    return best

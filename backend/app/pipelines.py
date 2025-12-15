from groq import Groq
from sentence_transformers import SentenceTransformer
from .config import settings
from .retrievers import load_vector_store
from sklearn.metrics.pairwise import cosine_similarity

client = Groq(api_key=settings.GROQ_API_KEY)

# Load embedder once
embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

def ask_llm(question, context, model="llama-3.3-70b-versatile"):
    prompt = f"""
Answer ONLY using facts explicitly present in the context.
If the answer is not found in the context, reply exactly:
"Answer not found in the document."
Do NOT infer, assume, or generalize.

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

def rerank_by_similarity(query, docs, embedder, top_k=3):
    query_emb = embedder.encode(query)

    scored = []
    for doc in docs:
        doc_emb = embedder.encode(doc.page_content)
        score = cosine_similarity([query_emb], [doc_emb])[0][0]
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]

def pipeline_d(question: str):
    vs = load_vector_store("pipeline_d")

    # Step 1: Query decomposition (forces diverse retrieval)
    subqueries = [
        question,
        "proposed system architecture",
        "database design",
        "modules and features",
        "results and discussion"
    ]

    all_docs = []

    # Step 2: Retrieve documents per subquery
    for q in subqueries:
        docs = vs.similarity_search(q, k=4)
        all_docs.extend(docs)

    # Remove duplicate chunks
    unique_docs = {
        doc.page_content: doc for doc in all_docs
    }.values()

    unique_docs = list(unique_docs)

    if not unique_docs:
        return {
            "pipeline": "D",
            "answer": "No relevant documents found.",
            "docs": []
        }

    # Step 3: Semantic reranking (final relevance filter)
    reranked_docs = rerank_by_similarity(
        query=question,
        docs=unique_docs,
        embedder=embedder,
        top_k=3
    )

    # Step 4: Build focused context
    context = "\n\n".join(doc.page_content for doc in reranked_docs)

    # Step 5: Generate grounded answer
    answer = ask_llm(question, context)

    return {
        "pipeline": "D",
        "answer": answer,
        "docs": [doc.page_content for doc in reranked_docs]
    }


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

    best = best.replace(".", "").upper()

    if best not in ["A", "B", "C", "D"]:
        best = "A"  # fallback

    return best

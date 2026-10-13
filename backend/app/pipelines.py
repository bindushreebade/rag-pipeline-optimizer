"""Retrieval pipelines A-D and answer generation.

A  Dense           bi-encoder cosine similarity (baseline)
B  Hybrid          dense + BM25 fused with Reciprocal Rank Fusion
C  Dense + Rerank  dense candidates re-scored by a cross-encoder
D  Hybrid + Rerank hybrid candidates re-scored by a cross-encoder (best)
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from .config import settings
from .retrievers import (
    STUDY_COLLECTION,
    get_bm25,
    load_vector_store,
    tokenize,
)

PIPELINE_NAMES = ("A", "B", "C", "D")

_client = None
_reranker = None


# ─── Lazy model handles ───────────────────────────────────────────────────────

def get_client():
    global _client
    if _client is None:
        from groq import Groq

        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
    return _reranker


# ─── Generation ───────────────────────────────────────────────────────────────

def ask_llm(question: str, context: str, model: str | None = None) -> str:
    prompt = f"""You are answering a question about a research paper.
Use ONLY facts explicitly present in the context below.
If the context does not contain the answer, reply exactly:
"Answer not found in the document."
Do not infer, assume, or add outside knowledge.

Context:
{context}

Question:
{question}

Give a concise, accurate answer in 1-4 sentences."""

    response = get_client().chat.completions.create(
        model=model or settings.ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_chunk_id(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return str(metadata.get("chunk_id") or hash(getattr(doc, "page_content", "")))


def _deduplicate(docs: list[Any]) -> list[Any]:
    seen, unique = set(), []
    for doc in docs:
        chunk_id = get_chunk_id(doc)
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(doc)
    return unique


# ─── Retrievers ───────────────────────────────────────────────────────────────

def dense_search(
    collection: str,
    question: str,
    k: int,
    source_filter: str | None = None,
) -> list[Any]:
    store = load_vector_store(collection)
    if store is None:
        raise FileNotFoundError(f"Collection '{collection}' was not found.")

    kwargs: dict[str, Any] = {"k": k}
    if source_filter:
        kwargs["filter"] = {"source": source_filter}
    return _deduplicate(store.similarity_search(question, **kwargs))


def sparse_search(
    collection: str,
    question: str,
    k: int,
    source_filter: str | None = None,
) -> list[Any]:
    loaded = get_bm25(collection)
    if loaded is None:
        raise FileNotFoundError(f"Collection '{collection}' was not found.")

    chunks, bm25 = loaded
    scores = bm25.get_scores(tokenize(question))
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

    results = []
    for index in order:
        if scores[index] <= 0:
            break
        doc = chunks[index]
        if source_filter and doc.metadata.get("source") != source_filter:
            continue
        results.append(doc)
        if len(results) >= k:
            break
    return results


def reciprocal_rank_fusion(
    ranked_lists: list[list[Any]],
    weights: list[float] | None = None,
    rrf_k: int | None = None,
) -> list[Any]:
    rrf_k = rrf_k or settings.RRF_K
    weights = weights or [1.0] * len(ranked_lists)

    scores: dict[str, float] = {}
    by_id: dict[str, Any] = {}
    for docs, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(docs, start=1):
            chunk_id = get_chunk_id(doc)
            by_id.setdefault(chunk_id, doc)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)

    ordered = sorted(scores, key=scores.get, reverse=True)
    return [by_id[chunk_id] for chunk_id in ordered]


def hybrid_search(
    collection: str,
    question: str,
    k: int,
    source_filter: str | None = None,
) -> list[Any]:
    dense = dense_search(collection, question, k, source_filter)
    sparse = sparse_search(collection, question, k, source_filter)
    return reciprocal_rank_fusion(
        [dense, sparse],
        weights=[settings.DENSE_WEIGHT, settings.BM25_WEIGHT],
    )


def rerank(question: str, docs: list[Any], top_k: int) -> list[Any]:
    if not docs:
        return []
    scores = get_reranker().predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# ─── Public retrieval entry point (shared by app and evaluator) ──────────────

def retrieve_pipeline_docs(
    pipeline_name: str,
    question: str,
    top_k: int | None = None,
    candidate_k: int | None = None,
    collection: str = STUDY_COLLECTION,
    source_filter: str | None = None,
) -> list[Any]:
    pipeline_name = pipeline_name.upper()
    top_k = top_k or settings.RETRIEVAL_K
    candidate_k = candidate_k or settings.CANDIDATE_K

    if pipeline_name not in PIPELINE_NAMES:
        raise ValueError(f"Unknown pipeline '{pipeline_name}'. Choose A, B, C, or D.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    candidate_k = max(candidate_k, top_k)

    if pipeline_name == "A":
        return dense_search(collection, question, candidate_k, source_filter)[:top_k]

    if pipeline_name == "B":
        return hybrid_search(collection, question, candidate_k, source_filter)[:top_k]

    if pipeline_name == "C":
        docs = dense_search(collection, question, candidate_k, source_filter)
        return rerank(question, docs, top_k)

    docs = hybrid_search(collection, question, candidate_k, source_filter)
    return rerank(question, docs, top_k)


# ─── Pipeline runners used by the API ────────────────────────────────────────

def _format_context(docs: list[Any]) -> str:
    parts = []
    for doc in docs:
        meta = doc.metadata or {}
        parts.append(
            f"[{meta.get('source', 'document')}, p.{meta.get('page', '?')}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def _run_pipeline(
    pipeline_name: str,
    question: str,
    generate_answer: bool = True,
    collection: str = STUDY_COLLECTION,
) -> dict[str, Any]:
    docs = retrieve_pipeline_docs(
        pipeline_name,
        question,
        top_k=settings.RETRIEVAL_K,
        collection=collection,
    )
    context = _format_context(docs)

    if not generate_answer:
        answer = ""
    elif context:
        answer = ask_llm(question, context)
    else:
        answer = "No relevant documents found."

    return {
        "pipeline": pipeline_name,
        "answer": answer,
        "docs": [doc.page_content for doc in docs],
        "chunk_ids": [get_chunk_id(doc) for doc in docs],
        "pages": [doc.metadata.get("page") for doc in docs],
    }


def pipeline_a(question: str, generate_answer: bool = True, **kw) -> dict[str, Any]:
    return _run_pipeline("A", question, generate_answer, **kw)


def pipeline_b(question: str, generate_answer: bool = True, **kw) -> dict[str, Any]:
    return _run_pipeline("B", question, generate_answer, **kw)


def pipeline_c(question: str, generate_answer: bool = True, **kw) -> dict[str, Any]:
    return _run_pipeline("C", question, generate_answer, **kw)


def pipeline_d(question: str, generate_answer: bool = True, **kw) -> dict[str, Any]:
    return _run_pipeline("D", question, generate_answer, **kw)


def run_all_pipelines(question: str) -> dict[str, Any]:
    results: dict[str, Any] = {
        "A": pipeline_a(question),
        "B": pipeline_b(question),
        "C": pipeline_c(question),
        "D": pipeline_d(question),
    }
    results["best_pipeline"] = judge_best_pipeline(question, results)
    return results


def judge_best_pipeline(question: str, pipeline_outputs: dict[str, Any]) -> str:
    """Pick the best generated answer using an LLM judge."""
    answers = "\n\n".join(
        f"Pipeline {name}:\n{pipeline_outputs[name]['answer']}"
        for name in PIPELINE_NAMES
    )
    prompt = f"""You are an evaluation LLM. Judge which pipeline produced the best
answer based on factual correctness, relevance, and grounding.

Question:
{question}

{answers}

Return only one letter: A, B, C, or D."""

    response = get_client().chat.completions.create(
        model=settings.JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    best = (response.choices[0].message.content or "").strip().upper()
    best = best.replace(".", "")[:1]
    return best if best in PIPELINE_NAMES else "D"


def answer_from_study_documents(question: str) -> dict[str, Any]:
    """Answer from uploaded documents using the best pipeline (D)."""
    if load_vector_store(STUDY_COLLECTION) is None:
        return {
            "answer": "Study documents not found. Please upload documents first.",
            "sources": [],
        }

    docs = retrieve_pipeline_docs(
        "D",
        question,
        top_k=settings.RETRIEVAL_K,
        collection=STUDY_COLLECTION,
    )
    answer = ask_llm(question, _format_context(docs)) if docs else (
        "Answer not found in the document."
    )

    sources = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        text = doc.page_content
        sources.append(
            {
                "source": meta.get("source", f"Document {index}"),
                "page": meta.get("page", ""),
                "text": text[:200] + ("..." if len(text) > 200 else ""),
            }
        )

    return {"answer": answer, "sources": sources, "model": settings.ANSWER_MODEL}
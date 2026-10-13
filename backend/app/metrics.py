"""Retrieval metrics for pipelines A-D.

Relevance is labelled at the *page* level, so the ground truth stays valid
when chunk size or overlap changes (the old chunk-ID labels broke every time
the chunking changed). A retrieved chunk is relevant when the page range it
covers overlaps a labelled page. Optionally add an ``"evidence"`` list of
phrases to a ground-truth item to require that the chunk also contains one of
them, which makes the labels much stricter.

Reported at k in (1, 3, 5, 10):
  MRR@k        mean reciprocal rank of the first relevant chunk
  Recall@k     fraction of labelled pages covered by the top-k chunks
  Hit@k        fraction of questions with >=1 relevant chunk in the top k
  Precision@k  fraction of the top-k chunks that are relevant
  nDCG@k       rank-aware gain with binary relevance

Splits: questions with an odd number (Q01, Q03, ...) are the *dev* split used
to choose hyper-parameters; even ones are the held-out *test* split used to
report results.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

try:
    from .pipelines import PIPELINE_NAMES, get_chunk_id, retrieve_pipeline_docs
    from .retrievers import EVAL_COLLECTION, read_chunk_records
except ImportError:  # running as a script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.pipelines import PIPELINE_NAMES, get_chunk_id, retrieve_pipeline_docs
    from app.retrievers import EVAL_COLLECTION, read_chunk_records

GROUND_TRUTH_PATH = Path(__file__).parent / "data" / "ground_truth.json"
K_VALUES = (1, 3, 5, 10)
MAX_K = max(K_VALUES)
SPLITS = ("all", "dev", "test")
_NEG_PAGE_RE = re.compile(r"-p(\d+)-")


# ─── Ground truth ─────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _relevant_pages(item: dict[str, Any]) -> set[int]:
    pages = {int(p) for p in item.get("source_pages", [])}
    for chunk_id in item.get("relevant_chunk_ids", []):
        match = _NEG_PAGE_RE.search(chunk_id)
        if match:
            pages.add(int(match.group(1)))
    return pages


def _in_split(item: dict[str, Any], split: str) -> bool:
    if split == "all":
        return True
    number = int(re.sub(r"\D", "", item["id"]) or 0)
    return (number % 2 == 1) == (split == "dev")


def _load_questions(split: str = "all") -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}.")

    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)

    if not isinstance(questions, list):
        raise ValueError("ground_truth.json must contain a JSON array.")

    selected = []
    for item in questions:
        if not item.get("id") or not item.get("question"):
            raise ValueError("Every ground-truth item needs 'id' and 'question'.")
        item["_pages"] = _relevant_pages(item)
        if not item["_pages"]:
            raise ValueError(f"{item['id']} has no labelled pages.")
        if _in_split(item, split):
            selected.append(item)
    return selected


def _page_range(metadata: dict[str, Any]) -> set[int]:
    start = int(metadata.get("page", 0) or 0)
    end = int(metadata.get("page_end", start) or start)
    return set(range(start, end + 1))


def _is_relevant(text: str, metadata: dict[str, Any], item: dict[str, Any]) -> bool:
    if not (_page_range(metadata) & item["_pages"]):
        return False
    phrases = item.get("evidence")
    if not phrases:
        return True
    haystack = _norm(text)
    return any(_norm(p) in haystack for p in phrases)


# ─── Scoring ──────────────────────────────────────────────────────────────────

def _score_question(
    item: dict[str, Any],
    ranked: list[tuple[str, set[int], bool]],
    total_relevant_chunks: int,
) -> dict[str, Any]:
    """ranked: (chunk_id, pages covered, is_relevant) in rank order."""
    out: dict[str, Any] = {"at_k": {}}
    rel_pages = item["_pages"]

    for k in K_VALUES:
        top = ranked[:k]
        flags = [rel for _, _, rel in top]

        rr = next((1.0 / r for r, flag in enumerate(flags, 1) if flag), 0.0)
        covered = set().union(*(p for _, p, _ in top)) if top else set()
        recall = len(covered & rel_pages) / len(rel_pages)
        precision = sum(flags) / k
        dcg = sum(1.0 / math.log2(r + 1) for r, flag in enumerate(flags, 1) if flag)
        ideal_n = min(k, total_relevant_chunks)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_n + 1))

        out["at_k"][k] = {
            "mrr": rr,
            "recall": recall,
            "hit": 1.0 if any(flags) else 0.0,
            "precision": precision,
            "ndcg": dcg / idcg if idcg else 0.0,
        }

    first = next((r for r, (_, _, rel) in enumerate(ranked, 1) if rel), None)
    out["first_relevant_rank"] = first
    out["retrieved_ids"] = [cid for cid, _, _ in ranked]
    out["relevant_pages"] = sorted(rel_pages)
    return out


def _aggregate(rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    count = len(rows) or 1
    return {
        k: {
            metric: sum(r["at_k"][k][metric] for r in rows) / count
            for metric in ("mrr", "recall", "hit", "precision", "ndcg")
        }
        for k in K_VALUES
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def calculate_pipeline_metrics(
    pipeline_name: str,
    retrieve_k: int = 30,
    final_k: int = 5,
    collection: str = EVAL_COLLECTION,
    split: str = "all",
    source_filter: str | None = None,
) -> dict[str, Any]:
    pipeline_name = pipeline_name.upper()
    if pipeline_name not in PIPELINE_NAMES:
        raise ValueError(f"Unknown pipeline '{pipeline_name}'. Choose one of {PIPELINE_NAMES}.")
    if retrieve_k < 1:
        raise ValueError("retrieve_k must be at least 1.")
    if final_k not in K_VALUES:
        raise ValueError(f"final_k must be one of {K_VALUES}.")

    questions = _load_questions(split)
    records = read_chunk_records(collection)
    if not records:
        raise FileNotFoundError(
            f"Collection '{collection}' is empty. Run "
            "`python -m app.rebuild_comparable_collections` first."
        )

    rows = []
    start = time.perf_counter()

    for item in questions:
        docs = retrieve_pipeline_docs(
            pipeline_name,
            item["question"],
            top_k=MAX_K,
            candidate_k=max(retrieve_k, MAX_K),
            collection=collection,
            source_filter=source_filter,
        )

        seen, ranked = set(), []
        for doc in docs:
            chunk_id = get_chunk_id(doc)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            ranked.append(
                (
                    chunk_id,
                    _page_range(doc.metadata),
                    _is_relevant(doc.page_content, doc.metadata, item),
                )
            )

        total_relevant = sum(
            _is_relevant(r["text"], r["metadata"], item) for r in records
        )
        row = _score_question(item, ranked, total_relevant)
        row["question_id"] = item["id"]
        rows.append(row)

    elapsed = time.perf_counter() - start
    summary = _aggregate(rows)
    at_final = summary[final_k]

    return {
        "pipeline": pipeline_name,
        "split": split,
        "questions": len(rows),
        "retrieve_k": retrieve_k,
        "final_k": final_k,
        "collection": collection,
        # Keys the Streamlit UI already reads (evaluated at final_k):
        "mrr_at_5": at_final["mrr"],
        "recall_at_5": at_final["recall"],
        "hit_at_5": at_final["hit"],
        "ndcg_at_5": at_final["ndcg"],
        "precision_at_5": at_final["precision"],
        "summary": {str(k): v for k, v in summary.items()},
        "latency_seconds": elapsed,
        "latency_per_query": elapsed / max(len(rows), 1),
        "rows": [
            {
                "question_id": r["question_id"],
                "rr": r["at_k"][final_k]["mrr"],
                "recall": r["at_k"][final_k]["recall"],
                "first_relevant_rank": r["first_relevant_rank"],
                "relevant_pages": r["relevant_pages"],
                "retrieved_ids": r["retrieved_ids"][:final_k],
            }
            for r in rows
        ],
    }


def calculate_all_pipeline_metrics(
    retrieve_k: int = 30,
    final_k: int = 5,
    collection: str = EVAL_COLLECTION,
    split: str = "all",
) -> dict[str, Any]:
    metrics = {
        name: calculate_pipeline_metrics(
            name,
            retrieve_k=retrieve_k,
            final_k=final_k,
            collection=collection,
            split=split,
        )
        for name in PIPELINE_NAMES
    }
    return {**metrics["D"], "pipelines": metrics}


def save_per_question_results(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")


def print_table(results: dict[str, dict[str, Any]], k: int = 5) -> None:
    header = f"{'Pipe':<5}{'MRR':>8}{'Recall':>8}{'Hit':>8}{'nDCG':>8}{'Prec':>8}{'s/query':>9}"
    print(header)
    for name, res in results.items():
        s = res["summary"][str(k)]
        print(
            f"{name:<5}{s['mrr']:>8.3f}{s['recall']:>8.3f}{s['hit']:>8.3f}"
            f"{s['ndcg']:>8.3f}{s['precision']:>8.3f}{res['latency_per_query']:>9.2f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="all")
    parser.add_argument("--retrieve-k", type=int, default=30)
    parser.add_argument("--final-k", type=int, default=5, choices=K_VALUES)
    args = parser.parse_args()

    out = {
        name: calculate_pipeline_metrics(
            name, retrieve_k=args.retrieve_k, final_k=args.final_k, split=args.split
        )
        for name in PIPELINE_NAMES
    }
    print(f"split={args.split} retrieve_k={args.retrieve_k} @k={args.final_k}")
    print_table(out, k=args.final_k)
    save_per_question_results(
        {"pipelines": out}, Path(__file__).parent / "data" / f"metrics_{args.split}.json"
    )
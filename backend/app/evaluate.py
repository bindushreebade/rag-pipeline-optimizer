from .pipelines import pipeline_a, pipeline_d
from .eval_questions import QUESTIONS

def run_eval():
    results = []

    for q in QUESTIONS:
        base = pipeline_a(q)
        opt = pipeline_d(q)

        results.append({
            "question": q,
            "baseline_answer": base["answer"],
            "baseline_docs": base["docs"],
            "optimized_answer": opt["answer"],
            "optimized_docs": opt["docs"],
        })

    return results

if __name__ == "__main__":
    data = run_eval()
    for d in data:
        print("\nQUESTION:", d["question"])
        print("\nBASELINE:", d["baseline_answer"])
        print("\nOPTIMIZED:", d["optimized_answer"])

"""Choose chunk size and candidate-K empirically, then report on held-out data.

    python -m app.sweep
    python -m app.sweep --chunk-sizes 600 1000 1500 --candidate-ks 20 30

For every (chunk size, candidate K) it evaluates each pipeline on the DEV
split (odd question numbers), picks the best setting per pipeline, then
re-evaluates only that setting on the TEST split (even question numbers).
Quote the TEST numbers on a resume: they were not used for tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import calculate_pipeline_metrics, print_table
from .pipelines import PIPELINE_NAMES
from .rebuild_comparable_collections import rebuild

OUTPUT = Path(__file__).parent / "data" / "sweep_results.json"
FINAL_K = 5


def score(result: dict) -> float:
    """Selection criterion on DEV: MRR@5 + Recall@5."""
    return result["mrr_at_5"] + result["recall_at_5"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[500, 750, 1000, 1500, 2000])
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument("--candidate-ks", type=int, nargs="+", default=[10, 20, 30, 50])
    args = parser.parse_args()

    grid = []  # every evaluated configuration on DEV
    for chunk_size in args.chunk_sizes:
        overlap = int(chunk_size * args.overlap_ratio)
        collection = f"sweep_cs{chunk_size}"
        n_chunks = rebuild(collection, chunk_size, overlap)

        for name in PIPELINE_NAMES:
            # Pipeline A ignores candidate_k, so evaluate it once.
            ks = args.candidate_ks[:1] if name == "A" else args.candidate_ks
            for candidate_k in ks:
                res = calculate_pipeline_metrics(
                    name,
                    retrieve_k=candidate_k,
                    final_k=FINAL_K,
                    collection=collection,
                    split="dev",
                )
                grid.append(
                    {
                        "pipeline": name,
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "candidate_k": candidate_k,
                        "n_chunks": n_chunks,
                        "dev": res["summary"][str(FINAL_K)],
                        "dev_score": score(res),
                    }
                )
                print(
                    f"{name} cs={chunk_size:<5} cand={candidate_k:<3} "
                    f"dev MRR@5={res['mrr_at_5']:.3f} R@5={res['recall_at_5']:.3f}"
                )

    best = {}
    for name in PIPELINE_NAMES:
        best[name] = max((g for g in grid if g["pipeline"] == name), key=lambda g: g["dev_score"])

    print("\nBest DEV configuration per pipeline:")
    for name, cfg in best.items():
        print(f"  {name}: chunk_size={cfg['chunk_size']} overlap={cfg['overlap']} candidate_k={cfg['candidate_k']}")

    test_results = {}
    for name, cfg in best.items():
        test_results[name] = calculate_pipeline_metrics(
            name,
            retrieve_k=cfg["candidate_k"],
            final_k=FINAL_K,
            collection=f"sweep_cs{cfg['chunk_size']}",
            split="test",
        )

    print(f"\nHELD-OUT TEST split @k={FINAL_K} (best dev config per pipeline):")
    print_table(test_results, k=FINAL_K)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "grid_dev": grid,
                "best_dev_config": best,
                "test": {n: {k: v for k, v in r.items() if k != "rows"} for n, r in test_results.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    top = best["D"]
    print(
        f"\nSaved {OUTPUT}\nSet in config.py / .env -> CHUNK_SIZE={top['chunk_size']}, "
        f"CHUNK_OVERLAP={top['overlap']}, CANDIDATE_K={top['candidate_k']}, then run "
        "`python -m app.rebuild_comparable_collections`."
    )


if __name__ == "__main__":
    main()
"""Stage 5: evaluate retrieval on its own, before any LLM is involved.

Run: `python -m training.eval_retrieval`

Why separate retrieval evaluation matters: when a RAG answer is wrong there are
two possible culprits, and end-to-end judgement cannot tell you which. If the
right passage was never retrieved, no prompt engineering will fix it. Measure
the retriever first; only then measure generation.

Metrics
-------
recall@k   : share of in-scope queries where at least one relevant document
             appears in the top k. This is the ceiling on answer quality.
MRR        : 1/rank of the first relevant document, averaged. Rewards putting
             the right passage first, which matters because the model reads the
             top passages most attentively.
refusal    : share of deliberately out-of-scope queries whose best score falls
             below the relevance floor, so the router refuses instead of
             answering from the nearest-but-irrelevant policy.

The refusal set is the one people forget. A retriever with perfect recall that
also returns confident nonsense for "what is the capital of France" produces a
banking assistant that answers questions it has no business answering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from app.config import settings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES = REPO_ROOT / "training" / "data" / "retrieval_queries.jsonl"
REPORTS = REPO_ROOT / "reports"

MIN_RECALL_AT_4 = 0.85   # quality gate: retrieval ceiling
MIN_REFUSAL_RATE = 0.75  # quality gate: out-of-scope rejection


def main() -> int:
    index = joblib.load(REPO_ROOT / "models" / "kb_index_v1.joblib")
    rows = [json.loads(line) for line in QUERIES.read_text().splitlines() if line.strip()]
    in_scope = [r for r in rows if r["relevant_docs"]]
    out_scope = [r for r in rows if not r["relevant_docs"]]
    floor = settings.retrieval_min_score

    results = {"k": {}, "queries": []}
    for k in (1, 3, 4, 8):
        hits = 0
        rr_total = 0.0
        for row in in_scope:
            got = index.search(row["query"], top_k=k, topics=None, min_score=0.0)
            docs = [h.chunk.doc_id for h in got]
            relevant = set(row["relevant_docs"])
            first = next((i for i, d in enumerate(docs) if d in relevant), None)
            if first is not None:
                hits += 1
                rr_total += 1.0 / (first + 1)
            if k == 4:
                results["queries"].append(
                    {
                        "query": row["query"],
                        "expected": sorted(relevant),
                        "retrieved_docs": docs,
                        "top_score": got[0].score if got else 0.0,
                        "hit": first is not None,
                    }
                )
        results["k"][f"recall@{k}"] = round(hits / len(in_scope), 4)
        results["k"][f"mrr@{k}"] = round(rr_total / len(in_scope), 4)

    # Does the intent classifier's topic filter actually help retrieval?
    # This is the central claim of the architecture, so it gets measured rather
    # than asserted: same queries, same retriever, topics supplied by the
    # classifier when it is confident.
    try:
        from app.classifier import IntentClassifier  # noqa: PLC0415
        from app.router import load_json_resource  # noqa: PLC0415

        clf = IntentClassifier.load(REPO_ROOT / "models" / "intent_clf_v1.joblib")
        routes = load_json_resource("intent_routes.json")["intents"]
        hits_f = 0
        rr_f = 0.0
        for row in in_scope:
            pred = clf.predict(row["query"])
            topics = routes.get(pred.intent, {}).get("topics") if pred.is_confident else None
            got = index.search(row["query"], top_k=4, topics=topics or None, min_score=0.0)
            docs = [h.chunk.doc_id for h in got]
            first = next((i for i, d in enumerate(docs) if d in set(row["relevant_docs"])), None)
            if first is not None:
                hits_f += 1
                rr_f += 1.0 / (first + 1)
        results["k"]["recall@4_topic_filtered"] = round(hits_f / len(in_scope), 4)
        results["k"]["mrr@4_topic_filtered"] = round(rr_f / len(in_scope), 4)
    except FileNotFoundError:
        results["k"]["recall@4_topic_filtered"] = None

    refused = 0
    for row in out_scope:
        got = index.search(row["query"], top_k=4, topics=None, min_score=floor)
        if not got:
            refused += 1
        else:
            results["queries"].append(
                {
                    "query": row["query"],
                    "expected": [],
                    "retrieved_docs": [h.chunk.doc_id for h in got],
                    "top_score": got[0].score,
                    "hit": False,
                }
            )
    results["n_in_scope"] = len(in_scope)
    results["n_out_of_scope"] = len(out_scope)
    results["relevance_floor"] = floor
    results["refusal_rate_out_of_scope"] = round(refused / len(out_scope), 4)
    results["gate"] = {
        "min_recall@4": MIN_RECALL_AT_4,
        "min_refusal_rate": MIN_REFUSAL_RATE,
        "passed": results["k"]["recall@4"] >= MIN_RECALL_AT_4
        and results["refusal_rate_out_of_scope"] >= MIN_REFUSAL_RATE,
    }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "retrieval_report.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({**results["k"], **{k: v for k, v in results.items() if k != "queries" and k != "k"}}, indent=2))
    misses = [q for q in results["queries"] if q["expected"] and not q["hit"]]
    if misses:
        print("\nmisses@4:")
        for m in misses:
            print(f"  {m['query']!r} -> {m['retrieved_docs']} (expected {m['expected']})")
    return 0 if results["gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

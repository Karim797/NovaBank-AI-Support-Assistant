"""Stage 5: evaluate retrieval on its own, before any LLM is involved.

Run: `python -m training.eval_retrieval`

The evaluation deliberately measures more than one number:
- lexical ceiling recall with no relevance floor
- the production retrieval path (classifier-derived soft topic prior + floor)
- a floor sweep showing the recall/refusal trade-off
- a harder paraphrase subset and a larger out-of-scope set

This avoids the misleading result produced by the older hard topic filter: a
confidently wrong intent could remove the correct document from the candidate
set entirely while the small hand-written seed set still reported 1.00 recall.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from app.classifier import IntentClassifier  # noqa: E402
from app.config import settings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES = REPO_ROOT / "training" / "data" / "retrieval_queries.jsonl"
REPORTS = REPO_ROOT / "reports"
ROUTES = REPO_ROOT / "src" / "app" / "resources" / "intent_routes.json"

# These gates are tied to the harder 50-query in-scope set, not the original
# corpus-derived seed set. They are regression floors, not marketing targets.
MIN_PRODUCTION_RECALL_AT_4 = 0.65
MIN_REFUSAL_RATE = 0.70
MAX_TOPIC_PRIOR_RECALL_DROP = 0.03
FLOOR_SWEEP = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

# Backwards-compatible import name used by older tests/docs.
MIN_RECALL_AT_4 = MIN_PRODUCTION_RECALL_AT_4


def _docs(hits) -> list[str]:
    return [h.chunk.doc_id for h in hits]


def _first_relevant(docs: list[str], relevant_docs: list[str]) -> int | None:
    relevant = set(relevant_docs)
    return next((i for i, doc_id in enumerate(docs) if doc_id in relevant), None)


def _metrics(
    index,
    rows: list[dict],
    *,
    k: int,
    min_score: float,
    topics_for: Callable[[dict], list[str] | None] | None = None,
) -> tuple[float, float, list[dict]]:
    hits_n = 0
    rr_total = 0.0
    details: list[dict] = []
    for row in rows:
        topics = topics_for(row) if topics_for else None
        got = index.search(
            row["query"],
            top_k=k,
            topics=topics,
            min_score=min_score,
        )
        docs = _docs(got)
        first = _first_relevant(docs, row["relevant_docs"])
        if first is not None:
            hits_n += 1
            rr_total += 1.0 / (first + 1)
        details.append(
            {
                "query": row["query"],
                "kind": row.get("kind", "seed"),
                "expected": sorted(row["relevant_docs"]),
                "retrieved_docs": docs,
                "top_score": got[0].score if got else 0.0,
                "hit": first is not None,
                "topics": topics or [],
            }
        )
    n = len(rows) or 1
    return hits_n / n, rr_total / n, details


def _refusal_rate(index, rows: list[dict], *, floor: float, topics_for) -> float:
    refused = 0
    for row in rows:
        got = index.search(
            row["query"],
            top_k=4,
            topics=topics_for(row),
            min_score=floor,
        )
        if not got:
            refused += 1
    return refused / (len(rows) or 1)


def main() -> int:
    index = joblib.load(REPO_ROOT / "models" / "kb_index_v1.joblib")
    clf = IntentClassifier.load(REPO_ROOT / "models" / "intent_clf_v1.joblib")
    routes = json.loads(ROUTES.read_text(encoding="utf-8"))["intents"]
    rows = [json.loads(line) for line in QUERIES.read_text().splitlines() if line.strip()]
    in_scope = [r for r in rows if r["relevant_docs"]]
    out_scope = [r for r in rows if not r["relevant_docs"]]
    paraphrases = [r for r in in_scope if r.get("kind") == "paraphrase"]
    floor = settings.retrieval_min_score

    def topics_for(row: dict) -> list[str] | None:
        pred = clf.predict(row["query"])
        if not pred.is_confident:
            return None
        topics = routes.get(pred.intent, {}).get("topics") or []
        return list(topics) or None

    results: dict = {"k": {}, "queries": []}

    # Lexical ceiling: no score floor, no classifier signal.
    for k in (1, 3, 4, 8):
        recall, mrr, details = _metrics(index, in_scope, k=k, min_score=0.0)
        results["k"][f"recall@{k}"] = round(recall, 4)
        results["k"][f"mrr@{k}"] = round(mrr, 4)
        if k == 4:
            results["queries"].extend(details)

    prior_recall, prior_mrr, _ = _metrics(
        index,
        in_scope,
        k=4,
        min_score=0.0,
        topics_for=topics_for,
    )
    results["k"]["recall@4_topic_prior"] = round(prior_recall, 4)
    results["k"]["mrr@4_topic_prior"] = round(prior_mrr, 4)

    production_recall, production_mrr, production_details = _metrics(
        index,
        in_scope,
        k=4,
        min_score=floor,
        topics_for=topics_for,
    )
    results["production_recall@4"] = round(production_recall, 4)
    results["production_mrr@4"] = round(production_mrr, 4)

    if paraphrases:
        p_recall, p_mrr, _ = _metrics(
            index,
            paraphrases,
            k=4,
            min_score=floor,
            topics_for=topics_for,
        )
        results["paraphrase_recall@4"] = round(p_recall, 4)
        results["paraphrase_mrr@4"] = round(p_mrr, 4)

    results["refusal_rate_out_of_scope"] = round(
        _refusal_rate(index, out_scope, floor=floor, topics_for=topics_for), 4
    )

    results["floor_sweep"] = []
    for candidate_floor in FLOOR_SWEEP:
        r, _, _ = _metrics(
            index,
            in_scope,
            k=4,
            min_score=candidate_floor,
            topics_for=topics_for,
        )
        refusal = _refusal_rate(
            index, out_scope, floor=candidate_floor, topics_for=topics_for
        )
        results["floor_sweep"].append(
            {
                "floor": candidate_floor,
                "in_scope_recall@4": round(r, 4),
                "out_of_scope_refusal": round(refusal, 4),
            }
        )

    results["n_in_scope"] = len(in_scope)
    results["n_paraphrase"] = len(paraphrases)
    results["n_out_of_scope"] = len(out_scope)
    results["relevance_floor"] = floor
    results["topic_prior_recall_delta_vs_unfiltered"] = round(
        prior_recall - results["k"]["recall@4"], 4
    )

    prior_drop = results["k"]["recall@4"] - prior_recall
    results["gate"] = {
        "min_production_recall@4": MIN_PRODUCTION_RECALL_AT_4,
        "min_refusal_rate": MIN_REFUSAL_RATE,
        "max_topic_prior_recall_drop": MAX_TOPIC_PRIOR_RECALL_DROP,
        "passed": production_recall >= MIN_PRODUCTION_RECALL_AT_4
        and results["refusal_rate_out_of_scope"] >= MIN_REFUSAL_RATE
        and prior_drop <= MAX_TOPIC_PRIOR_RECALL_DROP,
    }

    # Keep production-path misses separate from the lexical ceiling details.
    results["production_misses"] = [d for d in production_details if not d["hit"]]

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "retrieval_report.json").write_text(json.dumps(results, indent=2))

    summary = {
        **results["k"],
        "production_recall@4": results["production_recall@4"],
        "production_mrr@4": results["production_mrr@4"],
        "paraphrase_recall@4": results.get("paraphrase_recall@4"),
        "refusal_rate_out_of_scope": results["refusal_rate_out_of_scope"],
        "n_in_scope": results["n_in_scope"],
        "n_paraphrase": results["n_paraphrase"],
        "n_out_of_scope": results["n_out_of_scope"],
        "relevance_floor": floor,
        "gate": results["gate"],
    }
    print(json.dumps(summary, indent=2))

    if results["production_misses"]:
        print("\nproduction misses@4:")
        for miss in results["production_misses"]:
            print(
                f"  {miss['query']!r} -> {miss['retrieved_docs']} "
                f"(expected {miss['expected']}, topics={miss['topics']})"
            )

    return 0 if results["gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

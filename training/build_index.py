"""Stage 4: build the retrieval index artifact from the knowledge base.

Run: `python -m training.build_index [--dry-run]`

The index is treated exactly like the model: a versioned, hashed artifact built
by a script, not something the API constructs at startup. Reasons: startup must
be fast and deterministic; two replicas must serve byte-identical indexes; and
the corpus hash in the artifact lets you prove which document version produced a
given citation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.kb import corpus_sha256, iter_kb_documents, load_corpus  # noqa: E402
from app.retrieval import HybridRetriever, KnowledgeIndex  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = REPO_ROOT / "knowledge_base"
MODELS = REPO_ROOT / "models"
REPORTS = REPO_ROOT / "reports"
INDEX_VERSION = "kb-index-v1"

# Backward-compatible name used by audit/tests and older documentation.
corpus_hash = corpus_sha256


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print chunks, write nothing")
    args = ap.parse_args()

    documents = iter_kb_documents(KB_DIR)
    chunks = load_corpus(KB_DIR)
    lengths = [len(c.text) for c in chunks]
    stats = {
        "index_version": INDEX_VERSION,
        "n_documents": len(documents),
        "n_chunks": len(chunks),
        "topics": sorted({c.topic for c in chunks}),
        "chunk_chars_min": min(lengths),
        "chunk_chars_max": max(lengths),
        "chunk_chars_mean": round(sum(lengths) / len(lengths), 1),
        "corpus_sha256": corpus_sha256(KB_DIR),
        "built_at": datetime.now(tz=UTC).isoformat(),
    }

    if args.dry_run:
        for c in chunks:
            print(f"{len(c.text):5d}  {c.topic:12s}  {c.chunk_id}")
        print(json.dumps(stats, indent=2))
        return 0

    index = KnowledgeIndex(HybridRetriever(chunks), stats)
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, MODELS / "kb_index_v1.joblib")
    (REPORTS / "index_report.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"artifact -> {MODELS / 'kb_index_v1.joblib'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

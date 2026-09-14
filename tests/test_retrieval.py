"""Chunking and retrieval behaviour."""

import pytest
from conftest import REPO_ROOT, requires_artifacts

from app.kb import MAX_CHARS, load_corpus

pytestmark = requires_artifacts


@pytest.fixture(scope="module")
def chunks():
    return load_corpus(REPO_ROOT / "knowledge_base")


def test_chunk_ids_are_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_no_chunk_exceeds_the_size_budget(chunks):
    oversized = [(c.chunk_id, len(c.text)) for c in chunks if len(c.text) > MAX_CHARS]
    assert not oversized, f"oversized chunks: {oversized}"


def test_no_chunk_is_a_bare_heading(chunks):
    assert all(len(c.text) > 60 for c in chunks)


def test_every_chunk_carries_routing_metadata(chunks):
    assert all(c.doc_id and c.topic and c.section for c in chunks)


def test_retrieval_smoke_finds_the_obvious_document(index):
    hits = index.search("charged twice for the same purchase", top_k=4)
    assert hits, "no hits for an in-corpus question"
    assert any(h.chunk.doc_id == "duplicate-and-unrecognised-charges" for h in hits)


def test_scores_are_ordered_and_bounded(index):
    hits = index.search("atm withdrawal fee", top_k=5)
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    assert [h.rank for h in hits] == sorted(h.rank for h in hits)


def test_wrong_topic_prior_does_not_exclude_the_correct_document(index):
    """This is the regression the old hard filter could not satisfy."""
    hits = index.search(
        "charged twice for the same purchase",
        top_k=4,
        topics=["fees"],
    )
    assert any(h.chunk.doc_id == "duplicate-and-unrecognised-charges" for h in hits)


def test_relevance_floor_rejects_off_topic_questions(index):
    assert index.search("what is the capital of France", top_k=4, min_score=0.15) == []


def test_unknown_topic_is_just_a_noop_prior(index):
    plain = index.search("atm fee", top_k=3)
    unknown = index.search("atm fee", top_k=3, topics=["not-a-topic"])
    assert [h.chunk.chunk_id for h in unknown] == [h.chunk.chunk_id for h in plain]


def test_query_with_no_shared_vocabulary_returns_nothing(index):
    assert index.search("qwertyuiop zxcvbnm", top_k=4) == []


def test_bm25_idf_is_monotonic_in_document_frequency():
    from app.retrieval import bm25_sanity

    assert bm25_sanity(100, 1) > bm25_sanity(100, 50) > bm25_sanity(100, 99)

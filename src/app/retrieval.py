"""Retrieval.

Decision -> Reason -> Alternative -> Trade-off
- Decision: hybrid lexical retrieval — BM25 (Okapi, implemented on a sklearn
  count matrix) fused with TF-IDF cosine via Reciprocal Rank Fusion — plus an
  optional intent-derived topic **prior**.
- Reason: this corpus is ~100 chunks of jargon-dense policy text where the
  discriminating tokens are literal ("chargeback", "SEPA", "disposable",
  "GBP 200"). Lexical retrieval nails exact terminology, has zero model-download
  or GPU cost, is deterministic, and adds ~2 ms of latency. BM25 handles term
  saturation and length normalisation; TF-IDF cosine is more forgiving on short
  queries. Fusing them by rank avoids having to calibrate two different score
  scales.
- Alternative: dense embeddings (sentence-transformers / a hosted embedding API)
  plus a vector store (FAISS, pgvector, Chroma).
- Trade-off: dense retrieval wins on paraphrase and synonym gaps — "money didn't
  land" vs "balance not updated" — which is precisely where this retriever is
  weakest. The topic signal is deliberately a soft ranking bonus, not a hard
  candidate filter: a confidently wrong classifier prediction may reorder the
  candidates, but it cannot make the correct document impossible to retrieve.
  `EmbeddingBackend` below remains a real seam for a future dense arm.

Thresholding note: we rank by fused RRF plus the small topic prior but threshold
on raw TF-IDF cosine, because cosine is on a stable, interpretable 0-1 scale
across queries whereas an RRF score only means something relative to the other
candidates in that query.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from app.kb import Chunk

RRF_K = 60  # standard Reciprocal Rank Fusion constant
BM25_K1 = 1.5
BM25_B = 0.75
# About 15% of one top-ranked RRF arm. Enough to prefer an intent-consistent
# passage in a near tie, too small to swamp strong lexical evidence.
TOPIC_PRIOR_BOOST = 0.0025


class EmbeddingBackend(Protocol):
    """Seam for a future dense arm. Implement and pass to HybridRetriever."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float          # TF-IDF cosine, used for the relevance floor
    fused_score: float    # RRF + optional topic prior, used for ordering
    rank: int


class HybridRetriever:
    """Fitted once at index build time, then pickled into the index artifact."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus = [c.indexable_text for c in chunks]

        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1
        )
        self.tfidf_matrix = self.tfidf.fit_transform(corpus)  # already L2-normalised

        self.count = CountVectorizer(ngram_range=(1, 1), stop_words="english", min_df=1)
        counts = self.count.fit_transform(corpus).tocsc().astype(np.float32)
        self._bm25_matrix, self._bm25_idf = self._precompute_bm25(counts)

    # ---- BM25 -------------------------------------------------------------
    @staticmethod
    def _precompute_bm25(counts: sparse.csc_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Pre-apply the document-side of the BM25 weight so scoring a query is
        a single sparse matrix-vector product instead of a Python loop."""
        counts = counts.tocsr()
        n_docs = counts.shape[0]
        doc_len = np.asarray(counts.sum(axis=1)).ravel()
        avgdl = float(doc_len.mean()) or 1.0
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        weighted = counts.tocoo(copy=True)
        denom = weighted.data + BM25_K1 * (
            1 - BM25_B + BM25_B * doc_len[weighted.row] / avgdl
        )
        weighted.data = weighted.data * (BM25_K1 + 1) / denom
        return weighted.tocsr(), idf

    def _bm25_scores(self, query: str) -> np.ndarray:
        q = self.count.transform([query])
        if q.nnz == 0:
            return np.zeros(self._bm25_matrix.shape[0], dtype=np.float32)
        q_idf = q.multiply(self._bm25_idf).tocsr()
        return np.asarray(self._bm25_matrix.dot(q_idf.T).todense()).ravel()

    def _cosine_scores(self, query: str) -> np.ndarray:
        q = self.tfidf.transform([query])
        return np.asarray(self.tfidf_matrix.dot(q.T).todense()).ravel()

    # ---- search -----------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 4,
        topics: list[str] | None = None,
        topic_boost: float = TOPIC_PRIOR_BOOST,
    ) -> list[RetrievedChunk]:
        """Search the whole corpus, optionally nudging intent-consistent topics.

        `topics` is a prior, never a filter. That distinction is deliberate: if
        the classifier is confidently wrong, the correct document still remains
        eligible to rank in the top-k on lexical evidence alone.
        """
        rows = np.arange(len(self.chunks))
        if rows.size == 0:
            return []

        bm25 = self._bm25_scores(query)
        cosine = self._cosine_scores(query)

        # rank positions (0 = best) for each arm, then Reciprocal Rank Fusion
        bm25_rank = np.empty_like(bm25, dtype=int)
        bm25_rank[np.argsort(-bm25, kind="stable")] = np.arange(rows.size)
        cos_rank = np.empty_like(cosine, dtype=int)
        cos_rank[np.argsort(-cosine, kind="stable")] = np.arange(rows.size)
        fused = 1.0 / (RRF_K + bm25_rank + 1) + 1.0 / (RRF_K + cos_rank + 1)

        if topics and topic_boost > 0:
            topic_set = set(topics)
            fused = fused + np.fromiter(
                (topic_boost if chunk.topic in topic_set else 0.0 for chunk in self.chunks),
                dtype=float,
                count=len(self.chunks),
            )

        order = np.argsort(-fused, kind="stable")[:top_k]
        out: list[RetrievedChunk] = []
        for rank, idx in enumerate(order):
            if bm25[idx] <= 0 and cosine[idx] <= 0:
                continue  # no lexical overlap at all: never surface it
            out.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=round(float(cosine[idx]), 4),
                    fused_score=round(float(fused[idx]), 6),
                    rank=rank,
                )
            )
        return out


class KnowledgeIndex:
    """What gets pickled: retriever + corpus + build metadata. Never a bare
    vectorizer — inference needs the chunk texts and the metadata that says which
    corpus version produced them."""

    def __init__(self, retriever: HybridRetriever, metadata: dict) -> None:
        self.retriever = retriever
        self.metadata = metadata

    @property
    def version(self) -> str:
        return str(self.metadata.get("index_version", "unknown"))

    @property
    def n_chunks(self) -> int:
        return len(self.retriever.chunks)

    def search(
        self,
        query: str,
        top_k: int = 4,
        topics: list[str] | None = None,
        min_score: float = 0.0,
        topic_boost: float = TOPIC_PRIOR_BOOST,
    ) -> list[RetrievedChunk]:
        hits = self.retriever.search(
            query, top_k=top_k, topics=topics, topic_boost=topic_boost
        )
        return [h for h in hits if h.score >= min_score]


def bm25_sanity(n_docs: int, df: int) -> float:
    """Exposed for the unit test that pins the IDF formula."""
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

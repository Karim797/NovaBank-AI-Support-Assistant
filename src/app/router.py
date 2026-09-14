"""Routing and answer composition — the core of the system.

Design rule: **the LLM decides nothing.** Which path a question takes, which
documents it may see, whether the answer is allowed to be returned, and whether
a human is needed are all decided in ordinary Python here, from a classifier
probability, a retrieval score and a lookup table. The LLM is given a narrow
job — write prose from these passages, cite them — and its output is validated
before it reaches the customer.

Why that matters: an LLM asked "should I escalate this to a human?" gives a
plausible answer with no calibration and no audit trail. A threshold on a
measured probability gives you a number you can tune, a coverage/accuracy curve
you can plot (`training/evaluate.py`), and a reason code you can log.

The decision table
------------------
confident intent + deterministic policy  -> fixed compliance text, no LLM
confident intent + rag                   -> retrieval filtered to the intent's topics
low confidence                           -> retrieval over the whole KB (no filter)
best retrieval score < floor             -> refuse, escalate
LLM invalid/uncited after one repair     -> refuse, escalate
LLM provider down                        -> refuse, escalate

Every branch is a `RouteDecision` enum value that ends up in the response, the
log line and the feedback table.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.classifier import IntentClassifier
from app.config import Settings
from app.llm import LLMClient, LLMError
from app.prompts import (
    PROMPT_VERSION,
    REFUSAL_LLM_DOWN,
    REFUSAL_NO_CONTEXT,
    REPAIR_SUFFIX,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    format_context,
)
from app.retrieval import KnowledgeIndex, RetrievedChunk
from app.schemas import GroundedAnswer, RouteDecision, Source

logger = logging.getLogger(__name__)

RESOURCES = Path(__file__).parent / "resources"

# Patterns we refuse to store or log. Not a complete DLP solution — a deliberate,
# documented minimum for an MVP that must not put card data in a database.
_PAN = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_CVV = re.compile(r"\b(?:cvv|cvc|security code)\D{0,10}\d{3,4}\b", re.I)
_PIN = re.compile(r"\b(?:pin|passcode|otp|one[- ]time code)\D{0,10}\d{4,8}\b", re.I)


def redact(text: str) -> str:
    text = _PAN.sub("[REDACTED_CARD_NUMBER]", text)
    text = _CVV.sub("[REDACTED_CVV]", text)
    return _PIN.sub("[REDACTED_SECRET]", text)


@dataclass
class AnswerResult:
    answer: str
    intent: str
    intent_confidence: float
    intent_is_confident: bool
    route: RouteDecision
    sources: list[Source]
    needs_human_escalation: bool
    llm_model: str
    prompt_version: str = PROMPT_VERSION
    retrieval_top_score: float = 0.0
    timings_ms: dict[str, int] = field(default_factory=dict)


def load_json_resource(name: str) -> dict[str, Any]:
    return json.loads((RESOURCES / name).read_text(encoding="utf-8"))


class SupportRouter:
    def __init__(
        self,
        classifier: IntentClassifier,
        index: KnowledgeIndex,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self.classifier = classifier
        self.index = index
        self.llm = llm
        self.settings = settings
        routes = load_json_resource("intent_routes.json")
        self.intent_routes: dict[str, dict] = routes["intents"]
        self.default_route: dict = routes["default"]
        self.policies: dict[str, dict] = load_json_resource("deterministic_policies.json")[
            "policies"
        ]
        self._chunk_lookup = {c.chunk_id: c for c in index.retriever.chunks}
        self._validate_policy_sources()

    def _validate_policy_sources(self) -> None:
        """Fail loudly at startup if a policy cites a chunk that does not exist.
        A citation that 404s is worse than no citation, and this catches it at
        boot instead of in front of a customer."""
        for pid, policy in self.policies.items():
            unknown = [s for s in policy.get("sources", []) if s not in self._chunk_lookup]
            if unknown:
                raise ValueError(f"policy {pid} cites unknown chunk ids: {unknown}")

    # ------------------------------------------------------------------ main
    def answer(self, question: str) -> AnswerResult:
        timings: dict[str, int] = {}
        question = redact(question.strip())

        t0 = time.perf_counter()
        intent = self.classifier.predict(question, threshold=self.settings.confidence_threshold)
        timings["classify_ms"] = int((time.perf_counter() - t0) * 1000)

        route_cfg = (
            self.intent_routes.get(intent.intent, self.default_route)
            if intent.is_confident
            else self.default_route
        )
        escalate_flag = bool(route_cfg.get("escalate", False))

        # --- branch 1: deterministic policy, no LLM involved ---------------
        if intent.is_confident and route_cfg.get("route") == "deterministic":
            policy = self.policies[route_cfg["policy_id"]]
            return AnswerResult(
                answer=policy["answer"],
                intent=intent.intent,
                intent_confidence=intent.confidence,
                intent_is_confident=True,
                route=RouteDecision.DETERMINISTIC_POLICY,
                sources=[self._to_source(cid, 1.0) for cid in policy["sources"]],
                needs_human_escalation=bool(policy.get("escalate", False)),
                llm_model="deterministic",
                timings_ms=timings,
            )

        # --- retrieval ------------------------------------------------------
        t0 = time.perf_counter()
        topics = list(route_cfg.get("topics") or []) if intent.is_confident else []
        hits = self.index.search(
            question,
            top_k=self.settings.retrieval_top_k,
            topics=topics or None,
            min_score=self.settings.retrieval_min_score,
        )
        used_filter = bool(topics)
        if not hits and topics:
            # The topic filter can be wrong when the classifier is confidently
            # wrong. Widen once rather than refusing on a filter artefact.
            hits = self.index.search(
                question,
                top_k=self.settings.retrieval_top_k,
                topics=None,
                min_score=self.settings.retrieval_min_score,
            )
            used_filter = False
            logger.info("retrieval_filter_widened", extra={"intent": intent.intent})
        timings["retrieve_ms"] = int((time.perf_counter() - t0) * 1000)

        route = RouteDecision.RAG_TOPIC_FILTERED if used_filter else RouteDecision.RAG_UNFILTERED
        top_score = hits[0].score if hits else 0.0

        if not hits:
            return self._refusal(
                REFUSAL_NO_CONTEXT, intent, RouteDecision.NO_RELEVANT_CONTEXT, timings
            )

        # --- generation ------------------------------------------------------
        t0 = time.perf_counter()
        try:
            grounded, llm_model = self._generate_grounded(question, intent, hits)
        except LLMError as exc:
            logger.error("llm_unavailable", extra={"error_type": type(exc).__name__})
            timings["generate_ms"] = int((time.perf_counter() - t0) * 1000)
            return self._refusal(
                REFUSAL_LLM_DOWN, intent, RouteDecision.LLM_UNAVAILABLE, timings
            )
        timings["generate_ms"] = int((time.perf_counter() - t0) * 1000)

        if grounded is None:
            return self._refusal(
                REFUSAL_NO_CONTEXT, intent, RouteDecision.UNGROUNDED_ANSWER, timings, llm_model
            )
        if grounded.insufficient_context or not grounded.answer.strip():
            return self._refusal(
                REFUSAL_NO_CONTEXT, intent, RouteDecision.NO_RELEVANT_CONTEXT, timings, llm_model
            )

        allowed = {h.chunk.chunk_id: h for h in hits}
        valid_ids = [cid for cid in grounded.cited if cid in allowed]
        hallucinated = [cid for cid in grounded.cited if cid not in allowed]
        if hallucinated:
            logger.warning(
                "citation_not_in_context",
                extra={"invalid_citations": hallucinated, "intent": intent.intent},
            )
        if not valid_ids:
            # An answer that cites nothing retrievable is not grounded, whatever
            # it says. Refuse rather than ship an unverifiable banking answer.
            return self._refusal(
                REFUSAL_NO_CONTEXT, intent, RouteDecision.UNGROUNDED_ANSWER, timings, llm_model
            )

        return AnswerResult(
            answer=redact(grounded.answer.strip()),
            intent=intent.intent,
            intent_confidence=intent.confidence,
            intent_is_confident=intent.is_confident,
            route=route,
            sources=[self._hit_to_source(allowed[cid]) for cid in valid_ids],
            needs_human_escalation=escalate_flag,
            llm_model=llm_model,
            retrieval_top_score=top_score,
            timings_ms=timings,
        )

    # ------------------------------------------------------------- internals
    def _generate_grounded(
        self, question: str, intent, hits: list[RetrievedChunk]
    ) -> tuple[GroundedAnswer | None, str]:
        context = format_context(
            [(h.chunk.chunk_id, f"{h.chunk.title} — {h.chunk.section}", h.chunk.text) for h in hits]
        )
        user = USER_TEMPLATE.format(
            context=context,
            intent=intent.intent,
            confidence=intent.confidence,
            question=question,
        )
        response = self.llm.generate(SYSTEM_PROMPT, user)
        parsed = self._parse_grounded(response.text)
        if parsed is None:
            # One repair attempt. Two would double p99 latency for a case that
            # rarely recovers; measure before raising this number.
            logger.warning("llm_output_unparseable_retrying")
            response = self.llm.generate(SYSTEM_PROMPT, user + REPAIR_SUFFIX)
            parsed = self._parse_grounded(response.text)
        return parsed, response.model

    @staticmethod
    def _parse_grounded(text: str) -> GroundedAnswer | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.S)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return GroundedAnswer.model_validate_json(cleaned[start : end + 1])
        except (ValidationError, ValueError):
            return None

    def _refusal(
        self,
        message: str,
        intent,
        route: RouteDecision,
        timings: dict[str, int],
        llm_model: str = "none",
    ) -> AnswerResult:
        return AnswerResult(
            answer=message,
            intent=intent.intent,
            intent_confidence=intent.confidence,
            intent_is_confident=intent.is_confident,
            route=route,
            sources=[],
            needs_human_escalation=True,
            llm_model=llm_model,
            timings_ms=timings,
        )

    def _hit_to_source(self, hit: RetrievedChunk) -> Source:
        return Source(
            chunk_id=hit.chunk.chunk_id,
            doc_id=hit.chunk.doc_id,
            title=hit.chunk.title,
            section=hit.chunk.section,
            score=hit.score,
        )

    def _to_source(self, chunk_id: str, score: float) -> Source:
        c = self._chunk_lookup[chunk_id]
        return Source(
            chunk_id=c.chunk_id, doc_id=c.doc_id, title=c.title, section=c.section, score=score
        )

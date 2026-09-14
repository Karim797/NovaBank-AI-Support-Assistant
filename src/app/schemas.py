"""API and internal contracts.

Everything that crosses a boundary (HTTP in, HTTP out, LLM out) is a Pydantic
model. The LLM's JSON output is validated with the same machinery as user input:
a model that returns a citation to a document that was never retrieved is a
validation failure, not a formatting quirk.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_CHARS = 500


class RouteDecision(StrEnum):
    """Why the answer looks the way it does. Logged, returned, and asserted in tests."""

    DETERMINISTIC_POLICY = "deterministic_policy"      # fixed compliance-approved text
    RAG_TOPIC_FILTERED = "rag_topic_filtered"          # confident intent -> filtered retrieval
    RAG_UNFILTERED = "rag_unfiltered"                  # low confidence -> whole-KB retrieval
    NO_RELEVANT_CONTEXT = "no_relevant_context"        # retrieval below score floor -> refuse
    UNGROUNDED_ANSWER = "ungrounded_answer"            # LLM cited nothing valid -> refuse
    LLM_UNAVAILABLE = "llm_unavailable"                # provider failed -> refuse + escalate


class Source(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    score: float = Field(ge=0.0)


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)
    conversation_id: str | None = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class BatchChatRequest(BaseModel):
    items: list[ChatRequest] = Field(min_length=1, max_length=20)


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    intent: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    intent_is_confident: bool
    route: RouteDecision
    sources: list[Source]
    needs_human_escalation: bool
    model_version: str          # intent classifier artifact version
    llm_model: str              # generation model (or "deterministic"/"extractive-v1")
    prompt_version: str
    latency_ms: int


class IntentPrediction(BaseModel):
    intent: str
    confidence: float
    top_k: list[tuple[str, float]]


class GroundedAnswer(BaseModel):
    """Schema the LLM must return. Anything else is rejected."""

    # `answer` may legitimately be empty when insufficient_context is true;
    # the router turns that into the refusal template rather than rejecting it
    # as malformed output.
    answer: str = ""
    cited: list[str] = Field(default_factory=list)
    insufficient_context: bool = False


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=8, max_length=64)
    helpful: bool
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    status: Literal["recorded"]
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app_version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]
    model_version: str | None = None
    index_version: str | None = None
    corpus_sha256: str | None = None
    llm_provider: str
    confidence_threshold: float | None


class ErrorResponse(BaseModel):
    detail: str
    request_id: str

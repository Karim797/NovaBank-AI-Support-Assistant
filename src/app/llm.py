"""LLM abstraction.

Decision -> Reason -> Alternative -> Trade-off
- Decision: a two-method `LLMClient` interface with two implementations —
  `AnthropicLLM` (real API) and `ExtractiveLLM` (deterministic, offline).
- Reason: the router, the tests and the CI pipeline must be able to run the full
  request path without a paid API key and without network flakiness. Making the
  offline path a *first-class implementation of the same interface*, rather than
  a mock patched into tests, means CI exercises the real code path.
- Alternative: LangChain's LLM wrappers, or mocking `httpx` in tests.
- Trade-off: writing the client by hand costs ~80 lines and loses LangChain's
  ecosystem (callbacks, tracing integrations). For one provider and one call
  shape, hand-rolling is fewer moving parts and no version churn. If a second
  provider and tool-calling arrive, revisit.

Be honest about what ExtractiveLLM is: it is *not* a language model. It selects
the sentences from the retrieved passages that best overlap the question. It
keeps the demo and the test suite runnable and gives a sane degraded mode, but
its answers are stitched policy text, not fluent replies. `llm_model` in the
response says `extractive-v1` so nobody mistakes one for the other.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from app.config import Settings
from app.prompts import PROMPT_VERSION

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_version: str = PROMPT_VERSION
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMClient(ABC):
    name: str

    @abstractmethod
    def generate(self, system: str, user: str) -> LLMResponse: ...

    def healthy(self) -> bool:
        return True


class AnthropicLLM(LLMClient):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise LLMError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty")
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.max_retries = settings.llm_max_retries
        self._client = httpx.Client(
            base_url=settings.anthropic_base_url,
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=5.0),
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

    def generate(self, system: str, user: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,  # support answers should be reproducible
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.post("/v1/messages", json=payload)
                if r.status_code in RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"retryable status {r.status_code}", request=r.request, response=r
                    )
                r.raise_for_status()
                data = r.json()
                text = "".join(b.get("text", "") for b in data.get("content", []))
                usage = data.get("usage", {})
                return LLMResponse(
                    text=text,
                    model=data.get("model", self.model),
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                )
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in RETRYABLE_STATUS:
                    raise LLMError(f"non-retryable LLM error {status}") from exc
                last_error = exc
                if attempt < self.max_retries:
                    # exponential backoff with jitter: never hammer a struggling API
                    sleep_s = (2**attempt) * 0.5 + random.uniform(0, 0.25)
                    logger.warning(
                        "llm_retry", extra={"attempt": attempt + 1, "sleep_s": round(sleep_s, 2)}
                    )
                    time.sleep(sleep_s)
        raise LLMError(f"LLM unavailable after {self.max_retries + 1} attempts: {last_error}")

    def healthy(self) -> bool:
        return True


class ExtractiveLLM(LLMClient):
    """Deterministic, offline, no network. Selects sentences from the context."""

    name = "extractive"
    MODEL_ID = "extractive-v1"
    _STOP = {
        "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "i", "my", "me",
        "you", "your", "it", "to", "of", "in", "on", "for", "and", "or", "why", "how",
        "what", "when", "can", "not", "no", "be", "been", "have", "has", "with", "that",
        "this", "at", "as", "if", "but", "so", "there", "they", "we", "will", "would",
    }

    def generate(self, system: str, user: str) -> LLMResponse:
        question, passages = self._parse_user_prompt(user)
        q_tokens = self._tokens(question)
        scored: list[tuple[float, str, str]] = []
        for chunk_id, text in passages:
            for sentence in self._sentences(text):
                s_tokens = self._tokens(sentence)
                if not s_tokens:
                    continue
                overlap = len(q_tokens & s_tokens) / (len(q_tokens) or 1)
                scored.append((overlap, sentence, chunk_id))
        scored.sort(key=lambda t: -t[0])
        picked = [s for s in scored[:4] if s[0] > 0]

        if not picked:
            body = {
                "answer": "",
                "cited": [],
                "insufficient_context": True,
            }
        else:
            seen: list[str] = []
            cited: list[str] = []
            for _, sentence, chunk_id in picked:
                if sentence not in seen:
                    seen.append(sentence)
                if chunk_id not in cited:
                    cited.append(chunk_id)
            body = {
                "answer": " ".join(seen),
                "cited": cited,
                "insufficient_context": False,
            }
        return LLMResponse(text=json.dumps(body), model=self.MODEL_ID)

    # -- helpers -----------------------------------------------------------
    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in cls._STOP}

    @staticmethod
    def _sentences(text: str) -> list[str]:
        flat = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if len(s.strip()) > 25]

    _PASSAGE_RE = re.compile(
        r"\[\d+\] id: (?P<id>\S+)\nsection: [^\n]*\n"
        r"(?P<body>.*?)(?=\n\n\[\d+\] id: |\n\nDetected intent:|\Z)",
        re.S,
    )

    @classmethod
    def _parse_user_prompt(cls, user: str) -> tuple[str, list[tuple[str, str]]]:
        m = re.search(r"Customer question:\s*(.+?)(?:\n|$)", user, flags=re.S)
        question = m.group(1).strip() if m else ""
        passages = [
            (mt.group("id"), mt.group("body").strip()) for mt in cls._PASSAGE_RE.finditer(user)
        ]
        return question, passages


def build_llm(settings: Settings) -> LLMClient:
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        return AnthropicLLM(settings)
    if provider == "extractive":
        return ExtractiveLLM()
    raise LLMError(f"unknown LLM_PROVIDER={settings.llm_provider!r}")

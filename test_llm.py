"""LLM client behaviour: offline determinism and network failure handling."""

import httpx
import pytest

from app.config import Settings
from app.llm import AnthropicLLM, ExtractiveLLM, LLMError, build_llm
from app.prompts import SYSTEM_PROMPT, USER_TEMPLATE, format_context
from app.schemas import GroundedAnswer


def _prompt() -> str:
    context = format_context(
        [("fees-and-charges#fee-schedule", "Fees — schedule",
          "The ATM withdrawal fee is 2% of the amount above the free allowance, minimum GBP 1.")]
    )
    return USER_TEMPLATE.format(
        context=context, intent="cash_withdrawal_charge", confidence=0.9,
        question="what is the atm withdrawal fee",
    )


def test_extractive_llm_returns_schema_valid_json():
    out = ExtractiveLLM().generate(SYSTEM_PROMPT, _prompt())
    parsed = GroundedAnswer.model_validate_json(out.text)
    assert parsed.cited == ["fees-and-charges#fee-schedule"]
    assert "2%" in parsed.answer


def test_extractive_llm_is_deterministic():
    a = ExtractiveLLM().generate(SYSTEM_PROMPT, _prompt()).text
    b = ExtractiveLLM().generate(SYSTEM_PROMPT, _prompt()).text
    assert a == b


def test_extractive_llm_flags_insufficient_context():
    prompt = USER_TEMPLATE.format(
        context=format_context([("a#b", "Unrelated", "Cheques cannot be paid in.")]),
        intent="unknown", confidence=0.1, question="zzzz qqqq",
    )
    parsed = GroundedAnswer.model_validate_json(ExtractiveLLM().generate(SYSTEM_PROMPT, prompt).text)
    assert parsed.insufficient_context


def test_anthropic_client_requires_a_key():
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        build_llm(Settings(llm_provider="anthropic", anthropic_api_key=""))


def test_unknown_provider_fails_fast():
    with pytest.raises(LLMError, match="unknown LLM_PROVIDER"):
        build_llm(Settings(llm_provider="gpt-9000"))


def test_retryable_status_is_retried_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "overloaded"})
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": '{"answer":"ok","cited":[]}'}],
                  "model": "test-model", "usage": {"input_tokens": 10, "output_tokens": 5}},
        )

    monkeypatch.setattr("time.sleep", lambda *_: None)  # keep the test fast
    client = AnthropicLLM(Settings(llm_provider="anthropic", anthropic_api_key="k", llm_max_retries=2))
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    out = client.generate("sys", "user")
    assert calls["n"] == 2
    assert out.model == "test-model" and out.input_tokens == 10


def test_non_retryable_status_fails_immediately():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    client = AnthropicLLM(Settings(llm_provider="anthropic", anthropic_api_key="k"))
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    with pytest.raises(LLMError, match="non-retryable"):
        client.generate("sys", "user")


def test_exhausted_retries_raise_llm_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = AnthropicLLM(Settings(llm_provider="anthropic", anthropic_api_key="k", llm_max_retries=1))
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    with pytest.raises(LLMError, match="unavailable after 2 attempts"):
        client.generate("sys", "user")

"""Contract tests: what the API accepts and rejects, and what the LLM may return."""

import pytest
from pydantic import ValidationError

from app.schemas import BatchChatRequest, ChatRequest, FeedbackRequest, GroundedAnswer


def test_valid_question_is_trimmed():
    assert ChatRequest(question="  my card was charged twice  ").question == (
        "my card was charged twice"
    )


@pytest.mark.parametrize(
    "bad",
    ["", "  ", "hi", "x" * 501],  # blank, whitespace, too short, over the char cap
)
def test_invalid_questions_rejected(bad):
    with pytest.raises(ValidationError):
        ChatRequest(question=bad)


def test_batch_size_is_capped():
    with pytest.raises(ValidationError):
        BatchChatRequest(items=[ChatRequest(question="lost my card")] * 21)


def test_feedback_requires_boolean_verdict():
    with pytest.raises(ValidationError):
        FeedbackRequest(request_id="abcd1234", helpful="maybe")


def test_grounded_answer_allows_empty_answer_when_insufficient():
    parsed = GroundedAnswer.model_validate_json(
        '{"answer": "", "cited": [], "insufficient_context": true}'
    )
    assert parsed.insufficient_context and parsed.cited == []


def test_grounded_answer_rejects_wrong_types():
    with pytest.raises(ValidationError):
        GroundedAnswer.model_validate_json('{"answer": "x", "cited": "not-a-list"}')

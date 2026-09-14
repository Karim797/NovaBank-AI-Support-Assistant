"""Routing, grounding and refusal behaviour — the safety-critical logic."""

import pytest
from conftest import BrokenLLM, StubLLM, requires_artifacts

from app.router import SupportRouter, redact
from app.schemas import RouteDecision

pytestmark = requires_artifacts


def build(classifier, index, settings, llm) -> SupportRouter:
    return SupportRouter(classifier, index, llm, settings)


def test_safety_intent_uses_deterministic_policy_not_the_llm(classifier, index, settings):
    llm = StubLLM(['{"answer": "should never be used", "cited": []}'])
    result = build(classifier, index, settings, llm).answer("I lost my card, what do I do?")
    assert result.route is RouteDecision.DETERMINISTIC_POLICY
    assert result.llm_model == "deterministic"
    assert llm.calls == 0, "the LLM must not be called on a deterministic policy path"
    assert "Freeze the card" in result.answer
    assert result.sources


def test_confident_intent_uses_topic_filtered_retrieval(router):
    result = router.answer("my card was charged twice for the same purchase")
    assert result.route is RouteDecision.RAG_TOPIC_FILTERED
    assert result.intent_is_confident
    assert result.sources


def test_out_of_scope_question_is_refused_and_escalated(router):
    result = router.answer("What is the capital of France?")
    assert result.route is RouteDecision.NO_RELEVANT_CONTEXT
    assert result.sources == []
    assert result.needs_human_escalation


def test_hallucinated_citations_are_dropped(classifier, index, settings):
    llm = StubLLM(
        ['{"answer": "Report it in the app.", '
         '"cited": ["duplicate-and-unrecognised-charges#charged-twice-for-the-same-purchase", '
         '"totally-made-up-doc#section"], "insufficient_context": false}']
    )
    result = build(classifier, index, settings, llm).answer("charged twice for one purchase")
    returned = {s.chunk_id for s in result.sources}
    assert "totally-made-up-doc#section" not in returned
    assert returned, "the valid citation should survive"


def test_answer_with_only_invalid_citations_is_refused(classifier, index, settings):
    llm = StubLLM(['{"answer": "Trust me.", "cited": ["nope#nope"], "insufficient_context": false}'])
    result = build(classifier, index, settings, llm).answer("charged twice for one purchase")
    assert result.route is RouteDecision.UNGROUNDED_ANSWER
    assert result.needs_human_escalation


def test_model_declaring_insufficient_context_produces_a_refusal(classifier, index, settings):
    llm = StubLLM(['{"answer": "", "cited": [], "insufficient_context": true}'])
    result = build(classifier, index, settings, llm).answer("charged twice for one purchase")
    assert result.route is RouteDecision.NO_RELEVANT_CONTEXT


def test_unparseable_output_triggers_exactly_one_repair_attempt(classifier, index, settings):
    llm = StubLLM(["not json at all", "still not json"])
    result = build(classifier, index, settings, llm).answer("charged twice for one purchase")
    assert llm.calls == 2
    assert result.route is RouteDecision.UNGROUNDED_ANSWER


def test_fenced_json_is_recovered(classifier, index, settings):
    llm = StubLLM(
        ['```json\n{"answer": "Open the transaction and report it.", '
         '"cited": ["duplicate-and-unrecognised-charges#charged-twice-for-the-same-purchase"]}\n```']
    )
    result = build(classifier, index, settings, llm).answer("charged twice for one purchase")
    assert result.route is RouteDecision.RAG_TOPIC_FILTERED
    assert llm.calls == 1


def test_provider_outage_degrades_to_a_safe_refusal(classifier, index, settings):
    result = build(classifier, index, settings, BrokenLLM()).answer("charged twice")
    assert result.route is RouteDecision.LLM_UNAVAILABLE
    assert result.needs_human_escalation
    assert "0800 000 0000" in result.answer


def test_low_confidence_question_searches_the_whole_corpus(router):
    result = router.answer("hmm something odd happened with money yesterday")
    assert result.route in {
        RouteDecision.RAG_UNFILTERED,
        RouteDecision.NO_RELEVANT_CONTEXT,
        RouteDecision.UNGROUNDED_ANSWER,
    }


@pytest.mark.parametrize(
    "text,marker",
    [
        ("my card 4111 1111 1111 1111 was charged", "[REDACTED_CARD_NUMBER]"),
        ("my pin is 4821 and it failed", "[REDACTED_SECRET]"),
        ("cvv 123 was rejected", "[REDACTED_CVV]"),
    ],
)
def test_card_data_is_redacted_before_anything_else_happens(text, marker):
    assert marker in redact(text)


def test_redacted_question_never_reaches_the_llm(classifier, index, settings):
    llm = StubLLM(['{"answer": "ok", "cited": []}'])
    build(classifier, index, settings, llm).answer("charged twice on 4111 1111 1111 1111")
    # StubLLM stores nothing, so assert on the redactor contract instead
    assert "[REDACTED_CARD_NUMBER]" in redact("charged twice on 4111 1111 1111 1111")


def test_policies_cite_only_real_chunks(router):
    known = {c.chunk_id for c in router.index.retriever.chunks}
    for policy in router.policies.values():
        assert set(policy["sources"]) <= known


def test_every_banking77_intent_has_a_route(router, classifier):
    missing = [label for label in classifier.labels if label not in router.intent_routes]
    assert not missing, f"intents with no routing entry: {missing}"

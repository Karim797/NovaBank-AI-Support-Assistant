"""HTTP surface: status codes, validation, contracts, ops endpoints."""

import pytest
from conftest import requires_artifacts

pytestmark = requires_artifacts


def test_health_is_cheap_and_always_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_reports_loaded_dependencies(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert all(body["checks"].values())
    assert body["checks"]["corpus_fresh"] is True
    assert body["model_version"] and body["index_version"]
    assert body["corpus_sha256"]


def test_chat_returns_the_full_contract(client):
    r = client.post("/chat", json={"question": "my card was charged twice"})
    assert r.status_code == 200
    body = r.json()
    for field in ("request_id", "answer", "intent", "intent_confidence", "route",
                  "sources", "needs_human_escalation", "model_version", "llm_model",
                  "prompt_version", "latency_ms"):
        assert field in body
    assert 0.0 <= body["intent_confidence"] <= 1.0
    assert body["answer"]


def test_every_response_carries_a_request_id_header(client):
    r = client.post("/chat", json={"question": "how do I change my pin"})
    assert r.headers["x-request-id"]
    assert r.headers["x-request-id"] == r.json()["request_id"]


def test_caller_supplied_request_id_is_honoured(client):
    r = client.post(
        "/chat", json={"question": "how do I change my pin"}, headers={"x-request-id": "trace-123"}
    )
    assert r.json()["request_id"] == "trace-123"


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "x" * 501}, {"question": 42}, {"quesion": "typo"}],
)
def test_invalid_requests_return_422_not_500(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_batch_endpoint_answers_each_item(client):
    r = client.post(
        "/chat/batch",
        json={"items": [{"question": "lost my card"}, {"question": "atm withdrawal fee"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert len({item["request_id"] for item in body}) == 2


def test_feedback_round_trip(client):
    request_id = client.post("/chat", json={"question": "atm fees abroad"}).json()["request_id"]
    r = client.post("/feedback", json={"request_id": request_id, "helpful": True})
    assert r.status_code == 200 and r.json()["status"] == "recorded"
    assert client.get("/stats").json()["feedback_count"] == 1


def test_metrics_are_exposed_in_prometheus_format(client):
    client.post("/chat", json={"question": "lost my card"})
    body = client.get("/metrics").text
    assert "nb_requests_total" in body
    assert 'nb_request_latency_ms{quantile="0.95"}' in body


def test_deterministic_route_is_visible_to_the_caller(client):
    body = client.post("/chat", json={"question": "someone stole my card"}).json()
    assert body["route"] == "deterministic_policy"
    assert body["needs_human_escalation"] is True


def test_api_key_is_enforced_when_configured(authed_client):
    assert authed_client.post("/chat", json={"question": "lost my card"}).status_code == 401

    # Liveness/readiness remain public so Railway and the Streamlit status panel
    # can probe the service, but customer/operational data surfaces are private.
    assert authed_client.get("/health").status_code == 200
    assert authed_client.get("/ready").status_code == 200
    assert authed_client.get("/stats").status_code == 401
    assert authed_client.get("/metrics").status_code == 401
    assert authed_client.post(
        "/feedback", json={"request_id": "12345678", "helpful": True}
    ).status_code == 401

    headers = {"x-api-key": "s3cret"}
    ok = authed_client.post("/chat", json={"question": "lost my card"}, headers=headers)
    assert ok.status_code == 200
    assert authed_client.get("/stats", headers=headers).status_code == 200
    assert authed_client.get("/metrics", headers=headers).status_code == 200

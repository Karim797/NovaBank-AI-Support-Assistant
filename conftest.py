"""Shared fixtures.

Tests run against the **real** artifacts (model + index) rather than mocks,
because the failure mode this suite must catch is "the artifact and the code
drifted apart". A mocked classifier would happily pass while the shipped
pipeline was unloadable. Artifacts are produced by `make data train index`
before `make test` — the same order CI uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_PATH = REPO_ROOT / "models" / "intent_clf_v1.joblib"
INDEX_PATH = REPO_ROOT / "models" / "kb_index_v1.joblib"
REPORTS = REPO_ROOT / "reports"

requires_artifacts = pytest.mark.skipif(
    not (MODEL_PATH.exists() and INDEX_PATH.exists()),
    reason="run `make data train index` first",
)


@pytest.fixture(scope="session")
def classifier():
    from app.classifier import IntentClassifier

    return IntentClassifier.load(MODEL_PATH)


@pytest.fixture(scope="session")
def index():
    return joblib.load(INDEX_PATH)


@pytest.fixture(scope="session")
def settings():
    from app.config import Settings

    return Settings(
        model_path=MODEL_PATH,
        index_path=INDEX_PATH,
        llm_provider="extractive",
        feedback_db_url="sqlite:///:memory:",
        api_key="",
    )


@pytest.fixture
def router(classifier, index, settings):
    from app.llm import ExtractiveLLM
    from app.router import SupportRouter

    return SupportRouter(classifier, index, ExtractiveLLM(), settings)


def _configured_client(tmp_path, monkeypatch, api_key: str = ""):
    """Patch the live settings object rather than re-importing the app.

    Reloading `app.*` inside a fixture would create a *second* copy of every
    module, so `RouteDecision.X is RouteDecision.X` would compare enums from
    different class objects and fail in confusing ways. Patching attributes on
    the already-imported settings keeps one module graph for the whole session.
    """
    from fastapi.testclient import TestClient

    from app.config import settings as live_settings
    from app.main import app

    monkeypatch.setattr(live_settings, "model_path", MODEL_PATH)
    monkeypatch.setattr(live_settings, "index_path", INDEX_PATH)
    monkeypatch.setattr(live_settings, "llm_provider", "extractive")
    monkeypatch.setattr(live_settings, "feedback_db_url", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(live_settings, "api_key", api_key)
    return TestClient(app)


@pytest.fixture
def client(tmp_path, monkeypatch):
    with _configured_client(tmp_path, monkeypatch) as c:
        yield c


@pytest.fixture
def authed_client(tmp_path, monkeypatch):
    """Same app, but with an API key configured."""
    with _configured_client(tmp_path, monkeypatch, api_key="s3cret") as c:
        yield c


@pytest.fixture(scope="session")
def eval_report() -> dict:
    path = REPORTS / "eval_report.json"
    if not path.exists():
        pytest.skip("run `python -m training.evaluate` first")
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def retrieval_report() -> dict:
    path = REPORTS / "retrieval_report.json"
    if not path.exists():
        pytest.skip("run `python -m training.eval_retrieval` first")
    return json.loads(path.read_text())


class StubLLM:
    """Returns whatever raw text the test wants, through the real interface."""

    name = "stub"

    def __init__(self, payloads: list[str]) -> None:
        self.payloads = list(payloads)
        self.calls = 0

    def generate(self, system: str, user: str):
        from app.llm import LLMResponse

        self.calls += 1
        text = self.payloads.pop(0) if self.payloads else "{}"
        return LLMResponse(text=text, model="stub-v0")


class BrokenLLM:
    name = "broken"

    def generate(self, system: str, user: str):
        from app.llm import LLMError

        raise LLMError("provider down")

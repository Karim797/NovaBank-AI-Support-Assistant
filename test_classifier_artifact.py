"""The artifact contract: everything production inference needs is in the bundle."""

import joblib
import pytest
from conftest import MODEL_PATH, requires_artifacts

pytestmark = requires_artifacts


def test_bundle_has_every_required_key():
    bundle = joblib.load(MODEL_PATH)
    for key in ("schema_version", "pipeline", "labels", "confidence_threshold",
                "model_version", "metadata"):
        assert key in bundle, f"artifact is missing {key}"


def test_bundle_records_provenance():
    meta = joblib.load(MODEL_PATH)["metadata"]
    for key in ("trained_at", "git_sha", "dataset_sha256_train", "sklearn", "python"):
        assert meta.get(key), f"provenance field {key} is empty"


def test_label_space_is_banking77(classifier):
    assert len(classifier.labels) == 77
    assert len(set(classifier.labels)) == 77


def test_prediction_shape_and_label_validity(classifier):
    result = classifier.predict("my card was charged twice")
    assert result.intent in classifier.labels
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_k) == 3
    assert result.top_k == sorted(result.top_k, key=lambda t: -t[1])


def test_probabilities_sum_to_one(classifier):
    proba = classifier.pipeline.predict_proba(["where is my card"])[0]
    assert proba.sum() == pytest.approx(1.0, abs=1e-6)


def test_threshold_is_a_usable_probability(classifier):
    assert 0.0 < classifier.threshold < 1.0


def test_batch_matches_single_prediction(classifier):
    texts = ["lost my card", "why is my transfer pending", "atm fee"]
    batch = classifier.predict_batch(texts)
    for text, b in zip(texts, batch, strict=True):
        assert b.intent == classifier.predict(text).intent


def test_preprocessing_survives_messy_input(classifier):
    """Real support traffic has typos, emoji and casing chaos; the pipeline must
    not raise on any of it."""
    for text in ["WHERE IS MY CARD???", "cash withdrawl pendng 😩", "  fee?  ", "£200 atm"]:
        assert classifier.predict(text).intent in classifier.labels


def test_rejects_artifact_with_wrong_schema_version():
    from app.classifier import IntentClassifier

    bundle = dict(joblib.load(MODEL_PATH))
    bundle["schema_version"] = 999
    with pytest.raises(ValueError, match="schema"):
        IntentClassifier(bundle)

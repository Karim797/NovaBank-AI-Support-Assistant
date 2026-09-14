"""Intent classifier — inference side.

The training code writes a *bundle*, not a bare estimator. A bare
`joblib.dump(model)` is the single most common junior mistake in this area:
at serving time you then have no idea which label indices map to which intent
names, which threshold the evaluation used, which sklearn version produced the
pickle, or which data it saw. All of that lives in the bundle below, and
`/ready` echoes it so the running configuration is observable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 1
REQUIRED_KEYS = {
    "schema_version", "pipeline", "labels", "confidence_threshold",
    "model_version", "metadata",
}


@dataclass
class IntentResult:
    intent: str
    confidence: float
    is_confident: bool
    top_k: list[tuple[str, float]]


class IntentClassifier:
    def __init__(self, bundle: dict[str, Any]) -> None:
        missing = REQUIRED_KEYS - set(bundle)
        if missing:
            raise ValueError(f"intent artifact is missing keys: {sorted(missing)}")
        if bundle["schema_version"] != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"artifact schema {bundle['schema_version']} != expected "
                f"{ARTIFACT_SCHEMA_VERSION}; retrain or write a migration"
            )
        self.bundle = bundle
        self.pipeline = bundle["pipeline"]
        self.labels: list[str] = list(bundle["labels"])
        self.threshold: float = float(bundle["confidence_threshold"])
        self.model_version: str = str(bundle["model_version"])
        self.metadata: dict = dict(bundle["metadata"])

    @classmethod
    def load(cls, path: Path) -> IntentClassifier:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"intent model not found at {path}. Run `make train` first."
            )
        clf = cls(joblib.load(path))
        logger.info(
            "intent_model_loaded",
            extra={
                "model_version": clf.model_version,
                "n_labels": len(clf.labels),
                "test_macro_f1": clf.metadata.get("test_macro_f1"),
            },
        )
        return clf

    def predict(self, text: str, top_k: int = 3, threshold: float | None = None) -> IntentResult:
        proba = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        order = np.argsort(-proba)[:top_k]
        ranked = [(str(classes[i]), round(float(proba[i]), 4)) for i in order]
        best_label, best_p = ranked[0]
        th = self.threshold if threshold is None else threshold
        return IntentResult(
            intent=best_label,
            confidence=best_p,
            is_confident=best_p >= th,
            top_k=ranked,
        )

    def predict_batch(self, texts: list[str], threshold: float | None = None) -> list[IntentResult]:
        """One vectorisation pass for the whole batch — do not loop `predict`."""
        proba = self.pipeline.predict_proba(texts)
        classes = list(self.pipeline.classes_)
        th = self.threshold if threshold is None else threshold
        results = []
        for row in proba:
            order = np.argsort(-row)[:3]
            ranked = [(str(classes[i]), round(float(row[i]), 4)) for i in order]
            results.append(
                IntentResult(ranked[0][0], ranked[0][1], ranked[0][1] >= th, ranked)
            )
        return results

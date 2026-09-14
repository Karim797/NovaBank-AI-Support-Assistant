"""Stage 3: evaluate the shipped artifact on the official test split.

Run: `python -m training.evaluate`

This is the only place the test set is touched. It writes
reports/eval_report.json (consumed by the CI quality gate) and patches the test
metrics back into the artifact bundle, so a deployed model can always answer
"how good are you and on what data" from `/ready`.

Metric choice: **macro F1**. 77 classes with a 5.3x imbalance between the
largest and smallest; accuracy would let the model earn a good score while
failing the rare intents, and the rare intents here (`card_swallowed`,
`compromised_card`) are exactly the ones with the highest cost of being wrong.
Macro F1 weights every intent equally.

The de-duplicated number matters: the official split contains a handful of test
rows that appear verbatim in train. We report macro F1 with and without them so
the headline number cannot be quietly inflated by memorisation.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
MODELS = REPO_ROOT / "models"
REPORTS = REPO_ROOT / "reports"
ARTIFACT = MODELS / "intent_clf_v1.joblib"

# Quality gate. Observed test macro F1 is ~0.91; the gate sits at 0.85.
# Why 0.85 and not 0.90: the gate exists to catch *breakage* (a corrupted split,
# a dropped feature block, a library upgrade that changes tokenisation), not to
# police normal variance. A gate set one point under the current score fails on
# noise, gets muted, and then protects nothing. 0.85 is comfortably above the
# ~0.80 a word-only TF-IDF baseline reaches, so a real regression still trips it.
MIN_MACRO_F1 = 0.85


def main() -> int:
    if not ARTIFACT.exists():
        print("no artifact - run `python -m training.train_intent`", file=sys.stderr)
        return 1
    bundle = joblib.load(ARTIFACT)
    pipe = bundle["pipeline"]

    test = pd.read_csv(PROCESSED / "test.csv")
    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    seen = set(pd.concat([train.text, val.text]).str.strip().str.lower())

    proba = pipe.predict_proba(test.text)
    classes = np.array(pipe.classes_)
    pred = classes[proba.argmax(1)]
    conf = proba.max(1)
    y = test.category.to_numpy()
    correct = pred == y

    macro_f1 = float(f1_score(y, pred, average="macro"))
    mask_new = ~test.text.str.strip().str.lower().isin(seen).to_numpy()
    macro_f1_dedup = float(f1_score(y[mask_new], pred[mask_new], average="macro"))

    threshold = float(bundle["confidence_threshold"])
    keep = conf >= threshold
    report_dict = classification_report(y, pred, output_dict=True, zero_division=0)
    per_class = sorted(
        (
            {"intent": k, "f1": round(v["f1-score"], 3), "support": int(v["support"])}
            for k, v in report_dict.items()
            if isinstance(v, dict) and k not in {"macro avg", "weighted avg", "accuracy"}
        ),
        key=lambda d: d["f1"],
    )

    # Confusion pairs: which intents does the model mix up, and does it matter?
    cm = confusion_matrix(y, pred, labels=sorted(set(y)))
    label_list = sorted(set(y))
    pairs = Counter()
    for i, true_label in enumerate(label_list):
        for j, pred_label in enumerate(label_list):
            if i != j and cm[i][j] > 0:
                pairs[(true_label, pred_label)] = int(cm[i][j])

    report = {
        "model_version": bundle["model_version"],
        "n_test": int(len(test)),
        "test_macro_f1": round(macro_f1, 4),
        "test_macro_f1_dedup": round(macro_f1_dedup, 4),
        "n_test_dedup": int(mask_new.sum()),
        "test_weighted_f1": round(float(f1_score(y, pred, average="weighted")), 4),
        "test_accuracy": round(float(correct.mean()), 4),
        "confidence_threshold": threshold,
        "coverage_at_threshold": round(float(keep.mean()), 4),
        "accuracy_on_confident": round(float(correct[keep].mean()), 4),
        "accuracy_on_low_confidence": round(float(correct[~keep].mean()), 4)
        if (~keep).any()
        else None,
        "mean_confidence": round(float(conf.mean()), 4),
        "worst_10_intents": per_class[:10],
        "top_confusions": [
            {"true": t, "predicted": p, "count": n} for (t, p), n in pairs.most_common(10)
        ],
        "quality_gate": {
            "metric": "test_macro_f1",
            "minimum": MIN_MACRO_F1,
            "passed": macro_f1 >= MIN_MACRO_F1,
        },
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "eval_report.json").write_text(json.dumps(report, indent=2))
    np.savetxt(REPORTS / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    # Patch test metrics into the shipped bundle so the serving layer can report them.
    bundle["metadata"].update(
        {
            "test_macro_f1": report["test_macro_f1"],
            "test_macro_f1_dedup": report["test_macro_f1_dedup"],
            "test_accuracy": report["test_accuracy"],
            "coverage_at_threshold": report["coverage_at_threshold"],
            "evaluated_on_n": report["n_test"],
        }
    )
    joblib.dump(bundle, ARTIFACT)

    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=2))
    print("\nworst intents:", json.dumps(report["worst_10_intents"][:5], indent=2))
    print("\ntop confusions:", json.dumps(report["top_confusions"][:5], indent=2))
    if not report["quality_gate"]["passed"]:
        print(f"\nQUALITY GATE FAILED: {macro_f1:.4f} < {MIN_MACRO_F1}", file=sys.stderr)
        return 1
    print(f"\nquality gate passed: macro F1 {macro_f1:.4f} >= {MIN_MACRO_F1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

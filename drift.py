"""Drift and quality monitoring.

Run:
    python -m monitoring.drift --build-reference     # once, at release time
    python -m monitoring.drift                       # in a scheduled job

What is actually being monitored
--------------------------------
Two different things, often confused:

**Data drift — P(X) changed.** The questions coming in look different from the
ones the model was evaluated on: a new product launched, a fee changed, a
marketing campaign brought a wave of "how do I open an account". Detected here
with PSI on the confidence distribution and PSI/chi-square on the predicted
intent mix, plus a KS test on confidence.

**Concept drift — P(y|X) changed.** The same question now has a different
correct answer. In this system that happens when a *policy document changes*:
"what is the ATM fee" is unchanged as an input, but the right answer moved. No
input-distribution test can see this. It is caught by the corpus hash changing
and by the feedback signal falling.

**Data drift does not imply the model got worse.** A seasonal spike in
`card_arrival` shifts every input distribution while the model performs exactly
as before. Treat a drift alert as "go and look", never as "the model is broken"
— and never as a trigger for automatic retraining.

Getting real labels
-------------------
Ground truth for a support assistant arrives late and partially:
  - thumbs-down on `/feedback` (weak negative signal),
  - a human agent later re-tagging the conversation (strong label),
  - the customer re-asking the same thing (implicit failure).
All of them arrive with a `request_id`, which is why every prediction is logged
with one. Join `feedback.request_id -> predictions.request_id` and you have a
delayed, biased, but real evaluation set. Bias warning: people who leave
feedback are not a random sample, so treat the rate as a trend, not an accuracy.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

REFERENCE = REPO_ROOT / "monitoring" / "reference_window.json"
REPORTS = REPO_ROOT / "reports"

CONFIDENCE_BINS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
PSI_WARN, PSI_ALERT = 0.10, 0.25          # industry convention: <0.1 stable, >0.25 significant
KS_PVALUE_ALERT = 0.01
LOW_CONF_SHARE_ALERT = 0.25               # share of traffic below the routing threshold
REFUSAL_SHARE_ALERT = 0.20                # share of answers that refused
MIN_SAMPLES = 100                         # below this, every test is noise


def psi(reference: np.ndarray, current: np.ndarray, bins: list[float]) -> float:
    """Population Stability Index over fixed bins.

    Zero counts are floored at a small epsilon: PSI uses a log ratio, and one
    empty bin would otherwise return infinity and page someone at 3am.
    """
    eps = 1e-6
    ref_hist = np.histogram(reference, bins=bins)[0] / max(len(reference), 1)
    cur_hist = np.histogram(current, bins=bins)[0] / max(len(current), 1)
    ref_hist = np.clip(ref_hist, eps, None)
    cur_hist = np.clip(cur_hist, eps, None)
    return float(np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist)))


def categorical_psi(reference: dict[str, float], current: dict[str, float]) -> float:
    eps = 1e-6
    keys = set(reference) | set(current)
    total = 0.0
    for key in keys:
        r = max(reference.get(key, 0.0), eps)
        c = max(current.get(key, 0.0), eps)
        total += (c - r) * np.log(c / r)
    return float(total)


def ks_2sample(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov two-sample test on the confidence distribution.

    Caveat worth knowing in an interview: with a large enough sample, KS finds a
    statistically significant difference that is operationally meaningless. Read
    the statistic (effect size) alongside the p-value, and always alert on PSI
    or the statistic — never on the p-value alone.
    """
    try:
        from scipy import stats

        result = stats.ks_2samp(reference, current)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        return float("nan"), float("nan")


def build_reference() -> dict:
    """Reference = the model's behaviour on the evaluation set. This is the
    distribution the release was signed off against, so it is what production
    should be compared to — not to last week, which drifts with you."""
    import joblib
    import pandas as pd

    bundle = joblib.load(REPO_ROOT / "models" / "intent_clf_v1.joblib")
    pipe = bundle["pipeline"]
    test = pd.read_csv(REPO_ROOT / "data" / "processed" / "test.csv")
    proba = pipe.predict_proba(test.text)
    classes = np.array(pipe.classes_)
    pred = classes[proba.argmax(1)]
    conf = proba.max(1)
    counts = Counter(pred)
    reference = {
        "model_version": bundle["model_version"],
        "n": int(len(test)),
        "confidence": [round(float(c), 4) for c in conf],
        "intent_share": {k: round(v / len(pred), 6) for k, v in counts.items()},
        "low_confidence_share": round(float((conf < bundle["confidence_threshold"]).mean()), 4),
        "confidence_threshold": float(bundle["confidence_threshold"]),
    }
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps(reference, indent=2))
    print(f"reference window -> {REFERENCE} (n={reference['n']})")
    return reference


def load_current(db_path: str, limit: int = 5000) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT intent, intent_confidence, route, needs_escalation, latency_ms, model_version "
        "FROM predictions ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows


def analyse(reference: dict, rows: list[sqlite3.Row]) -> dict:
    conf = np.array([r["intent_confidence"] for r in rows], dtype=float)
    intents = Counter(r["intent"] for r in rows)
    intent_share = {k: v / len(rows) for k, v in intents.items()}
    ref_conf = np.array(reference["confidence"], dtype=float)
    threshold = reference["confidence_threshold"]

    conf_psi = psi(ref_conf, conf, CONFIDENCE_BINS)
    intent_psi = categorical_psi(reference["intent_share"], intent_share)
    ks_stat, ks_p = ks_2sample(ref_conf, conf)
    low_conf = float((conf < threshold).mean())
    refusals = sum(
        1 for r in rows if r["route"] in {"no_relevant_context", "ungrounded_answer", "llm_unavailable"}
    ) / len(rows)
    served_versions = sorted({r["model_version"] for r in rows})

    alerts = []
    if len(rows) < MIN_SAMPLES:
        alerts.append(f"INFO: only {len(rows)} samples; results are noise below {MIN_SAMPLES}")
    if conf_psi > PSI_ALERT:
        alerts.append(f"ALERT: confidence PSI {conf_psi:.3f} > {PSI_ALERT}")
    elif conf_psi > PSI_WARN:
        alerts.append(f"WARN: confidence PSI {conf_psi:.3f} > {PSI_WARN}")
    if intent_psi > PSI_ALERT:
        alerts.append(f"ALERT: intent-mix PSI {intent_psi:.3f} > {PSI_ALERT}")
    if not np.isnan(ks_p) and ks_p < KS_PVALUE_ALERT and ks_stat > 0.1:
        alerts.append(f"ALERT: KS statistic {ks_stat:.3f} (p={ks_p:.2g})")
    if low_conf > LOW_CONF_SHARE_ALERT:
        alerts.append(f"ALERT: {low_conf:.1%} of traffic below the routing threshold")
    if refusals > REFUSAL_SHARE_ALERT:
        alerts.append(f"ALERT: {refusals:.1%} of answers were refusals - check retrieval and the KB")
    if len(served_versions) > 1:
        alerts.append(f"INFO: more than one model version served in this window: {served_versions}")

    return {
        "n_current": len(rows),
        "n_reference": reference["n"],
        "confidence_psi": round(conf_psi, 4),
        "intent_mix_psi": round(intent_psi, 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_pvalue": round(ks_p, 6),
        "low_confidence_share": round(low_conf, 4),
        "reference_low_confidence_share": reference["low_confidence_share"],
        "refusal_share": round(refusals, 4),
        "p95_latency_ms": int(np.percentile([r["latency_ms"] for r in rows], 95)),
        "escalation_share": round(sum(r["needs_escalation"] for r in rows) / len(rows), 4),
        "top_intents": dict(intents.most_common(5)),
        "new_intents_vs_reference": sorted(set(intent_share) - set(reference["intent_share"])),
        "model_versions_served": served_versions,
        "alerts": alerts,
        "interpretation": (
            "Drift in P(X) alone does not mean the model degraded. Confirm with the "
            "feedback join (predictions.request_id -> feedback.request_id) before retraining."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-reference", action="store_true")
    ap.add_argument("--db", default=str(REPO_ROOT / "feedback.db"))
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--fail-on-alert", action="store_true", help="non-zero exit for a cron job")
    args = ap.parse_args()

    if args.build_reference:
        build_reference()
        return 0
    if not REFERENCE.exists():
        print("no reference window; run --build-reference first", file=sys.stderr)
        return 1
    if not Path(args.db).exists():
        print(f"no prediction log at {args.db}", file=sys.stderr)
        return 1

    rows = load_current(args.db, args.limit)
    if not rows:
        print("no predictions logged yet", file=sys.stderr)
        return 1
    report = analyse(json.loads(REFERENCE.read_text()), rows)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "drift_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    hard = [a for a in report["alerts"] if a.startswith("ALERT")]
    return 1 if (hard and args.fail_on_alert) else 0


if __name__ == "__main__":
    sys.exit(main())

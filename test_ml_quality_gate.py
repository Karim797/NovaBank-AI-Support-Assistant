"""Deployment gates.

These are the tests that must fail a release. They read the evaluation reports
written by the training pipeline, so CI cannot deploy a model that was never
evaluated: a missing report is a failure, not a skip-and-ship.
"""

import pytest

from training.eval_retrieval import MIN_RECALL_AT_4, MIN_REFUSAL_RATE
from training.evaluate import MIN_MACRO_F1

pytestmark = pytest.mark.quality_gate


def test_intent_macro_f1_meets_minimum(eval_report):
    assert eval_report["test_macro_f1"] >= MIN_MACRO_F1, (
        f"macro F1 {eval_report['test_macro_f1']} below gate {MIN_MACRO_F1}"
    )


def test_headline_metric_is_not_carried_by_duplicated_rows(eval_report):
    """If de-duplicating the test set moves macro F1 materially, the headline
    number is partly memorisation and must not be reported on its own."""
    delta = eval_report["test_macro_f1"] - eval_report["test_macro_f1_dedup"]
    assert delta < 0.01, f"macro F1 drops {delta:.4f} once train/test duplicates are removed"


def test_confident_traffic_is_more_accurate_than_the_rest(eval_report):
    """The whole routing gate rests on this. If confidence does not separate
    correct from incorrect predictions, the threshold is decoration."""
    assert eval_report["accuracy_on_confident"] > eval_report["test_accuracy"]
    assert eval_report["accuracy_on_confident"] > eval_report["accuracy_on_low_confidence"]


def test_coverage_is_high_enough_to_be_useful(eval_report):
    assert eval_report["coverage_at_threshold"] >= 0.85


def test_retrieval_recall_meets_minimum(retrieval_report):
    assert retrieval_report["k"]["recall@4"] >= MIN_RECALL_AT_4


def test_out_of_scope_questions_are_refused(retrieval_report):
    assert retrieval_report["refusal_rate_out_of_scope"] >= MIN_REFUSAL_RATE


def test_topic_filtering_does_not_hurt_recall(retrieval_report):
    """The architecture claims the intent classifier improves retrieval. If a
    change ever makes the filter harmful, this fails and the filter should go."""
    filtered = retrieval_report["k"].get("recall@4_topic_filtered")
    if filtered is None:
        pytest.skip("filtered recall not measured")
    assert filtered >= retrieval_report["k"]["recall@4"]

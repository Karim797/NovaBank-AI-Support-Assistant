"""Deployment gates.

These are the tests that must fail a release. They read the evaluation reports
written by the training pipeline, so CI cannot deploy a model that was never
evaluated: a missing report is a failure, not a skip-and-ship.
"""

import pytest

from training.eval_retrieval import (
    MAX_TOPIC_PRIOR_RECALL_DROP,
    MIN_PRODUCTION_RECALL_AT_4,
    MIN_REFUSAL_RATE,
)
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


def test_bootstrap_interval_contains_headline_f1(eval_report):
    low, high = eval_report["test_macro_f1_bootstrap_95ci"]
    assert low <= eval_report["test_macro_f1"] <= high


def test_calibration_is_measured(eval_report):
    assert 0.0 <= eval_report["ece_10_bins"] <= 1.0
    assert len(eval_report["calibration_bins"]) == 10


def test_confident_traffic_is_more_accurate_than_the_rest(eval_report):
    """The whole routing gate rests on this. If confidence does not separate
    correct from incorrect predictions, the threshold is decoration."""
    assert eval_report["accuracy_on_confident"] > eval_report["test_accuracy"]
    assert eval_report["accuracy_on_confident"] > eval_report["accuracy_on_low_confidence"]


def test_coverage_is_high_enough_to_be_useful(eval_report):
    assert eval_report["coverage_at_threshold"] >= 0.85


def test_production_retrieval_recall_meets_minimum(retrieval_report):
    assert retrieval_report["production_recall@4"] >= MIN_PRODUCTION_RECALL_AT_4


def test_out_of_scope_questions_are_refused(retrieval_report):
    assert retrieval_report["refusal_rate_out_of_scope"] >= MIN_REFUSAL_RATE


def test_topic_prior_does_not_materially_hurt_ceiling_recall(retrieval_report):
    """A soft prior may reorder near-ties, but unlike the old hard filter it
    must not destroy recall when the classifier is confidently wrong."""
    unfiltered = retrieval_report["k"]["recall@4"]
    prior = retrieval_report["k"]["recall@4_topic_prior"]
    assert unfiltered - prior <= MAX_TOPIC_PRIOR_RECALL_DROP


def test_harder_retrieval_set_is_actually_present(retrieval_report):
    assert retrieval_report["n_in_scope"] >= 50
    assert retrieval_report["n_paraphrase"] >= 20
    assert retrieval_report["n_out_of_scope"] >= 20

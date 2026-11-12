"""Metric math checked against hand-computed values."""

import pytest

from src.benchmarker import hit_at_k, recall_at_k, reciprocal_rank_at_k

RANKED = ["R1", "R2", "R3", "R4", "R5", "R6"]


# Test 5 --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "relevant,k,hit,recall,rr",
    [
        ({"R1"}, 5, 1.0, 1.0, 1.0),  # first place
        ({"R3"}, 5, 1.0, 1.0, 1 / 3),  # third place
        ({"R2", "R4"}, 5, 1.0, 1.0, 0.5),  # first relevant counts for MRR
        ({"R6"}, 5, 0.0, 0.0, 0.0),  # outside top-K: no credit at all
        ({"R5", "X"}, 5, 1.0, 0.5, 0.2),  # one of two relevant found
        ({"X"}, 5, 0.0, 0.0, 0.0),  # missed query -> 0.0, no division by zero
        (set(), 5, 0.0, 0.0, 0.0),  # no labels -> 0.0
    ],
)
def test_metric_math(relevant, k, hit, recall, rr):
    assert hit_at_k(RANKED, relevant, k) == hit
    assert recall_at_k(RANKED, relevant, k) == pytest.approx(recall)
    assert reciprocal_rank_at_k(RANKED, relevant, k) == pytest.approx(rr)


def test_duplicates_and_ties_do_not_inflate_scores():
    ranked = ["R1", "R1", "R1", "R2"]  # duplicate ids (e.g. tied scores returned twice)
    assert recall_at_k(ranked, {"R1"}, 3) == 1.0  # never above 1
    assert reciprocal_rank_at_k(ranked, {"R2"}, 2) == 0.5  # R2 is the 2nd distinct result


def test_mean_reciprocal_rank_over_queries():
    runs = [(["a", "b"], {"a"}), (["a", "b"], {"b"}), (["a", "b"], {"z"})]
    mrr = sum(reciprocal_rank_at_k(r, rel, 2) for r, rel in runs) / len(runs)
    assert mrr == pytest.approx((1 + 0.5 + 0) / 3)

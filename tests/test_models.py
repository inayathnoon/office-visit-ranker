"""Ranking metrics, calibration, and the claims the README makes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from visit_ranker.models import calibration, evaluate


def test_ndcg_at_1_equals_hit_at_1(scored):
    """With one relevant item per group they are the same number, and the
    README says so rather than quoting both as independent evidence."""
    per_trip = evaluate.trip_level(scored)
    assert per_trip["ndcg_at_1"].mean() == pytest.approx(per_trip["hit_at_1"].mean())


def test_metrics_are_ordered_as_definitions_require(scored):
    per_trip = evaluate.trip_level(scored)
    assert per_trip["hit_at_1"].mean() <= per_trip["recall_at_2"].mean()
    assert per_trip["recall_at_2"].mean() <= per_trip["recall_at_3"].mean()
    assert per_trip["ndcg_at_1"].mean() <= per_trip["ndcg_at_3"].mean()


def test_a_perfect_ranking_scores_one():
    frame = pd.DataFrame(
        {
            "trip_id": ["T1"] * 3 + ["T2"] * 3,
            "office_id": ["A", "B", "C"] * 2,
            "label": [1, 0, 0, 1, 0, 0],
            "score": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
        }
    )
    per_trip = evaluate.trip_level(frame)
    assert per_trip["ndcg_at_1"].mean() == 1.0
    assert per_trip["reciprocal_rank"].mean() == 1.0


def test_a_reversed_ranking_scores_badly():
    frame = pd.DataFrame(
        {
            "trip_id": ["T1"] * 3,
            "office_id": ["A", "B", "C"],
            "label": [1, 0, 0],
            "score": [1.0, 2.0, 3.0],
        }
    )
    per_trip = evaluate.trip_level(frame)
    assert per_trip["ndcg_at_1"].iloc[0] == 0.0
    assert per_trip["reciprocal_rank"].iloc[0] == pytest.approx(1 / 3)


def test_ties_are_broken_deterministically():
    """A baseline that scores everything the same must not be flattered by the
    order the frame happened to arrive in."""
    frame = pd.DataFrame(
        {
            "trip_id": ["T1"] * 3,
            "office_id": ["C", "B", "A"],
            "label": [0, 0, 1],
            "score": [1.0, 1.0, 1.0],
        }
    )
    first = evaluate.trip_level(frame)["rank"].iloc[0]
    second = evaluate.trip_level(frame.iloc[::-1])["rank"].iloc[0]
    assert first == second


def test_calibrated_probabilities_are_monotone_in_the_margin(scored):
    top = calibration.top_one_frame(scored)
    calibrator = calibration.fit_calibrator(top.assign(correct=top["correct"]))
    margins = np.linspace(top["margin"].min(), top["margin"].max(), 25)
    probabilities = calibrator.probability(margins)
    assert np.all(np.diff(probabilities) >= -1e-9)


def test_abstention_never_reduces_accuracy_with_fallback(scored):
    """Sending the top two when unsure cannot do worse than sending one."""
    top = calibration.top_one_frame(scored)
    calibrator = calibration.fit_calibrator(top)
    top["probability"] = calibrator.probability(top["margin"].to_numpy())
    top = calibration.add_top2_flag(top, scored)
    curve = calibration.abstention_curve(top)
    no_abstention = float(curve.iloc[0]["accuracy_covered"])
    assert curve["accuracy_with_fallback"].max() >= no_abstention - 1e-9


def test_coverage_falls_as_the_threshold_rises(scored):
    top = calibration.top_one_frame(scored)
    calibrator = calibration.fit_calibrator(top)
    top["probability"] = calibrator.probability(top["margin"].to_numpy())
    top = calibration.add_top2_flag(top, scored)
    curve = calibration.abstention_curve(top)
    assert curve["coverage"].is_monotonic_decreasing

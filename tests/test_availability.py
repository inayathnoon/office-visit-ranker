"""The availability contract: the invariant, and the routing around it."""

from __future__ import annotations

import pytest

from visit_ranker.models import availability, ranker
from visit_ranker.models.availability import AvailabilityMask, mask_from_key


def test_a_mask_lists_only_the_features_it_has():
    mask = AvailabilityMask(frozenset({"team"}))
    columns = mask.features()
    assert "team_visit_share" in columns
    assert "habit_last_visit" not in columns
    assert "leader_is_top1" not in columns
    # Office and trip context are never missing.
    assert "office_log_headcount" in columns
    assert "trip_length_days" in columns


def test_the_empty_mask_still_has_something_to_rank_on():
    columns = AvailabilityMask(frozenset()).features()
    assert len(columns) > 5
    # Nothing from a behavioural family. prior_visit_count is deliberately
    # present: a count of zero is a fact, unlike a share over an empty
    # history, which is undefined.
    behavioural = ("habit", "emp_visit", "leader", "team_", "l3_", "dept_recent")
    assert all(not c.startswith(behavioural) for c in columns)
    assert "prior_visit_count" in columns


def test_fallback_walks_down_and_terminates():
    mask = AvailabilityMask(frozenset({"personal", "leader", "team", "department"}))
    seen = []
    current = mask
    while current is not None:
        seen.append(current.key)
        current = current.drop_one()
    assert seen[-1] == "none"
    # Leader is given up first: it is the least informative of the four.
    assert "leader" not in seen[1]


def test_resolution_prefers_an_exact_model_then_falls_back():
    trained = {"none", "department+team"}
    exact, steps = availability.resolve_model(mask_from_key("department+team"), trained)
    assert exact.key == "department+team" and steps == 0

    fallen, steps = availability.resolve_model(mask_from_key("department+leader+team"), trained)
    assert fallen.key == "department+team" and steps == 1


def test_a_model_never_reads_a_feature_its_row_lacks(candidates):
    """The invariant. If this breaks, the whole contract is decoration."""
    train = candidates[candidates["split"] == "train"]
    trained = set(availability.training_masks(train))
    for key in candidates["mask_key"].unique():
        mask = mask_from_key(str(key))
        resolved, _ = availability.resolve_model(mask, trained)
        allowed = set(mask.features())
        illegal = [c for c in resolved.features() if c not in allowed]
        assert not illegal, f"mask {key} routed to {resolved.key}, which reads {illegal}"


def test_scoring_raises_rather_than_guessing_on_a_contract_breach(candidates):
    """Scoring enforces the contract at the moment of use, not just at setup."""
    train = candidates[candidates["split"] == "train"]
    validation = candidates[candidates["split"] == "validation"]
    family = ranker.train_family(train, validation)

    # Deliberately mislabel rich rows as having nothing, so the router picks a
    # model that reads features the (claimed) mask does not have.
    broken = candidates[candidates["split"] == "test"].copy()
    broken["mask_key"] = "none"
    family.models["none"] = family.models[max(family.models, key=lambda k: len(family.features[k]))]
    family.features["none"] = family.features[
        max(family.features, key=lambda k: len(family.features[k]))
    ]
    with pytest.raises(RuntimeError, match="would read"):
        family.score(broken)


def test_every_trip_is_scored_by_some_model(scored):
    assert scored["scored_by"].ne("").all()
    assert scored["score"].notna().all()

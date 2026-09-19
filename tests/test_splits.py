"""The time split. Asserted, because a random one would flatter every metric."""

from __future__ import annotations

import pandas as pd

from visit_ranker.models.tuning import assert_no_overlap, split_boundaries


def test_splits_do_not_overlap_in_time(candidates):
    result = assert_no_overlap(candidates)
    assert result["passed"], result["problems"]


def test_every_split_is_present_and_non_empty(candidates):
    boundaries = split_boundaries(candidates).set_index("split")
    for split in ("train", "validation", "test"):
        assert split in boundaries.index
        assert boundaries.loc[split, "trips"] > 0


def test_train_is_the_largest_split(candidates):
    counts = candidates.drop_duplicates("trip_id")["split"].value_counts()
    assert counts["train"] > counts["validation"]
    assert counts["train"] > counts["test"]


def test_a_trip_belongs_to_exactly_one_split(candidates):
    per_trip = candidates.groupby("trip_id")["split"].nunique()
    assert int(per_trip.max()) == 1


def test_split_boundary_matches_the_config(candidates, cfg):
    train_last = (
        pd.to_datetime(candidates[candidates["split"] == "train"]["arrival_date"]).max().date()
    )
    validation_first = (
        pd.to_datetime(candidates[candidates["split"] == "validation"]["arrival_date"]).min().date()
    )
    assert train_last < cfg.train_end <= validation_first

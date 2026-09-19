"""Leakage checks, one per behavioural feature family.

Each feature is re-derived by brute force for a sample of trips - walking the
raw trip table and using only rows strictly before the trip - and compared
against what the builder produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from visit_ranker.config import RAW_DIR


@pytest.fixture(scope="module")
def raw_trips():
    path = RAW_DIR / "trips" / "part-000.parquet"
    if not path.exists():
        pytest.skip("no generated data")
    trips = pd.read_parquet(path)
    trips["arrival_date"] = pd.to_datetime(trips["arrival_date"])
    return trips.sort_values(["arrival_date", "trip_id"])


def _prior_visits(raw_trips, emp_id, city, arrival_date):
    """Every visit by this employee to this city, strictly before the trip."""
    return raw_trips[
        (raw_trips["emp_id"] == emp_id)
        & (raw_trips["destination_city"] == city)
        & (raw_trips["arrival_date"] < arrival_date)
    ]


def test_habit_matches_a_brute_force_as_of_computation(candidates, raw_trips):
    sample = candidates.drop_duplicates("trip_id").sample(40, random_state=0)
    for trip in sample.itertuples(index=False):
        prior = _prior_visits(raw_trips, trip.emp_id, trip.destination_city, trip.arrival_date)
        rows = candidates[candidates["trip_id"] == trip.trip_id]
        if prior.empty:
            assert rows["habit_last_visit"].sum() == 0
            continue
        expected = prior.iloc[-1]["visited_office_id"]
        flagged = rows[rows["habit_last_visit"] > 0]["office_id"].tolist()
        assert flagged == [expected]


def test_personal_visit_share_matches_prior_visits_only(candidates, raw_trips):
    sample = candidates.drop_duplicates("trip_id").sample(30, random_state=1)
    for trip in sample.itertuples(index=False):
        prior = _prior_visits(raw_trips, trip.emp_id, trip.destination_city, trip.arrival_date)
        rows = candidates[candidates["trip_id"] == trip.trip_id].set_index("office_id")
        for office, share in rows["emp_visit_share"].items():
            expected = (prior["visited_office_id"] == office).mean() if len(prior) else 0.0
            assert share == pytest.approx(expected, abs=1e-9)


def test_prior_visit_count_excludes_the_trip_itself(candidates, raw_trips):
    sample = candidates.drop_duplicates("trip_id").sample(30, random_state=2)
    for trip in sample.itertuples(index=False):
        prior = _prior_visits(raw_trips, trip.emp_id, trip.destination_city, trip.arrival_date)
        assert trip.prior_visit_count == float(len(prior))


def test_first_visit_flag_agrees_with_the_history(candidates, raw_trips):
    sample = candidates.drop_duplicates("trip_id").sample(40, random_state=3)
    for trip in sample.itertuples(index=False):
        prior = _prior_visits(raw_trips, trip.emp_id, trip.destination_city, trip.arrival_date)
        assert bool(trip.is_first_ever_visit_to_city) == prior.empty


def test_no_feature_perfectly_predicts_the_label(candidates):
    """A feature that separates the label exactly is a leak, not a signal."""
    from visit_ranker.features.build import FEATURE_COLUMNS

    for column in FEATURE_COLUMNS:
        values = candidates[column]
        if values.nunique() < 2:
            continue
        correlation = abs(float(np.corrcoef(values, candidates["label"])[0, 1]))
        assert correlation < 0.95, f"{column} correlates {correlation:.3f} with the label"


def test_label_is_exactly_one_office_per_trip(candidates):
    per_trip = candidates.groupby("trip_id")["label"].sum()
    assert set(per_trip.unique()) == {1}

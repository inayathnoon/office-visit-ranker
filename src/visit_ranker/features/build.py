"""Candidate features, every one computed as-of the trip.

The unit of work is a **candidate row**: one row per (trip, office-in-the-
destination-city), with a label of 1 for the office actually visited. A trip is
a ranking group, which is what makes this a ranking problem rather than a
classification one - the model's job is to order a handful of offices, not to
answer yes or no about each independently.

THE RULE THAT MATTERS
Every behavioural feature is built from taps strictly BEFORE the trip's arrival
date. That is enforced structurally: the history is walked in date order and a
trip only ever sees the state accumulated before it. A grouped aggregate over
the whole panel would be far faster and would silently include the trip's own
visit in its own features, which produces a model that is perfect in validation
and useless in production.

``tests/test_leakage.py`` re-derives each feature independently and checks it
against a brute-force as-of computation on a sample of trips.

AVAILABILITY
Some features cannot exist for some trips. A first-time visitor to a city has
no habit and no personal history there; a contractor with no manager in the HR
extract has no leader gravity. Those are recorded as an availability mask
rather than imputed, because imputing a zero tells the model "the manager uses
none of these offices", which is a different and false statement from "we do
not know where the manager sits". See models/availability.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import RAW_DIR, Config, load_config

# Half-life in days for the recency-decayed variants. Roughly a quarter: a
# visit from last month should count for more than one from last year, and the
# decay is what lets the model track a department that has relocated.
RECENCY_HALF_LIFE_DAYS = 90.0

# How many top offices to encode for each "top-N" feature family.
TOP_N = 2

FEATURE_COLUMNS = [
    # 1. habit
    "habit_last_visit",
    "habit_last_visit_decayed",
    # 2. the employee's own history in this city
    "emp_visit_share",
    "emp_visit_share_decayed",
    "emp_is_top1",
    "emp_is_top2",
    # 3. leader gravity
    "leader_visit_share",
    "leader_is_top1",
    # 4. immediate team
    "team_visit_share",
    "team_visit_share_decayed",
    "team_is_top1",
    # 5. L3 department
    "l3_visit_share",
    "l3_visit_share_decayed",
    "l3_is_top1",
    # 6. recent departmental travel
    "dept_recent_visit_share",
    # 8. office context
    "office_log_headcount",
    "office_distance_km",
    "office_is_largest_in_city",
    "office_dept_mix_similarity",
    "office_has_dedicated_space",
    "office_dedicated_desks",
    # 9. trip context
    "trip_length_days",
    "arrival_weekday",
    "days_since_previous_visit",
    "is_first_ever_visit_to_city",
    "candidates_in_city",
    # A COUNT of prior visits, which is always available: zero is a fact, not
    # a missing value. The distinction from the personal-history family is the
    # whole basis of the availability contract - "they have never been here"
    # is information, while "the share of their previous visits that went to
    # this office" is undefined when there are no previous visits. The first
    # belongs in trip context; the second must be masked out.
    "prior_visit_count",
]

# Which features belong to which availability family. A feature whose family
# is unavailable is not scored, not imputed.
FEATURE_FAMILIES = {
    "habit": ["habit_last_visit", "habit_last_visit_decayed"],
    "personal_history": [
        "emp_visit_share",
        "emp_visit_share_decayed",
        "emp_is_top1",
        "emp_is_top2",
    ],
    "leader": ["leader_visit_share", "leader_is_top1"],
    "team": ["team_visit_share", "team_visit_share_decayed", "team_is_top1"],
    "department": [
        "l3_visit_share",
        "l3_visit_share_decayed",
        "l3_is_top1",
        "dept_recent_visit_share",
    ],
    "office_context": [
        "office_log_headcount",
        "office_distance_km",
        "office_is_largest_in_city",
        "office_dept_mix_similarity",
        "office_has_dedicated_space",
        "office_dedicated_desks",
    ],
    "trip_context": [
        "trip_length_days",
        "arrival_weekday",
        "days_since_previous_visit",
        "is_first_ever_visit_to_city",
        "candidates_in_city",
        "prior_visit_count",
    ],
}

# Families that are never missing. Office and trip context are knowable before
# anyone has ever travelled anywhere.
ALWAYS_AVAILABLE = ("office_context", "trip_context")


@dataclass
class VisitHistory:
    """Visits accumulated so far, keyed for constant-time lookup.

    Deliberately a mutable object walked forward in date order rather than a
    set of pandas aggregations. It is slower per trip and it makes the as-of
    guarantee structural: the object simply does not contain the future.
    """

    by_emp_city: dict[tuple[str, str], list[tuple[str, pd.Timestamp]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    by_team_city: dict[tuple[str, str], list[tuple[str, pd.Timestamp]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    by_l3_city: dict[tuple[str, str], list[tuple[str, pd.Timestamp]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    recent_by_l3_city: dict[tuple[str, str], list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def record(self, trip, employee: pd.Series) -> None:
        key_emp = (trip.emp_id, trip.destination_city)
        entry = (trip.visited_office_id, trip.arrival_date)
        self.by_emp_city[key_emp].append(entry)
        self.by_team_city[(employee["dept_l4"], trip.destination_city)].append(entry)
        self.by_l3_city[(employee["dept_l3"], trip.destination_city)].append(entry)

        recent = self.recent_by_l3_city[(employee["dept_l3"], trip.destination_city)]
        recent.append(trip.visited_office_id)
        # Only the last 30 departmental visits count, so a department that has
        # moved is reflected within a month rather than being outvoted by two
        # years of history at the old building.
        if len(recent) > 30:
            recent.pop(0)


def _share(history: list[tuple[str, pd.Timestamp]], office: str) -> float:
    if not history:
        return 0.0
    return sum(1 for o, _ in history if o == office) / len(history)


def _decayed_share(
    history: list[tuple[str, pd.Timestamp]], office: str, as_of: pd.Timestamp
) -> float:
    if not history:
        return 0.0
    weights = np.array(
        [0.5 ** (max((as_of - when).days, 0) / RECENCY_HALF_LIFE_DAYS) for _, when in history]
    )
    total = weights.sum()
    if total <= 0:
        return 0.0
    matching = np.array([1.0 if o == office else 0.0 for o, _ in history])
    return float((weights * matching).sum() / total)


def _top_offices(history: list[tuple[str, pd.Timestamp]], n: int) -> list[str]:
    if not history:
        return []
    counts: dict[str, int] = defaultdict(int)
    for office, _ in history:
        counts[office] += 1
    return [o for o, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def load_raw(cfg: Config | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    tables = {}
    for name in ("offices", "employees", "space_allocation", "trips"):
        tables[name] = pd.read_parquet(RAW_DIR / name / "part-000.parquet")
    tables["trips"]["arrival_date"] = pd.to_datetime(tables["trips"]["arrival_date"])
    return tables


def build_candidates(cfg: Config | None = None) -> pd.DataFrame:
    """One row per (trip, candidate office), with features and a label."""
    cfg = cfg or load_config()
    tables = load_raw(cfg)
    offices = tables["offices"]
    employees = tables["employees"].set_index("emp_id")
    trips = tables["trips"].sort_values(["arrival_date", "trip_id"]).reset_index(drop=True)

    offices_by_city = {city: group for city, group in offices.groupby("city")}
    dedicated = {
        (row.office_id, row.dept_l1): row.desks
        for row in tables["space_allocation"].itertuples(index=False)
    }
    largest_in_city = (
        offices.sort_values("headcount", ascending=False)
        .drop_duplicates("city")
        .set_index("city")["office_id"]
        .to_dict()
    )

    history = VisitHistory()
    last_visit_date: dict[tuple[str, str], pd.Timestamp] = {}
    rows: list[dict] = []

    for trip in trips.itertuples(index=False):
        employee = employees.loc[trip.emp_id]
        city = trip.destination_city
        candidates = offices_by_city[city]

        emp_history = history.by_emp_city.get((trip.emp_id, city), [])
        team_history = history.by_team_city.get((employee["dept_l4"], city), [])
        l3_history = history.by_l3_city.get((employee["dept_l3"], city), [])
        recent_dept = history.recent_by_l3_city.get((employee["dept_l3"], city), [])

        manager = employee["manager_id"]
        has_leader = manager is not None and not (isinstance(manager, float) and np.isnan(manager))
        leader_history: list[tuple[str, pd.Timestamp]] = []
        leader_home: str | None = None
        if has_leader:
            leader_history = history.by_emp_city.get((str(manager), city), [])
            manager_row = employees.loc[str(manager)]
            if manager_row["base_city"] == city:
                leader_home = manager_row["home_office_id"]

        # Availability is decided here, once, from the state as of this trip.
        available = {
            "habit": bool(emp_history),
            "personal_history": bool(emp_history),
            "leader": bool(has_leader and (leader_history or leader_home)),
            "team": bool(team_history),
            "department": bool(l3_history),
            "office_context": True,
            "trip_context": True,
        }

        emp_top = _top_offices(emp_history, TOP_N)
        team_top = _top_offices(team_history, 1)
        l3_top = _top_offices(l3_history, 1)
        leader_top = _top_offices(leader_history, 1)
        if leader_home is not None and not leader_top:
            leader_top = [leader_home]

        habit_office = emp_history[-1][0] if emp_history else None
        previous_date = last_visit_date.get((trip.emp_id, city))
        days_since = (trip.arrival_date - previous_date).days if previous_date is not None else -1

        for office in candidates.itertuples(index=False):
            key = (office.office_id, employee["dept_l1"])
            rows.append(
                {
                    "trip_id": trip.trip_id,
                    "emp_id": trip.emp_id,
                    "office_id": office.office_id,
                    "destination_city": city,
                    "arrival_date": trip.arrival_date,
                    "label": int(office.office_id == trip.visited_office_id),
                    # 1. habit
                    "habit_last_visit": float(office.office_id == habit_office),
                    "habit_last_visit_decayed": (
                        _decayed_share(emp_history[-1:], office.office_id, trip.arrival_date)
                        if emp_history
                        else 0.0
                    ),
                    # 2. personal history
                    "emp_visit_share": _share(emp_history, office.office_id),
                    "emp_visit_share_decayed": _decayed_share(
                        emp_history, office.office_id, trip.arrival_date
                    ),
                    "emp_is_top1": float(bool(emp_top) and office.office_id == emp_top[0]),
                    "emp_is_top2": float(office.office_id in emp_top[:2]),
                    # 3. leader
                    "leader_visit_share": _share(leader_history, office.office_id),
                    "leader_is_top1": float(bool(leader_top) and office.office_id == leader_top[0]),
                    # 4. team
                    "team_visit_share": _share(team_history, office.office_id),
                    "team_visit_share_decayed": _decayed_share(
                        team_history, office.office_id, trip.arrival_date
                    ),
                    "team_is_top1": float(bool(team_top) and office.office_id == team_top[0]),
                    # 5. department
                    "l3_visit_share": _share(l3_history, office.office_id),
                    "l3_visit_share_decayed": _decayed_share(
                        l3_history, office.office_id, trip.arrival_date
                    ),
                    "l3_is_top1": float(bool(l3_top) and office.office_id == l3_top[0]),
                    # 6. recent departmental travel
                    "dept_recent_visit_share": (
                        recent_dept.count(office.office_id) / len(recent_dept)
                        if recent_dept
                        else 0.0
                    ),
                    # 8. office context
                    "office_log_headcount": float(np.log(office.headcount)),
                    "office_distance_km": float(office.distance_km),
                    "office_is_largest_in_city": float(
                        largest_in_city.get(city) == office.office_id
                    ),
                    "office_dept_mix_similarity": float(
                        getattr(office, f"mix_{employee['dept_l1']}")
                    ),
                    "office_has_dedicated_space": float(key in dedicated),
                    "office_dedicated_desks": float(dedicated.get(key, 0)),
                    # 9. trip context
                    "trip_length_days": float(trip.trip_length_days),
                    "arrival_weekday": float(trip.arrival_date.weekday()),
                    "days_since_previous_visit": float(days_since),
                    "is_first_ever_visit_to_city": float(not emp_history),
                    "candidates_in_city": float(len(candidates)),
                    "prior_visit_count": float(len(emp_history)),
                    # availability, carried per row
                    **{f"available_{family}": value for family, value in available.items()},
                }
            )

        history.record(trip, employee)
        last_visit_date[(trip.emp_id, city)] = trip.arrival_date

    frame = pd.DataFrame(rows)
    frame["split"] = _assign_split(frame["arrival_date"], cfg)
    return frame


def _assign_split(dates: pd.Series, cfg: Config) -> pd.Series:
    """Train on months 1-18, validate on 19-21, test on 22-24.

    By time, never at random. Two trips by the same employee to the same city
    weeks apart share a habit that is the strongest feature in the model; a
    random split puts one on each side of the boundary and the score comes back
    flattering. The boundaries are asserted in tests/test_splits.py.
    """
    as_date = pd.to_datetime(dates).dt.date
    return pd.Series(
        np.where(
            as_date < cfg.train_end,
            "train",
            np.where(as_date < cfg.validation_end, "validation", "test"),
        ),
        index=dates.index,
    )

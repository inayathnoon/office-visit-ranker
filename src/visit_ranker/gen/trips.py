"""Trips, and the choice of which office to walk into.

This is the file that decides whether the repository is worth anything. If the
choice is random there is nothing to learn; if it is deterministic the model
hits 100% and the evaluation is meaningless. It has to be *learnable but not
fully learnable*, and the ceiling has to be knowable.

The choice is a multinomial logit over the offices in the destination city:

    utility_j = w_habit    * visited_this_office_on_the_last_visit
              + w_leader   * the employee's manager usually uses this office
              + w_team     * the employee's L4 team usually uses this office
              + w_l3       * the employee's L3 department usually uses it
              + w_size     * standardised log headcount
              + w_distance * standardised negative distance from the centre
              + w_dept_mix * similarity of the office's mix to the traveller's
              + w_dedicated* the traveller's department holds desks here
              + Gumbel(0, scale)

with an exploration share that ignores utility completely. That share is the
IRREDUCIBLE ERROR. Nothing in the feature set predicts it, so a model reporting
accuracy above the implied ceiling has found a leak, not a signal - which is
the single most useful thing this simulator provides.

Trips are generated in time order because the habit term depends on the
previous visit. That ordering is also why every feature must be computed
as-of the trip rather than over the whole window.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ..config import Config
from .rng import gumbel, lognormal_from_mean, substream


def _standardise(values: np.ndarray) -> np.ndarray:
    spread = values.std()
    return (values - values.mean()) / spread if spread > 1e-9 else np.zeros_like(values)


def _department_mix_similarity(office_row: pd.Series, department: str) -> float:
    return float(office_row[f"mix_{department}"])


class TripSimulator:
    """Holds the lookups the trip loop needs, and the running history."""

    def __init__(
        self,
        cfg: Config,
        employees: pd.DataFrame,
        offices: pd.DataFrame,
        allocation: pd.DataFrame,
    ) -> None:
        self.cfg = cfg
        self.rng = substream(cfg.seed, "trips")
        self.employees = employees.set_index("emp_id")
        self.offices = offices

        self.offices_by_city = {
            city: group.reset_index(drop=True) for city, group in offices.groupby("city")
        }
        # Pre-standardise the office attributes within each city. The choice is
        # between offices in one city, so standardising across the whole estate
        # would make every office in a small city look identical.
        self.city_features: dict[str, dict[str, np.ndarray]] = {}
        for city, group in self.offices_by_city.items():
            self.city_features[city] = {
                "size": _standardise(np.log(group["headcount"].to_numpy(dtype=float))),
                "distance": _standardise(-group["distance_km"].to_numpy(dtype=float)),
            }

        self.dedicated = {
            (row.office_id, row.dept_l1) for row in allocation.itertuples(index=False)
        }

        # Where each person sits, and which office each team and department
        # occupies most in each city. Gravity comes from these as well as from
        # travel history: colleagues who are BASED somewhere are the reason a
        # visitor goes to that building.
        self.home_office = employees.set_index("emp_id")["home_office_id"].to_dict()
        self.team_home = self._modal_home(employees, "dept_l4")
        self.l3_home = self._modal_home(employees, "dept_l3")

        # Running history, updated as trips are generated. Keyed so the habit
        # and gravity terms can be looked up in constant time.
        self.last_visit: dict[tuple[str, str], str] = {}
        self.visits_by_emp_city: dict[tuple[str, str], list[str]] = {}
        self.visits_by_team_city: dict[tuple[str, str], list[str]] = {}
        self.visits_by_l3_city: dict[tuple[str, str], list[str]] = {}
        self.relocated: dict[str, dict[str, str]] = {}

    @staticmethod
    def _modal_home(employees: pd.DataFrame, level: str) -> dict[tuple[str, str], str]:
        """The office each group occupies most, per city."""
        counts = (
            employees.dropna(subset=["home_office_id"])
            .groupby([level, "base_city", "home_office_id"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
            .drop_duplicates([level, "base_city"])
        )
        return {
            (row[level], row["base_city"]): row["home_office_id"] for _, row in counts.iterrows()
        }

    # --- gravity terms ------------------------------------------------------

    def _modal_office(self, history: list[str]) -> str | None:
        if not history:
            return None
        values, counts = np.unique(np.array(history), return_counts=True)
        return str(values[int(counts.argmax())])

    def _leader_office(self, manager_id: str | None, city: str) -> str | None:
        """Where the manager works, in that city.

        Their desk if they are based there, otherwise the office they use most
        when they visit. Checking the desk first is what makes the feature
        dense enough to matter.
        """
        if manager_id is None or (isinstance(manager_id, float) and np.isnan(manager_id)):
            return None
        manager = str(manager_id)
        if self.employees.loc[manager, "base_city"] == city:
            home = self.home_office.get(manager)
            if home is not None:
                return str(home)
        return self._modal_office(self.visits_by_emp_city.get((manager, city), []))

    # --- the choice ---------------------------------------------------------

    def choose_office(
        self, emp_id: str, employee: pd.Series, city: str, month_index: int
    ) -> tuple[str, dict]:
        candidates = self.offices_by_city[city]
        n = len(candidates)
        weights = self.cfg.choice.weights

        if n == 1:
            office = str(candidates["office_id"].iloc[0])
            return office, {"exploration": False, "n_candidates": 1}

        habit_office = self.last_visit.get((emp_id, city))
        leader_office = self._leader_office(employee["manager_id"], city)
        team_office = self._modal_office(
            self.visits_by_team_city.get((employee["dept_l4"], city), [])
        ) or self.team_home.get((employee["dept_l4"], city))
        l3_office = self._modal_office(
            self.visits_by_l3_city.get((employee["dept_l3"], city), [])
        ) or self.l3_home.get((employee["dept_l3"], city))

        # A relocated department's gravity moves to its new office from the
        # relocation month onward. A model that pools the whole history learns
        # the post-move office and looks clairvoyant on pre-move trips.
        relocation = self.relocated.get(employee["dept_l3"])
        if relocation and month_index >= self.cfg.travel.relocation["at_month"]:
            moved_to = relocation.get(city)
            if moved_to is not None:
                team_office = moved_to
                l3_office = moved_to

        office_ids = candidates["office_id"].to_numpy()
        features = self.city_features[city]
        department = employee["dept_l1"]

        utility = (
            weights["habit"] * (office_ids == habit_office).astype(float)
            + weights["leader"] * (office_ids == leader_office).astype(float)
            + weights["team"] * (office_ids == team_office).astype(float)
            + weights["l3_department"] * (office_ids == l3_office).astype(float)
            + weights["size"] * features["size"]
            + weights["distance"] * features["distance"]
            + weights["dept_mix"]
            * _standardise(candidates[f"mix_{department}"].to_numpy(dtype=float))
            + weights["dedicated_space"]
            * np.array([(o, department) in self.dedicated for o in office_ids], dtype=float)
        )
        utility = utility + gumbel(self.rng, self.cfg.choice.noise_scale, n)

        exploring = bool(self.rng.random() < self.cfg.choice.exploration_share)
        index = int(self.rng.integers(n)) if exploring else int(utility.argmax())

        return str(office_ids[index]), {
            "exploration": exploring,
            "n_candidates": n,
            "had_habit": habit_office is not None,
            "had_leader": leader_office is not None,
            "had_team": team_office is not None,
        }

    def record(self, emp_id: str, employee: pd.Series, city: str, office: str) -> None:
        self.last_visit[(emp_id, city)] = office
        self.visits_by_emp_city.setdefault((emp_id, city), []).append(office)
        self.visits_by_team_city.setdefault((employee["dept_l4"], city), []).append(office)
        self.visits_by_l3_city.setdefault((employee["dept_l3"], city), []).append(office)


def generate_trips(
    cfg: Config, employees: pd.DataFrame, offices: pd.DataFrame, allocation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (trips, trip_taps, realised truth)."""
    rng = substream(cfg.seed, "travel")
    simulator = TripSimulator(cfg, employees, offices, allocation)

    # Who travels.
    n = len(employees)
    travellers = employees[rng.random(n) < cfg.profile.traveller_share].copy()
    trips_each = np.maximum(
        1,
        np.round(
            lognormal_from_mean(
                rng,
                cfg.profile.trips_per_traveller["mean"],
                cfg.profile.trips_per_traveller["sigma"],
                len(travellers),
            )
        ).astype(int),
    )

    # Each traveller has a small set of cities they actually visit, and the set
    # is NOT drawn uniformly. People travel to where their organisation is:
    # their department's other sites, and their manager.
    #
    # Drawing destinations uniformly was the first version, and it made leader
    # gravity fire on 4% of trips - your manager is almost never in a randomly
    # chosen city. Weighting by organisational presence is both the realistic
    # story and what gives the feature something to do.
    city_pool = cfg.cities
    bounds = cfg.travel.destination_cities_per_traveller
    dept_presence = employees.groupby(["dept_l3", "base_city"]).size().rename("n").reset_index()
    presence_lookup = {
        (row.dept_l3, row.base_city): row.n for row in dept_presence.itertuples(index=False)
    }
    manager_city = employees.set_index("emp_id")["base_city"].to_dict()

    destinations: dict[str, list[str]] = {}
    for row in travellers.itertuples(index=False):
        options = [c for c in city_pool if c != row.base_city]
        if not options:
            continue
        weights = np.array(
            [1.0 + presence_lookup.get((row.dept_l3, city), 0) for city in options],
            dtype=float,
        )
        boss_city = manager_city.get(row.manager_id) if row.manager_id else None
        if boss_city in options:
            # You visit your manager. This single term is what lifts leader
            # gravity from a curiosity to a real feature.
            weights[options.index(boss_city)] *= 6.0
        count = min(int(rng.integers(bounds["min"], bounds["max"] + 1)), len(options))
        destinations[row.emp_id] = list(
            rng.choice(options, size=count, replace=False, p=weights / weights.sum())
        )

    # Departments that relocate mid-history.
    l3_departments = sorted(employees["dept_l3"].unique())
    relocating = rng.choice(
        l3_departments,
        size=min(cfg.travel.relocation["departments"], len(l3_departments)),
        replace=False,
    )
    for department in relocating:
        moves = {}
        for city, group in simulator.offices_by_city.items():
            if len(group) > 1:
                moves[city] = str(group["office_id"].iloc[int(rng.integers(len(group)))])
        simulator.relocated[str(department)] = moves

    # Trip dates, generated then sorted, because the habit term depends on
    # what happened before.
    window_days = (cfg.end_date - cfg.start_date).days
    rows: list[dict] = []
    for row, count in zip(travellers.itertuples(index=False), trips_each, strict=True):
        offsets = np.sort(rng.integers(0, window_days, size=count))
        pool = destinations.get(row.emp_id)
        if not pool:
            continue
        previous_city: str | None = None
        for offset in offsets:
            if previous_city is not None and rng.random() < cfg.travel.repeat_destination_share:
                city = previous_city
            else:
                city = str(pool[int(rng.integers(len(pool)))])
            previous_city = city
            rows.append(
                {
                    "emp_id": row.emp_id,
                    "destination_city": city,
                    "arrival_date": cfg.start_date + timedelta(days=int(offset)),
                }
            )

    trip_frame = pd.DataFrame(rows).sort_values(["arrival_date", "emp_id"]).reset_index(drop=True)
    trip_frame["trip_id"] = [f"T{i:07d}" for i in range(len(trip_frame))]

    length_cfg = cfg.travel.trip_length_days
    trip_frame["trip_length_days"] = rng.integers(
        length_cfg["min"], length_cfg["max"] + 1, size=len(trip_frame)
    )
    trip_frame["origin_city"] = trip_frame["emp_id"].map(employees.set_index("emp_id")["base_city"])

    # Walk the trips in time order, choosing an office for each.
    chosen: list[str] = []
    diagnostics: list[dict] = []
    for trip in trip_frame.itertuples(index=False):
        employee = simulator.employees.loc[trip.emp_id]
        month_index = int((trip.arrival_date - cfg.start_date).days / 30.44)
        office, diagnostic = simulator.choose_office(
            trip.emp_id, employee, trip.destination_city, month_index
        )
        simulator.record(trip.emp_id, employee, trip.destination_city, office)
        chosen.append(office)
        diagnostics.append(diagnostic)

    trip_frame["visited_office_id"] = chosen
    diagnostic_frame = pd.DataFrame(diagnostics)
    for column in diagnostic_frame.columns:
        trip_frame[f"truth_{column}"] = diagnostic_frame[column].to_numpy()

    taps = _trip_taps(cfg, trip_frame, rng)

    multi = trip_frame[trip_frame["truth_n_candidates"] > 1]
    realised = {
        "trips": int(len(trip_frame)),
        "travellers": int(trip_frame["emp_id"].nunique()),
        "trips_with_a_real_choice": int(len(multi)),
        "mean_candidates": round(float(trip_frame["truth_n_candidates"].mean()), 3),
        "exploration_share_realised": round(float(multi["truth_exploration"].mean()), 4),
        "share_with_habit": round(float(multi["truth_had_habit"].mean()), 4),
        "share_with_leader": round(float(multi["truth_had_leader"].mean()), 4),
        "share_with_team": round(float(multi["truth_had_team"].mean()), 4),
        # The attainable ceiling on multi-candidate trips: an exploring
        # traveller can still be right by luck, at 1/n.
        "accuracy_ceiling": round(
            float(
                1
                - multi["truth_exploration"].mean()
                * (1 - (1.0 / multi["truth_n_candidates"]).mean())
            ),
            4,
        ),
    }
    return trip_frame, taps, realised


def _trip_taps(cfg: Config, trips: pd.DataFrame, rng) -> pd.DataFrame:
    """Badge taps recorded during each trip.

    The published evidence. The label is derivable from these, and the features
    are built from the history of them - so this is the only place the model's
    view of the past comes from.
    """
    rows = []
    for trip in trips.itertuples(index=False):
        n_taps = max(
            1,
            int(
                round(
                    trip.trip_length_days
                    * rng.normal(
                        cfg.travel.taps_per_trip_day["mean"],
                        cfg.travel.taps_per_trip_day["sigma"],
                    )
                )
            ),
        )
        for day in range(min(n_taps, trip.trip_length_days * 3)):
            rows.append(
                {
                    "tap_id": f"{trip.trip_id}-{day}",
                    "emp_id": trip.emp_id,
                    "office_id": trip.visited_office_id,
                    "city": trip.destination_city,
                    "tap_date": trip.arrival_date
                    + timedelta(days=int(day % max(trip.trip_length_days, 1))),
                }
            )
    return pd.DataFrame(rows)


PUBLISHED_TRIP_COLUMNS = [
    "trip_id",
    "emp_id",
    "origin_city",
    "destination_city",
    "arrival_date",
    "trip_length_days",
    "visited_office_id",
]

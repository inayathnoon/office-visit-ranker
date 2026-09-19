"""Cities, offices, and the organisation that travels between them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .rng import choice_from_mix, lognormal_from_mean, substream


def generate_offices(cfg: Config) -> pd.DataFrame:
    """One row per office.

    Office count per city follows a Zipf-ish law, so a hub has a dozen
    candidates and a small city has one. That spread is the point: a ranking
    problem with one candidate is not a ranking problem, and reporting accuracy
    without splitting on city size averages the two together and hides both.
    """
    rng = substream(cfg.seed, "offices")
    cities = cfg.cities
    bounds = cfg.geography.offices_per_city

    ranks = np.arange(1, len(cities) + 1)
    weights = 1.0 / ranks**cfg.geography.office_count_zipf
    counts = np.clip(
        np.round(bounds["max"] * weights / weights.max()).astype(int),
        bounds["min"],
        bounds["max"],
    )

    size_cfg = cfg.geography.office_size
    distance_cfg = cfg.geography.distance_km

    rows: list[dict] = []
    seq = 0
    for city, n_offices in zip(cities, counts, strict=True):
        for index in range(n_offices):
            seq += 1
            rows.append(
                {
                    "office_id": f"OF{seq:04d}",
                    "office_name": f"{city} {index + 1}",
                    "city": city,
                    "headcount": int(
                        max(
                            size_cfg["min"],
                            lognormal_from_mean(rng, size_cfg["mean"], size_cfg["sigma"], 1)[0],
                        )
                    ),
                    # The first office in a city is the central one; the rest
                    # are progressively further out.
                    "distance_km": round(
                        float(
                            min(
                                distance_cfg["max"],
                                lognormal_from_mean(
                                    rng, distance_cfg["mean"], distance_cfg["sigma"], 1
                                )[0]
                                * (0.25 if index == 0 else 1.0),
                            )
                        ),
                        2,
                    ),
                }
            )

    offices = pd.DataFrame(rows)
    offices["offices_in_city"] = offices.groupby("city")["office_id"].transform("count")

    # Each office has a department mix. A traveller is drawn toward offices
    # whose mix resembles their own department, which is the "my people sit
    # there" effect that makes a big office not automatically the answer.
    departments = cfg.org.departments_l1
    mix = rng.dirichlet(np.full(len(departments), 1.2), size=len(offices))
    for i, department in enumerate(departments):
        offices[f"mix_{department}"] = mix[:, i].round(4)
    offices["dominant_department"] = [departments[i] for i in mix.argmax(axis=1)]
    return offices


def generate_org(cfg: Config, offices: pd.DataFrame) -> pd.DataFrame:
    """Employees, their teams, their managers and their base city."""
    rng = substream(cfg.seed, "org")
    n = cfg.profile.n_employees

    teams: list[dict] = []
    for l1 in cfg.org.departments_l1:
        for i2 in range(1, int(rng.integers(cfg.org.l2_per_l1["min"], cfg.org.l2_per_l1["max"] + 1)) + 1):
            l2 = f"{l1} Division {i2}"
            for i3 in range(1, int(rng.integers(cfg.org.l3_per_l2["min"], cfg.org.l3_per_l2["max"] + 1)) + 1):
                l3 = f"{l2} Group {i3}"
                for i4 in range(1, int(rng.integers(cfg.org.l4_per_l3["min"], cfg.org.l4_per_l3["max"] + 1)) + 1):
                    teams.append(
                        {"dept_l1": l1, "dept_l2": l2, "dept_l3": l3, "dept_l4": f"{l3} Team {i4}"}
                    )

    sizes = np.maximum(
        cfg.org.team_size["min"],
        lognormal_from_mean(rng, cfg.org.team_size["mean"], cfg.org.team_size["sigma"], len(teams)),
    )
    team_index = rng.choice(len(teams), size=n, p=sizes / sizes.sum())
    team_frame = pd.DataFrame([teams[i] for i in team_index]).reset_index(drop=True)

    # Base city, weighted by the city's total office headcount.
    by_city = offices.groupby("city", as_index=False)["headcount"].sum()
    city_p = by_city["headcount"].to_numpy(dtype=float)
    city_p = city_p / city_p.sum()
    base_city = rng.choice(by_city["city"].to_numpy(), size=n, p=city_p)

    employee_type = choice_from_mix(rng, cfg.org.employee_type_mix, n)

    employees = pd.DataFrame(
        {
            "emp_id": [f"E{500000 + i}" for i in range(n)],
            "employee_type": employee_type,
            "dept_l1": team_frame["dept_l1"],
            "dept_l2": team_frame["dept_l2"],
            "dept_l3": team_frame["dept_l3"],
            "dept_l4": team_frame["dept_l4"],
            "base_city": base_city,
        }
    )

    # Managers. The first employee of each L4 team manages it, and a share of
    # employees have no manager at all - which is what makes leader gravity a
    # genuinely MISSING feature rather than a null to impute.
    manager_of_team = employees.groupby("dept_l4")["emp_id"].first().to_dict()
    manager = employees["dept_l4"].map(manager_of_team).to_numpy()
    manager = np.where(employees["emp_id"].to_numpy() == manager, None, manager)

    drop_rate = employees["employee_type"].map(cfg.org.no_manager_share_by_type).to_numpy()
    manager = np.where(rng.random(n) < drop_rate, None, manager)
    employees["manager_id"] = manager

    # Everyone sits somewhere. A home office matters because it is what makes
    # leader and team gravity dense: the first version derived a leader's
    # usual office purely from their own travel history, so it fired on 0.3%
    # of trips and the feature was effectively dead. In reality you travel to
    # where your manager and your team already sit.
    home_office = np.empty(n, dtype=object)
    for city, group in offices.groupby("city"):
        members = np.flatnonzero(employees["base_city"].to_numpy() == city)
        if members.size == 0:
            continue
        weights = group["headcount"].to_numpy(dtype=float)
        home_office[members] = rng.choice(
            group["office_id"].to_numpy(), size=members.size, p=weights / weights.sum()
        )
    employees["home_office_id"] = home_office
    return employees


def generate_space_allocation(cfg: Config, offices: pd.DataFrame) -> pd.DataFrame:
    """Which departments hold dedicated space in which office.

    A team with desks of its own in a building is far more likely to use it,
    and unlike the behavioural features this one is knowable on a traveller's
    first ever visit - which is what makes it carry the cold-start case.
    """
    rng = substream(cfg.seed, "space")
    rows = []
    for office in offices.itertuples(index=False):
        for department in cfg.org.departments_l1:
            mix = getattr(office, f"mix_{department}")
            # Dedicated space follows the office's department mix, so it is
            # correlated with the mix feature rather than independent of it.
            probability = cfg.geography.dedicated_space_share * (0.4 + 2.2 * mix)
            if rng.random() < min(probability, 0.85):
                rows.append(
                    {
                        "office_id": office.office_id,
                        "city": office.city,
                        "dept_l1": department,
                        "desks": int(rng.integers(8, 120)),
                    }
                )
    return pd.DataFrame(rows)

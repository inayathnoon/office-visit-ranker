"""Typed access to ``conf/sim.yaml``.

Loaded once, validated, passed explicitly. The weights under ``choice`` are the
answer the model is looking for, so they live here and nowhere else.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "sim.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Profile(_Base):
    n_employees: int
    n_cities: int
    n_months: int
    traveller_share: float
    trips_per_traveller: dict[str, float]
    end_date: date


class Geography(_Base):
    cities: list[str]
    offices_per_city: dict[str, int]
    office_count_zipf: float
    office_size: dict[str, float]
    distance_km: dict[str, float]
    dedicated_space_share: float


class Org(_Base):
    departments_l1: list[str]
    l2_per_l1: dict[str, int]
    l3_per_l2: dict[str, int]
    l4_per_l3: dict[str, int]
    team_size: dict[str, float]
    employee_type_mix: dict[str, float]
    no_manager_share_by_type: dict[str, float]


class Choice(_Base):
    weights: dict[str, float]
    exploration_share: float
    noise_scale: float


class Travel(_Base):
    trip_length_days: dict[str, int]
    destination_cities_per_traveller: dict[str, int]
    repeat_destination_share: float
    taps_per_trip_day: dict[str, float]
    relocation: dict[str, int]


class Config(_Base):
    seed: int
    profile_name: str
    profile: Profile
    geography: Geography
    org: Org
    choice: Choice
    travel: Travel

    @property
    def end_date(self) -> date:
        return self.profile.end_date

    @property
    def start_date(self) -> date:
        return self.end_date - timedelta(days=int(self.profile.n_months * 30.44) - 1)

    @property
    def cities(self) -> list[str]:
        return self.geography.cities[: self.profile.n_cities]

    def month_boundary(self, month_index: int) -> date:
        """First day of the given month index, counting from the window start."""
        return self.start_date + timedelta(days=int(month_index * 30.44))

    @property
    def train_end(self) -> date:
        """Months 1-18 train. Split by time, asserted by a test."""
        return self.month_boundary(18)

    @property
    def validation_end(self) -> date:
        """Months 19-21 validate, 22-24 test."""
        return self.month_boundary(21)


def _coerce_date(value: Any) -> date:
    return value if isinstance(value, date) else datetime.strptime(str(value), "%Y-%m-%d").date()


def load_config(path: str | Path | None = None, profile: str | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text())

    profiles = raw.pop("profiles")
    chosen = profile or os.environ.get("OVR_PROFILE") or raw["profile"]
    if chosen not in profiles:
        raise ValueError(f"Unknown profile {chosen!r}; available: {sorted(profiles)}")

    profile_raw = dict(profiles[chosen])
    profile_raw["end_date"] = _coerce_date(profile_raw["end_date"])

    defined = len(raw["geography"]["cities"])
    if profile_raw["n_cities"] > defined:
        raise ValueError(
            f"Profile {chosen!r} wants {profile_raw['n_cities']} cities but only {defined} "
            "are defined in geography.cities"
        )

    data = dict(raw)
    data["profile"] = Profile(**profile_raw)
    data["profile_name"] = chosen
    return Config(**data)


@lru_cache(maxsize=4)
def get_config(profile: str | None = None) -> Config:
    return load_config(profile=profile)


# --- Filesystem layout -----------------------------------------------------

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRUTH_DIR = DATA_DIR / "truth"
OUT_DIR = REPO_ROOT / "out"
MODEL_DIR = REPO_ROOT / "models"
DOCS_DIR = REPO_ROOT / "docs"
IMG_DIR = DOCS_DIR / "img"
MLRUNS_DIR = REPO_ROOT / "mlruns"
GROUND_TRUTH = TRUTH_DIR / "ground_truth.json"

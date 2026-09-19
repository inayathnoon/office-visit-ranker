"""The feature-availability contract.

Most of this repository is an ordinary ranking problem. This file is the part
worth reading.

A first-time visitor to a city has no habit and no personal history there. A
contractor with no manager in the HR extract has no leader gravity. Those
features are not zero - they are *absent*, and the difference matters:

    imputed zero  =>  "the manager uses none of these offices"
    absent        =>  "we do not know where the manager sits"

The first is a claim the data does not support, and a tree model will happily
learn from it. At scoring time the same zero then appears for two populations
that behave completely differently - people whose manager genuinely prefers
other buildings, and people who have no manager at all.

So instead of imputing, a trip is routed to a model trained on **exactly** the
features it has. The models are keyed by availability mask, and a mask with too
little training data falls back through a declared hierarchy rather than
guessing.

The invariant that makes it safe is asserted, not assumed: no model is ever
handed a row whose mask lacks a feature that model was trained on. See
``tests/test_availability.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..features.build import ALWAYS_AVAILABLE, FEATURE_FAMILIES

# Families that can be missing, in the order they are given up when falling
# back. Least informative first: losing leader gravity costs less than losing
# the traveller's own history, so it is the first thing dropped.
#
# "personal" covers habit and personal history together - they become available
# at exactly the same moment, the traveller's first visit to the city, so
# treating them as separate masks would double the model count and never
# produce a mask that had one without the other.
OPTIONAL_FAMILIES = ("leader", "department", "team", "personal")

FALLBACK_ORDER = OPTIONAL_FAMILIES

# A mask needs at least this many training trips to get its own model. Below
# it, the mask falls back: a model fitted on thirty trips is worse than a more
# general model fitted on three thousand.
MIN_TRAINING_TRIPS = 60


@dataclass(frozen=True)
class AvailabilityMask:
    """Which optional feature families a trip has."""

    families: frozenset[str]

    @property
    def key(self) -> str:
        return "+".join(sorted(self.families)) if self.families else "none"

    def features(self) -> list[str]:
        """Exactly the columns a model for this mask may use."""
        columns: list[str] = []
        for family in ALWAYS_AVAILABLE:
            columns.extend(FEATURE_FAMILIES[family])
        for family in sorted(self.families):
            if family == "personal":
                columns.extend(FEATURE_FAMILIES["habit"])
                columns.extend(FEATURE_FAMILIES["personal_history"])
            else:
                columns.extend(FEATURE_FAMILIES[family])
        return columns

    def covers(self, other: AvailabilityMask) -> bool:
        """True if a model for this mask can score a row with ``other``."""
        return self.families <= other.families

    def drop_one(self) -> AvailabilityMask | None:
        """The next mask down the fallback hierarchy."""
        for family in FALLBACK_ORDER:
            if family in self.families:
                return AvailabilityMask(self.families - {family})
        return None


def mask_for_row(row: pd.Series) -> AvailabilityMask:
    families = set()
    if bool(row.get("available_habit")) and bool(row.get("available_personal_history")):
        families.add("personal")
    for family in ("leader", "team", "department"):
        if bool(row.get(f"available_{family}")):
            families.add(family)
    return AvailabilityMask(frozenset(families))


def mask_keys(frame: pd.DataFrame) -> pd.Series:
    """Vectorised mask key per row."""
    personal = frame["available_habit"].astype(bool) & frame[
        "available_personal_history"
    ].astype(bool)
    parts = []
    for index in range(len(frame)):
        families = set()
        if personal.iloc[index]:
            families.add("personal")
        for family in ("leader", "team", "department"):
            if bool(frame[f"available_{family}"].iloc[index]):
                families.add(family)
        parts.append(AvailabilityMask(frozenset(families)).key)
    return pd.Series(parts, index=frame.index, name="mask_key")


def mask_from_key(key: str) -> AvailabilityMask:
    return AvailabilityMask(frozenset() if key == "none" else frozenset(key.split("+")))


def resolve_model(
    mask: AvailabilityMask, trained: set[str]
) -> tuple[AvailabilityMask, int]:
    """Find the model to score this mask with, and how far it fell back.

    Walks down the hierarchy until it reaches a mask that has a trained model.
    The empty mask always has one, so the walk terminates.
    """
    current: AvailabilityMask | None = mask
    steps = 0
    while current is not None:
        if current.key in trained:
            return current, steps
        current = current.drop_one()
        steps += 1
    raise RuntimeError("no model available, not even for the empty mask")


def training_masks(frame: pd.DataFrame, min_trips: int = MIN_TRAINING_TRIPS) -> list[str]:
    """Masks with enough training trips to deserve their own model.

    The empty mask is always included: it is the floor of the fallback
    hierarchy and something has to be able to score a trip that has nothing.
    """
    per_trip = frame.drop_duplicates("trip_id")
    counts = per_trip["mask_key"].value_counts()
    keys = {key for key, count in counts.items() if count >= min_trips}
    keys.add("none")
    return sorted(keys)


def availability_report(frame: pd.DataFrame) -> pd.DataFrame:
    """How trips are distributed across masks, and where they get scored."""
    per_trip = frame.drop_duplicates("trip_id")
    trained = set(training_masks(frame[frame["split"] == "train"]))
    rows = []
    for key, count in per_trip["mask_key"].value_counts().items():
        mask = mask_from_key(str(key))
        resolved, steps = resolve_model(mask, trained)
        rows.append(
            {
                "mask": key,
                "trips": int(count),
                "share": round(count / len(per_trip), 4),
                "n_features": len(mask.features()),
                "scored_by": resolved.key,
                "fallback_steps": steps,
                "has_own_model": steps == 0,
            }
        )
    return pd.DataFrame(rows).sort_values("trips", ascending=False).reset_index(drop=True)

"""Ranking metrics, and the segments that matter.

With exactly one relevant office per trip, several of the standard ranking
metrics collapse into simpler things, and saying so is more useful than
quoting them as though they were independent:

* **NDCG@1** equals Hit@1 - with a single relevant item at rank 1 the
  discount is 1 and the ideal DCG is 1.
* **MRR** is the mean of 1/rank of the correct office.
* **Recall@k** is the share of trips where the correct office is in the top k.

NDCG@3 is the one that carries information the others do not, because it
distinguishes "second" from "third" when the top-1 is wrong.

Everything is reported by segment as well as overall. A headline that averages
a first-time visitor to a twelve-office city with a repeat visitor to a
two-office city describes neither.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ranks(frame: pd.DataFrame, score_column: str) -> pd.Series:
    """Rank of each candidate within its trip, 1 = best.

    Ties are broken by office_id rather than by position, so a baseline that
    scores every candidate identically gets a deterministic - and honestly
    mediocre - result rather than whatever order the frame happened to be in.
    """
    ordered = frame.sort_values(
        ["trip_id", score_column, "office_id"], ascending=[True, False, True]
    )
    return ordered.groupby("trip_id", sort=False).cumcount().add(1).reindex(frame.index)


def trip_level(frame: pd.DataFrame, score_column: str = "score") -> pd.DataFrame:
    """One row per trip: the rank the correct office was given."""
    working = frame.copy()
    working["rank"] = _ranks(working, score_column)
    correct = working[working["label"] == 1].copy()

    group_size = working.groupby("trip_id", sort=False).size().rename("candidates")
    correct = correct.join(group_size, on="trip_id")
    correct["reciprocal_rank"] = 1.0 / correct["rank"]
    correct["hit_at_1"] = (correct["rank"] == 1).astype(float)
    correct["recall_at_2"] = (correct["rank"] <= 2).astype(float)
    correct["recall_at_3"] = (correct["rank"] <= 3).astype(float)
    # With one relevant item, DCG@k = 1/log2(rank+1) when rank <= k, else 0,
    # and the ideal DCG is 1 - so NDCG is just the discount.
    correct["ndcg_at_1"] = np.where(correct["rank"] <= 1, 1.0, 0.0)
    correct["ndcg_at_3"] = np.where(correct["rank"] <= 3, 1.0 / np.log2(correct["rank"] + 1), 0.0)
    return correct


METRICS = ["ndcg_at_1", "ndcg_at_3", "reciprocal_rank", "recall_at_2", "hit_at_1"]


def summarise(per_trip: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    if not by:
        row: dict[str, float] = {
            metric: round(float(per_trip[metric].mean()), 4) for metric in METRICS
        }
        row["trips"] = float(len(per_trip))
        frame = pd.DataFrame([row])
        frame["trips"] = frame["trips"].astype(int)
        return frame
    grouped = per_trip.groupby(by, sort=True)
    out = grouped[METRICS].mean().round(4)
    out["trips"] = grouped.size()
    return out.reset_index()


def add_segments(per_trip: pd.DataFrame) -> pd.DataFrame:
    """The segments worth splitting on.

    City size and first-versus-repeat visit are the two that change the problem
    rather than just the population: one changes how many wrong answers there
    are, the other changes which features exist.
    """
    out = per_trip.copy()
    out["visitor_type"] = np.where(
        out["is_first_ever_visit_to_city"] > 0, "first visit", "repeat visit"
    )
    out["city_size"] = pd.cut(
        out["candidates"],
        bins=[0, 2, 4, 8, 100],
        labels=["2 offices", "3-4 offices", "5-8 offices", "9+ offices"],
    )
    return out


def evaluate_scored(scored: pd.DataFrame, score_column: str = "score") -> dict:
    """Overall and segmented metrics for one scored frame."""
    per_trip = add_segments(trip_level(scored, score_column))
    return {
        "overall": summarise(per_trip),
        "by_visitor_type": summarise(per_trip, ["visitor_type"]),
        "by_city_size": summarise(per_trip, ["city_size"]),
        "by_mask": summarise(per_trip, ["mask_key"]),
        "by_scored_by": (summarise(per_trip, ["scored_by"]) if "scored_by" in per_trip else None),
        "per_trip": per_trip,
    }


def lift_table(results: dict[str, pd.DataFrame], reference: str) -> pd.DataFrame:
    """Every model's headline metrics, and its lift over a reference rule."""
    rows: list[dict] = []
    for name, overall in results.items():
        row: dict = {"model": name}
        row.update({metric: float(overall[metric].iloc[0]) for metric in METRICS})
        row["trips"] = int(overall["trips"].iloc[0])
        rows.append(row)
    table = pd.DataFrame(rows)

    base = table[table["model"] == reference]
    if not base.empty:
        for metric in METRICS:
            denominator = float(base[metric].iloc[0])
            lift = (
                table[metric] / denominator - 1
                if denominator > 0
                else pd.Series(np.nan, index=table.index)
            )
            table[f"lift_{metric}"] = lift.round(4)
    return table.sort_values("ndcg_at_3", ascending=False).reset_index(drop=True)

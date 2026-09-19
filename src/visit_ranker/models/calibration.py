"""Calibrating the top-1 score, and knowing when to abstain.

A LambdaMART score is not a probability. It is a number whose ordering is
meaningful and whose magnitude is not, and sending "we are 87% sure" to a
workplace-services team on the basis of a raw margin would be inventing a
quantity.

So the top-1 margin is mapped to a probability by isotonic regression fitted
on the validation window - monotone, non-parametric, and it cannot invent
structure the ranking did not already have.

**Abstention** is what the calibrated number is for. When the top-1 probability
is below a threshold, the service returns the top two as a tie and the message
covers both buildings. That is a real operational option: access can be granted
to two offices, and a meal instruction can name two receptions. The cost is
paid in the message, not in a wrong door.

The threshold is not chosen by taste. The coverage/accuracy trade-off is
computed across the whole range and published, because moving it is a business
decision about how often you are willing to send a vaguer instruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class TopOneCalibrator:
    isotonic: IsotonicRegression

    def probability(self, margin: np.ndarray) -> np.ndarray:
        return np.clip(self.isotonic.predict(margin), 0.001, 0.999)


def top_one_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per trip: the top-ranked office, its margin, and whether it was right.

    The margin - top score minus runner-up - rather than the raw top score,
    because the raw score has no fixed zero across trips or across models in
    the family. The gap between first and second is comparable; the height of
    the first is not.
    """
    ordered = scored.sort_values(["trip_id", "score"], ascending=[True, False])
    grouped = ordered.groupby("trip_id", sort=False)

    top = grouped.head(1).set_index("trip_id")
    second = grouped.nth(1)
    second = second.set_index("trip_id")["score"] if len(second) else pd.Series(dtype=float)

    frame = pd.DataFrame(
        {
            "predicted_office": top["office_id"],
            "score": top["score"],
            "runner_up_score": top.index.map(second).astype(float),
            "correct": top["label"].astype(float),
            "candidates": grouped.size(),
            "mask_key": top["mask_key"],
            "scored_by": top.get("scored_by", pd.Series(index=top.index, dtype=object)),
            "is_first_ever_visit_to_city": top["is_first_ever_visit_to_city"],
        }
    )
    # A single-candidate trip has no runner-up; its margin is defined as large,
    # because there is nothing to be uncertain between.
    frame["margin"] = np.where(
        frame["runner_up_score"].isna(), 10.0, frame["score"] - frame["runner_up_score"]
    )
    return frame.reset_index()


def fit_calibrator(validation_top: pd.DataFrame) -> TopOneCalibrator:
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(validation_top["margin"], validation_top["correct"])
    return TopOneCalibrator(isotonic=isotonic)


def reliability(
    top: pd.DataFrame, probability_column: str = "probability", bins: int = 10
) -> pd.DataFrame:
    """Predicted probability against observed accuracy, by bin."""
    working = top.copy()
    working["bin"] = pd.qcut(working[probability_column], q=bins, duplicates="drop", labels=False)
    out = (
        working.groupby("bin", sort=True)
        .agg(
            predicted=(probability_column, "mean"),
            observed=("correct", "mean"),
            trips=("correct", "size"),
        )
        .reset_index(drop=True)
    )
    return out.round(4)


def expected_calibration_error(reliability_table: pd.DataFrame) -> float:
    weights = reliability_table["trips"] / reliability_table["trips"].sum()
    return float(
        (weights * (reliability_table["predicted"] - reliability_table["observed"]).abs()).sum()
    )


def abstention_curve(top: pd.DataFrame, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Coverage against accuracy as the abstention threshold moves.

    ``coverage`` is the share of trips answered with a single office.
    ``accuracy_covered`` is how often that single answer is right.
    ``accuracy_with_fallback`` counts an abstention as correct when the right
    office is in the top two - which is what the service actually sends.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.95, 20)

    rows = []
    for threshold in thresholds:
        answered = top["probability"] >= threshold
        covered = top[answered]
        abstained = top[~answered]
        accuracy_with_fallback = (
            covered["correct"].sum() + abstained["correct_in_top2"].sum()
        ) / max(len(top), 1)
        rows.append(
            {
                "threshold": round(float(threshold), 3),
                "coverage": round(float(answered.mean()), 4),
                "accuracy_covered": (
                    round(float(covered["correct"].mean()), 4) if len(covered) else np.nan
                ),
                "accuracy_with_fallback": round(float(accuracy_with_fallback), 4),
                "abstentions": int((~answered).sum()),
            }
        )
    return pd.DataFrame(rows)


def add_top2_flag(top: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Whether the correct office is within the top two for each trip."""
    ordered = scored.sort_values(["trip_id", "score"], ascending=[True, False])
    rank = ordered.groupby("trip_id", sort=False).cumcount().add(1)
    ordered = ordered.assign(rank=rank)
    correct = ordered[ordered["label"] == 1][["trip_id", "rank"]]
    correct["correct_in_top2"] = (correct["rank"] <= 2).astype(float)
    return top.merge(correct[["trip_id", "correct_in_top2"]], on="trip_id", how="left").fillna(
        {"correct_in_top2": 0.0}
    )


def recommend_threshold(curve: pd.DataFrame, minimum_coverage: float = 0.70) -> dict:
    """Pick a threshold, and say what it costs.

    The rule: the highest threshold that still answers at least
    ``minimum_coverage`` of trips with a single office. Chosen this way rather
    than by maximising accuracy, because a threshold that abstains on
    everything has perfect accuracy on the two trips it answers and is useless.
    """
    eligible = curve[curve["coverage"] >= minimum_coverage]
    chosen = curve.iloc[0] if eligible.empty else eligible.iloc[eligible["threshold"].argmax()]
    baseline = curve.iloc[0]
    return {
        "threshold": float(chosen["threshold"]),
        "coverage": float(chosen["coverage"]),
        "accuracy_covered": float(chosen["accuracy_covered"]),
        "accuracy_with_fallback": float(chosen["accuracy_with_fallback"]),
        "accuracy_without_abstention": float(baseline["accuracy_covered"]),
        "minimum_coverage": minimum_coverage,
    }

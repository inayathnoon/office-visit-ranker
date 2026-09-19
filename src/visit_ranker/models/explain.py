"""SHAP explanations, and the comparison the simulator makes possible.

Two jobs.

**Explaining a single trip.** The service returns the three features that
pushed the winning office up, because "go to building 3" is an instruction
nobody audits and "go to building 3, because you went there last time and your
team sits there" is one somebody can correct.

**Checking the model found the right structure.** The simulator's choice
weights are known, so global SHAP importance can be compared against them.
That is a stronger test than any accuracy number: a model can rank well by
leaning on one dominant feature while getting the rest of the structure wrong,
and the comparison catches it.

The comparison is between two quantities that are not on the same scale - a
logit coefficient and a mean absolute SHAP value - so both are normalised to
shares and the *ordering* and relative magnitude are what is compared, not the
absolute numbers. Saying so matters; presenting them as directly comparable
would be overclaiming.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Which engineered features correspond to which term in the simulator's
# utility function. A term may be represented by several features - habit is
# both the raw flag and its decayed variant - and they are summed.
TRUTH_TO_FEATURES = {
    "habit": ["habit_last_visit", "habit_last_visit_decayed"],
    "leader": ["leader_visit_share", "leader_is_top1"],
    "team": ["team_visit_share", "team_visit_share_decayed", "team_is_top1"],
    "l3_department": [
        "l3_visit_share",
        "l3_visit_share_decayed",
        "l3_is_top1",
        "dept_recent_visit_share",
    ],
    "size": ["office_log_headcount", "office_is_largest_in_city"],
    "distance": ["office_distance_km"],
    "dept_mix": ["office_dept_mix_similarity"],
    "dedicated_space": ["office_has_dedicated_space", "office_dedicated_desks"],
}


def shap_values(booster, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """SHAP values from LightGBM's own implementation.

    Uses ``pred_contrib`` rather than the shap package's explainer: for a tree
    ensemble it is the same TreeSHAP computation, it is exact, and it avoids
    carrying a second implementation that could drift.
    """
    contributions = booster.predict(frame[columns], pred_contrib=True)
    # The last column is the expected value, not a feature.
    return np.asarray(contributions)[:, :-1]


def global_importance(booster, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = shap_values(booster, frame, columns)
    importance = np.abs(values).mean(axis=0)
    table = pd.DataFrame({"feature": columns, "mean_abs_shap": importance})
    table["share"] = table["mean_abs_shap"] / max(table["mean_abs_shap"].sum(), 1e-12)
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def compare_with_truth(importance: pd.DataFrame, truth_weights: dict[str, float]) -> pd.DataFrame:
    """Recovered importance against the simulator's own weights.

    Both are normalised to shares of their own total. They are different
    quantities - a utility coefficient and a mean absolute attribution - so
    what is being compared is the ordering and the rough proportions, not the
    numbers themselves.
    """
    by_feature = importance.set_index("feature")["mean_abs_shap"]
    rows = []
    for term, features in TRUTH_TO_FEATURES.items():
        present = [f for f in features if f in by_feature.index]
        rows.append(
            {
                "utility_term": term,
                "features": ", ".join(present) if present else "(not available to this model)",
                "recovered_shap": float(by_feature.reindex(present).sum()) if present else 0.0,
                "planted_weight": float(truth_weights.get(term, 0.0)),
            }
        )
    table = pd.DataFrame(rows)
    table["recovered_share"] = (
        table["recovered_shap"] / max(table["recovered_shap"].sum(), 1e-12)
    ).round(4)
    table["planted_share"] = (
        table["planted_weight"] / max(table["planted_weight"].sum(), 1e-12)
    ).round(4)
    table["recovered_rank"] = table["recovered_share"].rank(ascending=False).astype(int)
    table["planted_rank"] = table["planted_share"].rank(ascending=False).astype(int)
    table["rank_error"] = (table["recovered_rank"] - table["planted_rank"]).abs()
    return table.sort_values("planted_share", ascending=False).reset_index(drop=True)


def rank_correlation(comparison: pd.DataFrame) -> float:
    """Spearman correlation between recovered and planted importance ranks."""
    from scipy.stats import spearmanr

    result = spearmanr(comparison["recovered_share"], comparison["planted_share"])
    return round(float(result.statistic), 4)


def explain_trip(
    booster,
    frame: pd.DataFrame,
    columns: list[str],
    top_k: int = 3,
) -> pd.DataFrame:
    """Per-candidate SHAP contributions for the offices of one trip."""
    values = shap_values(booster, frame, columns)
    rows = []
    for position, (_, row) in enumerate(frame.iterrows()):
        contributions = pd.Series(values[position], index=columns).sort_values(
            key=np.abs, ascending=False
        )
        drivers = contributions.head(top_k)
        rows.append(
            {
                "office_id": row["office_id"],
                "score": row.get("score", np.nan),
                "is_correct": bool(row["label"]),
                "top_drivers": "; ".join(f"{name} {value:+.3f}" for name, value in drivers.items()),
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

"""LambdaMART rankers, one per availability mask, plus the baselines.

A trip is a ranking group. That framing is the whole reason this is not a
classification problem: the question is not "will this person use office A",
it is "of these five offices, which one" - and the answer must be a ranking
because the service sends one instruction, with a fallback.

LightGBM's ``lambdarank`` objective optimises NDCG directly over groups, which
is the right loss here. A binary classifier trained on the same rows would
optimise per-office log loss and be indifferent to whether the right office
came first or third within a trip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .availability import (
    AvailabilityMask,
    mask_from_key,
    resolve_model,
    training_masks,
)

BASE_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3],
    "learning_rate": 0.06,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambdarank_truncation_level": 12,
    "verbose": -1,
}


@dataclass
class RankerFamily:
    """One model per availability mask, plus the routing between them."""

    models: dict[str, lgb.Booster] = field(default_factory=dict)
    features: dict[str, list[str]] = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def trained_keys(self) -> set[str]:
        return set(self.models)

    def route(self, mask_key: str) -> tuple[str, int]:
        resolved, steps = resolve_model(mask_from_key(mask_key), self.trained_keys)
        return resolved.key, steps

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score every row with the model its mask routes to.

        Rows are grouped by mask so each model is called once, and the columns
        handed to it are exactly the ones it was trained on - which is the
        invariant the availability contract exists to guarantee.
        """
        out = frame.copy()
        out["score"] = np.nan
        out["scored_by"] = ""
        out["fallback_steps"] = 0

        for mask_key, group in frame.groupby("mask_key", sort=False):
            model_key, steps = self.route(str(mask_key))
            booster = self.models[model_key]
            columns = self.features[model_key]

            # The contract, enforced at the moment of use. A model must never
            # be handed a feature that this row's mask does not have.
            allowed = set(mask_from_key(str(mask_key)).features())
            illegal = [c for c in columns if c not in allowed]
            if illegal:
                raise RuntimeError(
                    f"model {model_key!r} would read {illegal} from a row with mask "
                    f"{mask_key!r}, which does not have them"
                )

            out.loc[group.index, "score"] = booster.predict(group[columns])
            out.loc[group.index, "scored_by"] = model_key
            out.loc[group.index, "fallback_steps"] = steps
        return out


def _groups(frame: pd.DataFrame) -> np.ndarray:
    """Group sizes for LightGBM, in the order the frame is sorted."""
    return frame.groupby("trip_id", sort=False).size().to_numpy()


def train_family(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    params: dict | None = None,
    num_boost_round: int = 300,
    seed: int = 0,
) -> RankerFamily:
    """Fit one ranker per availability mask that has enough training trips."""
    params = dict(BASE_PARAMS, **(params or {}), seed=seed)
    family = RankerFamily(params=params)

    for key in training_masks(train):
        mask = mask_from_key(key)
        columns = mask.features()

        # A model for mask M trains on every trip whose mask COVERS M: those
        # trips all have the features M uses. Training only on exact matches
        # would starve the general models, which are precisely the ones that
        # have to carry the hardest cases.
        subset = train[train["mask_key"].map(lambda k: mask.covers(mask_from_key(str(k))))]
        if subset.empty:
            continue
        subset = subset.sort_values(["trip_id", "office_id"])

        dataset = lgb.Dataset(
            subset[columns], label=subset["label"], group=_groups(subset), free_raw_data=False
        )
        valid_sets = [dataset]
        valid_names = ["train"]

        validation_subset = validation[
            validation["mask_key"].map(lambda k: mask.covers(mask_from_key(str(k))))
        ]
        if not validation_subset.empty:
            validation_subset = validation_subset.sort_values(["trip_id", "office_id"])
            valid_sets.append(
                lgb.Dataset(
                    validation_subset[columns],
                    label=validation_subset["label"],
                    group=_groups(validation_subset),
                    reference=dataset,
                    free_raw_data=False,
                )
            )
            valid_names.append("validation")

        booster = lgb.train(
            params,
            dataset,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(40, verbose=False)] if len(valid_sets) > 1 else None,
        )
        family.models[key] = booster
        family.features[key] = columns

    return family


# --- Baselines --------------------------------------------------------------
#
# Every baseline is a rule somebody would actually propose in the meeting where
# this model is pitched, which is the only reason to implement them. A lift
# table against "just send them to the biggest office" is the honest way to
# report what the model is worth.


def baseline_scores(frame: pd.DataFrame, name: str) -> pd.Series:
    """Score candidates by a simple rule. Higher is better, as with the model."""
    if name == "most_popular_in_city":
        return frame["office_log_headcount"]
    if name == "employee_last_choice":
        # Falls back to office size when the traveller has no history, which is
        # what anyone implementing this rule would actually do.
        return frame["habit_last_visit"] * 10.0 + frame["office_log_headcount"] * 0.01
    if name == "leader_most_frequent":
        return frame["leader_is_top1"] * 10.0 + frame["office_log_headcount"] * 0.01
    if name == "team_most_frequent":
        return frame["team_is_top1"] * 10.0 + frame["office_log_headcount"] * 0.01
    if name == "nearest_to_centre":
        return -frame["office_distance_km"]
    raise ValueError(f"unknown baseline {name!r}")


BASELINES = (
    "most_popular_in_city",
    "nearest_to_centre",
    "employee_last_choice",
    "leader_most_frequent",
    "team_most_frequent",
)


def fit_logistic_baseline(train: pd.DataFrame, columns: list[str]):
    """Logistic regression on the same features, scored per candidate.

    The point of comparison for "was the ranking objective worth it". It sees
    identical features and optimises per-row log loss instead of NDCG over the
    group.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
    )
    model.fit(train[columns], train["label"])
    return model


@dataclass
class TrainedBaselines:
    logistic: object | None
    logistic_columns: list[str]

    def score(self, frame: pd.DataFrame, name: str) -> pd.Series:
        if name == "logistic_regression":
            if self.logistic is None:
                raise RuntimeError("logistic baseline was not fitted")
            probabilities = self.logistic.predict_proba(frame[self.logistic_columns])[:, 1]
            return pd.Series(probabilities, index=frame.index)
        return baseline_scores(frame, name)


def train_baselines(train: pd.DataFrame) -> TrainedBaselines:
    # The logistic baseline is fitted on the features available to EVERY trip,
    # so it can score the whole test set without its own routing layer. Giving
    # it the full feature set and imputing zeros would be exactly the mistake
    # the availability contract exists to avoid.
    columns = AvailabilityMask(frozenset()).features()
    return TrainedBaselines(
        logistic=fit_logistic_baseline(train, columns), logistic_columns=columns
    )

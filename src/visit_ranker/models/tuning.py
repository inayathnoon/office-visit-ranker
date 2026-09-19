"""Optuna tuning, on the time-based split and nothing else.

The search space is small on purpose. With a few thousand trips, an aggressive
search over a wide space finds a configuration that suits the validation window
rather than the problem, and the test window then disappoints. The parameters
here are the ones that actually change a LambdaMART's behaviour on small
groups: tree size, regularisation, and how deep the lambda truncation goes.

Every trial trains on months 1-18 and is scored on 19-21. The test window is
never touched during tuning - it is opened once, at the end, and the boundaries
are asserted in tests/test_splits.py.
"""

from __future__ import annotations

import optuna
import pandas as pd

from . import evaluate, ranker

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(trial: optuna.Trial, train: pd.DataFrame, validation: pd.DataFrame) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 60),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "lambdarank_truncation_level": trial.suggest_int("lambdarank_truncation_level", 5, 15),
    }
    family = ranker.train_family(train, validation, params=params, num_boost_round=250)
    scored = family.score(validation)
    per_trip = evaluate.trip_level(scored)
    return float(per_trip["ndcg_at_3"].mean())


def tune_params(
    train: pd.DataFrame, validation: pd.DataFrame, n_trials: int = 20, seed: int = 0
) -> dict:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: _objective(t, train, validation), n_trials=n_trials)
    return {
        "best_params": study.best_params,
        "best_validation_ndcg_at_3": round(float(study.best_value), 4),
        "trials": n_trials,
        "history": [{"number": t.number, "value": t.value, **t.params} for t in study.trials],
    }


def split_boundaries(frame: pd.DataFrame) -> pd.DataFrame:
    """First and last date in each split, for the assertion in the tests."""
    working = frame.copy()
    working["arrival_date"] = pd.to_datetime(working["arrival_date"])
    return (
        working.groupby("split")
        .agg(
            first=("arrival_date", "min"),
            last=("arrival_date", "max"),
            trips=("trip_id", "nunique"),
        )
        .reset_index()
    )


def assert_no_overlap(frame: pd.DataFrame) -> dict:
    """Train must end before validation starts, and so on.

    A random split would put two trips by the same traveller weeks apart on
    opposite sides of the boundary; the habit feature then carries the answer
    across, and the score comes back flattering.
    """
    boundaries = split_boundaries(frame).set_index("split")
    problems = []
    if {"train", "validation"} <= set(boundaries.index) and boundaries.loc[
        "train", "last"
    ] >= boundaries.loc["validation", "first"]:
        problems.append("train overlaps validation")
    if {"validation", "test"} <= set(boundaries.index) and boundaries.loc[
        "validation", "last"
    ] >= boundaries.loc["test", "first"]:
        problems.append("validation overlaps test")
    return {"passed": not problems, "problems": problems, "boundaries": boundaries.reset_index()}

"""Dagster asset graph, with MLflow tracking and a drift monitor.

# NOTE: no `from __future__ import annotations` here. Dagster resolves the
# context parameter's type at decoration time and postponed annotations turn
# it into a string it cannot match.

Assets are thin wrappers over functions that work standalone, so the pipeline
runs from the Makefile or a REPL without Dagster present.
"""

import json

import mlflow
import numpy as np
import pandas as pd
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    AssetSelection,
    Definitions,
    MetadataValue,
    Output,
    ScheduleDefinition,
    asset,
    asset_check,
    define_asset_job,
)

from ..config import GROUND_TRUTH, MLRUNS_DIR, OUT_DIR, load_config
from ..features.build import build_candidates
from ..gen.run import generate_all
from ..models import availability, evaluate, ranker
from ..models.tuning import assert_no_overlap
from ..reporting import charts as charts_module
from ..reporting.results import run_pipeline


@asset(
    group_name="data",
    compute_kind="python",
    description="Offices, employees, space allocation, trips and trip taps.",
)
def source_data(context: AssetExecutionContext) -> Output[dict]:
    cfg = load_config()
    result = generate_all(cfg)
    realised = result["ground_truth"]["realised"]
    return Output(
        result["row_counts"],
        metadata={
            "trips": MetadataValue.int(realised["trips"]),
            "mean_candidates": MetadataValue.float(realised["mean_candidates"]),
            "accuracy_ceiling": MetadataValue.float(realised["accuracy_ceiling"]),
        },
    )


@asset(
    group_name="features",
    deps=[source_data],
    compute_kind="python",
    description="Candidate rows with as-of features and an availability mask.",
)
def candidate_features(context: AssetExecutionContext) -> Output[dict]:
    cfg = load_config()
    frame = build_candidates(cfg)
    frame["mask_key"] = availability.mask_keys(frame)

    splits = assert_no_overlap(frame)
    if not splits["passed"]:
        raise RuntimeError(f"time split is not clean: {splits['problems']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_DIR / "candidates.parquet", index=False)
    report = availability.availability_report(frame)
    return Output(
        {"rows": len(frame)},
        metadata={
            "rows": MetadataValue.int(len(frame)),
            "trips": MetadataValue.int(int(frame["trip_id"].nunique())),
            "availability": MetadataValue.md(report.to_markdown(index=False)),
            "splits": MetadataValue.md(splits["boundaries"].to_markdown(index=False)),
        },
    )


@asset(
    group_name="models",
    deps=[candidate_features],
    compute_kind="python",
    description="One LambdaMART per availability mask, evaluated against every baseline.",
)
def ranker_family(context: AssetExecutionContext) -> Output[dict]:
    cfg = load_config()
    result = run_pipeline(cfg)

    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment("office-visit-ranker")
    for row in result["lift"].itertuples(index=False):
        with mlflow.start_run(run_name=str(row.model)):
            mlflow.log_params({"model": row.model, "profile": cfg.profile_name, "seed": cfg.seed})
            mlflow.log_metrics(
                {
                    "ndcg_at_1": float(row.ndcg_at_1),
                    "ndcg_at_3": float(row.ndcg_at_3),
                    "mrr": float(row.reciprocal_rank),
                    "recall_at_2": float(row.recall_at_2),
                    "trips": float(row.trips),
                }
            )

    (OUT_DIR / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "lift": result["lift"].to_dict(orient="records"),
                "threshold": result["threshold"],
                "ece": result["ece"],
                "rank_correlation": result["rank_correlation"],
            },
            indent=2,
            default=float,
        )
    )
    headline = result["lift"][result["lift"]["model"] == "lambdamart_family"].iloc[0]
    return Output(
        {"ndcg_at_3": float(headline["ndcg_at_3"])},
        metadata={
            "lift": MetadataValue.md(result["lift"].to_markdown(index=False)),
            "ndcg_at_1": MetadataValue.float(float(headline["ndcg_at_1"])),
            "ceiling": MetadataValue.float(result["truth"]["realised"]["accuracy_ceiling"]),
        },
    )


@asset(
    group_name="serving",
    deps=[ranker_family],
    compute_kind="python",
    description="Next-7-days rankings for every in-flight trip, written to Parquet.",
)
def batch_rankings(context: AssetExecutionContext) -> Output[dict]:
    """The batch job a workplace-services team would actually consume.

    Scores the most recent window of trips and writes a ranked list per trip.
    Written as a full replace rather than an append: a re-run for the same
    window must produce the same file, or nobody can tell a model change from
    a rerun.
    """
    frame = pd.read_parquet(OUT_DIR / "candidates.parquet")
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    family = ranker.train_family(train, validation)

    upcoming = frame[frame["split"] == "test"]
    scored = family.score(upcoming).sort_values(["trip_id", "score"], ascending=[True, False])
    scored["rank"] = scored.groupby("trip_id", sort=False).cumcount() + 1
    output = scored[scored["rank"] <= 3][
        [
            "trip_id",
            "emp_id",
            "destination_city",
            "arrival_date",
            "office_id",
            "rank",
            "score",
            "mask_key",
            "scored_by",
            "fallback_steps",
        ]
    ]
    path = OUT_DIR / "batch_rankings.parquet"
    output.to_parquet(path, index=False)
    return Output(
        {"rows": len(output)},
        metadata={
            "trips": MetadataValue.int(int(output["trip_id"].nunique())),
            "path": MetadataValue.path(str(path)),
        },
    )


@asset(
    group_name="monitoring",
    deps=[ranker_family],
    compute_kind="python",
    description="Feature drift, prediction shift, and a backtest of NDCG once actuals land.",
)
def monitoring(context: AssetExecutionContext) -> Output[dict]:
    """Monitoring, as a stub with the shape of the real thing.

    Drift is computed between the training window and the most recent one, on
    the features the model actually uses. The backtest recomputes NDCG on the
    latest window now that the labels exist - which in production is the only
    honest measure, because a live ranking has no label until somebody walks
    through a door.
    """
    frame = pd.read_parquet(OUT_DIR / "candidates.parquet")
    train = frame[frame["split"] == "train"]
    recent = frame[frame["split"] == "test"]

    from ..features.build import FEATURE_COLUMNS

    drift = []
    for column in FEATURE_COLUMNS:
        drift.append(
            {
                "feature": column,
                "psi": round(_psi(train[column].to_numpy(), recent[column].to_numpy()), 4),
                "train_mean": round(float(train[column].mean()), 4),
                "recent_mean": round(float(recent[column].mean()), 4),
            }
        )
    drift_frame = pd.DataFrame(drift).sort_values("psi", ascending=False)

    family = ranker.train_family(train, frame[frame["split"] == "validation"])
    scored = family.score(recent)
    per_trip = evaluate.trip_level(scored)
    backtest = {
        "trips": int(len(per_trip)),
        "ndcg_at_1": round(float(per_trip["ndcg_at_1"].mean()), 4),
        "ndcg_at_3": round(float(per_trip["ndcg_at_3"].mean()), 4),
    }

    drift_frame.to_csv(OUT_DIR / "feature_drift.csv", index=False)
    return Output(
        {"backtest": backtest},
        metadata={
            "worst_psi": MetadataValue.float(float(drift_frame["psi"].max())),
            "drifting_features": MetadataValue.int(int((drift_frame["psi"] > 0.25).sum())),
            "drift": MetadataValue.md(drift_frame.head(10).to_markdown(index=False)),
            "backtest": MetadataValue.json(backtest),
        },
    )


@asset(
    group_name="reporting",
    deps=[ranker_family],
    compute_kind="python",
    description="README charts.",
)
def charts(context: AssetExecutionContext) -> Output[dict]:
    result = run_pipeline(load_config())
    paths = charts_module.render_all(result)
    return Output(
        {"charts": len(paths)}, metadata={"charts": MetadataValue.json([str(p) for p in paths])}
    )


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index, with bucket count capped by sample size."""
    usable = min(len(expected), len(actual))
    buckets = min(buckets, max(usable // 100, 2))
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_share = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_share = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


@asset_check(asset=ranker_family, name="accuracy_below_the_simulator_ceiling", blocking=True)
def check_below_ceiling() -> AssetCheckResult:
    """Nothing may score above what the exploration share makes attainable.

    This is the leakage detector. The simulator decides 12% of choices at
    random, so no feature can predict them; a model above the ceiling is
    reading something it should not have, and blocking is the right severity
    because the number would be presented as real.
    """
    summary = json.loads((OUT_DIR / "pipeline_summary.json").read_text())
    truth = json.loads(GROUND_TRUTH.read_text())
    ceiling = float(truth["realised"]["accuracy_ceiling"])
    headline = next(row for row in summary["lift"] if row["model"] == "lambdamart_family")
    achieved = float(headline["ndcg_at_1"])
    return AssetCheckResult(
        passed=achieved <= ceiling + 1e-9,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "hit_at_1": MetadataValue.float(achieved),
            "ceiling": MetadataValue.float(ceiling),
            "share_of_ceiling": MetadataValue.float(round(achieved / ceiling, 4)),
        },
    )


@asset_check(asset=candidate_features, name="no_model_reads_a_missing_feature", blocking=True)
def check_availability_contract() -> AssetCheckResult:
    """The invariant the availability contract exists to guarantee."""
    frame = pd.read_parquet(OUT_DIR / "candidates.parquet")
    train = frame[frame["split"] == "train"]
    trained = set(availability.training_masks(train))

    violations = []
    for key in sorted(frame["mask_key"].unique()):
        mask = availability.mask_from_key(str(key))
        resolved, _ = availability.resolve_model(mask, trained)
        allowed = set(mask.features())
        illegal = [c for c in resolved.features() if c not in allowed]
        if illegal:
            violations.append({"mask": key, "model": resolved.key, "illegal": illegal})

    return AssetCheckResult(
        passed=not violations,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "masks_checked": MetadataValue.int(int(frame["mask_key"].nunique())),
            "violations": MetadataValue.json(violations),
        },
    )


full_refresh_job = define_asset_job(name="full_refresh", selection=AssetSelection.all())

nightly_schedule = ScheduleDefinition(
    name="nightly_retrain",
    job=full_refresh_job,
    cron_schedule="0 3 * * *",
    execution_timezone="UTC",
)

defs = Definitions(
    assets=[source_data, candidate_features, ranker_family, batch_rankings, monitoring, charts],
    asset_checks=[check_below_ceiling, check_availability_contract],
    jobs=[full_refresh_job],
    schedules=[nightly_schedule],
)

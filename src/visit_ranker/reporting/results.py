"""The pipeline `make demo` runs, and the table it prints."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..config import GROUND_TRUTH, OUT_DIR, Config, load_config
from ..features.build import build_candidates
from ..models import availability, calibration, evaluate, explain, ranker


def _segment_comparison(test: pd.DataFrame, scored: pd.DataFrame, baselines) -> pd.DataFrame:
    """Model against the strongest simple rule, split by visitor type."""
    model_trips = evaluate.add_segments(evaluate.trip_level(scored))

    habit = test.copy()
    habit["score"] = baselines.score(habit, "employee_last_choice")
    habit_trips = evaluate.add_segments(evaluate.trip_level(habit))

    rows = []
    for segment in ("first visit", "repeat visit"):
        model_slice = model_trips[model_trips["visitor_type"] == segment]
        habit_slice = habit_trips[habit_trips["visitor_type"] == segment]
        if model_slice.empty:
            continue
        rows.append(
            {
                "segment": segment,
                "trips": int(len(model_slice)),
                "model_ndcg_at_1": round(float(model_slice["ndcg_at_1"].mean()), 4),
                "habit_rule_ndcg_at_1": round(float(habit_slice["ndcg_at_1"].mean()), 4),
                "model_ndcg_at_3": round(float(model_slice["ndcg_at_3"].mean()), 4),
                "habit_rule_ndcg_at_3": round(float(habit_slice["ndcg_at_3"].mean()), 4),
            }
        )
    table = pd.DataFrame(rows)
    table["lift_ndcg_at_1"] = (
        table["model_ndcg_at_1"] / table["habit_rule_ndcg_at_1"].replace(0, np.nan) - 1
    ).round(4)
    return table


def _coldest_model_key(family) -> str | None:
    """The trained model with the fewest features that still has gravity terms.

    Used for the structure-recovery check: it is the model that cannot lean on
    habit, so its attribution reflects the gravity terms directly.
    """
    candidates = [key for key in family.models if "personal" not in key and key != "none"]
    if not candidates:
        return None
    return max(candidates, key=lambda key: len(family.features[key]))


def run_pipeline(cfg: Config | None = None, tune: bool = False) -> dict:
    cfg = cfg or load_config()
    started = time.time()

    candidates = build_candidates(cfg)
    candidates["mask_key"] = availability.mask_keys(candidates)

    train = candidates[candidates["split"] == "train"]
    validation = candidates[candidates["split"] == "validation"]
    test = candidates[candidates["split"] == "test"]

    params = None
    tuning = None
    if tune:
        from ..models.tuning import tune_params

        tuning = tune_params(train, validation)
        params = tuning["best_params"]

    family = ranker.train_family(train, validation, params=params)
    baselines = ranker.train_baselines(train)

    scored_test = family.score(test)
    scored_validation = family.score(validation)

    results: dict[str, pd.DataFrame] = {}
    ranker_metrics = evaluate.evaluate_scored(scored_test)
    results["lambdamart_family"] = ranker_metrics["overall"]

    for name in [*ranker.BASELINES, "logistic_regression"]:
        working = test.copy()
        working["score"] = baselines.score(working, name)
        results[name] = evaluate.evaluate_scored(working)["overall"]

    lift = evaluate.lift_table(results, reference="most_popular_in_city")

    # Where the model actually earns its keep.
    #
    # Overall it beats "send them where they went last time" by about a point
    # of NDCG@1, which on its own is a thin argument for a gradient-boosted
    # ranker. The overall number hides the shape: the habit rule is excellent
    # on repeat visitors and has literally nothing to say about a first-time
    # visitor, and those are precisely the trips where getting it wrong is
    # most visible to the traveller.
    segment_comparison = _segment_comparison(test, scored_test, baselines)

    # --- calibration and abstention ---------------------------------------
    validation_top = calibration.top_one_frame(scored_validation)
    calibrator = calibration.fit_calibrator(validation_top)

    test_top = calibration.top_one_frame(scored_test)
    test_top["probability"] = calibrator.probability(test_top["margin"].to_numpy())
    test_top = calibration.add_top2_flag(test_top, scored_test)

    reliability = calibration.reliability(test_top)
    curve = calibration.abstention_curve(test_top)
    threshold = calibration.recommend_threshold(curve)

    # --- explanation and structure recovery -------------------------------
    truth = json.loads(GROUND_TRUTH.read_text())
    richest = max(family.models, key=lambda key: len(family.features[key]))
    richest_rows = scored_test[scored_test["scored_by"] == richest]
    if richest_rows.empty:
        richest_rows = test[test["mask_key"] == richest]

    importance = explain.global_importance(
        family.models[richest], richest_rows, family.features[richest]
    )
    comparison = explain.compare_with_truth(importance, truth["choice_weights"])
    correlation = explain.rank_correlation(comparison)

    # The same comparison on the cold-start model, which has no habit feature.
    #
    # Habit absorbs the gravity terms: a traveller went to the leader's office
    # last time BECAUSE of the leader, so once habit is in the model the
    # leader feature is nearly redundant and SHAP attributes almost nothing to
    # it. The importance of a feature in a model is not its weight in the
    # world, and with a simulator it is possible to show that rather than
    # assert it. The cold-start model has no such shortcut available, so its
    # attribution is much closer to the planted structure.
    cold_key = _coldest_model_key(family)
    cold_comparison = None
    cold_correlation = None
    if cold_key is not None:
        cold_rows = scored_test[scored_test["scored_by"] == cold_key]
        if len(cold_rows) < 200:
            cold_rows = test[
                test["mask_key"].map(
                    lambda k: availability.mask_from_key(cold_key).covers(
                        availability.mask_from_key(str(k))
                    )
                )
            ]
        if not cold_rows.empty:
            cold_importance = explain.global_importance(
                family.models[cold_key], cold_rows, family.features[cold_key]
            )
            cold_comparison = explain.compare_with_truth(cold_importance, truth["choice_weights"])
            cold_correlation = explain.rank_correlation(cold_comparison)

    worked_examples = []
    for trip_id in richest_rows["trip_id"].drop_duplicates().head(2):
        trip_rows = richest_rows[richest_rows["trip_id"] == trip_id]
        worked_examples.append(
            (
                trip_id,
                explain.explain_trip(family.models[richest], trip_rows, family.features[richest]),
            )
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(OUT_DIR / "candidates.parquet", index=False)
    scored_test.to_parquet(OUT_DIR / "scored_test.parquet", index=False)
    lift.to_csv(OUT_DIR / "lift_table.csv", index=False)
    comparison.to_csv(OUT_DIR / "importance_vs_truth.csv", index=False)
    curve.to_csv(OUT_DIR / "abstention_curve.csv", index=False)

    return {
        "cfg": cfg,
        "cold_key": cold_key,
        "cold_comparison": cold_comparison,
        "cold_rank_correlation": cold_correlation,
        "elapsed": time.time() - started,
        "truth": truth,
        "candidates": candidates,
        "availability": availability.availability_report(candidates),
        "family": family,
        "scored_test": scored_test,
        "metrics": ranker_metrics,
        "lift": lift,
        "segment_comparison": segment_comparison,
        "reliability": reliability,
        "ece": calibration.expected_calibration_error(reliability),
        "abstention_curve": curve,
        "threshold": threshold,
        "importance": importance,
        "comparison": comparison,
        "rank_correlation": correlation,
        "worked_examples": worked_examples,
        "tuning": tuning,
        "test_top": test_top,
    }


def render(result: dict) -> str:
    cfg: Config = result["cfg"]
    truth = result["truth"]
    rule = "=" * 100
    out = [
        rule,
        f"office-visit-ranker  |  profile: {cfg.profile_name}  |  seed: {cfg.seed}",
        f"window: {cfg.start_date} to {cfg.end_date}  |  train<{cfg.train_end} "
        f"validate<{cfg.validation_end} test after",
        rule,
    ]

    realised = truth["realised"]
    out.append("\nWHAT THE SIMULATOR PLANTED")
    out.append(
        f"  {realised['trips']:,} trips by {realised['travellers']:,} travellers, "
        f"{realised['mean_candidates']} candidate offices per trip"
    )
    out.append(
        f"  exploration share {realised['exploration_share_realised']:.3f} "
        f"-> ACCURACY CEILING {realised['accuracy_ceiling']:.4f}"
    )
    out.append(f"  choice weights: {truth['choice_weights']}")

    out.append("\nFEATURE AVAILABILITY, and which model scores each mask")
    report = result["availability"]
    out.append(
        f"  {'mask':34s} {'trips':>6s} {'share':>7s} {'feats':>6s} "
        f"{'scored by':34s} {'fallback':>8s}"
    )
    for row in report.itertuples(index=False):
        out.append(
            f"  {row.mask:34s} {row.trips:>6d} {row.share:>7.3f} {row.n_features:>6d} "
            f"{row.scored_by:34s} {row.fallback_steps:>8d}"
        )

    out.append("\nRANKING QUALITY ON THE HELD-OUT TEST WINDOW")
    lift = result["lift"]
    out.append(
        f"  {'model':26s} {'NDCG@1':>8s} {'NDCG@3':>8s} {'MRR':>7s} {'R@2':>7s} "
        f"{'lift NDCG@3':>12s}"
    )
    for row in lift.itertuples(index=False):
        lift_value = getattr(row, "lift_ndcg_at_3", np.nan)
        out.append(
            f"  {row.model:26s} {row.ndcg_at_1:>8.4f} {row.ndcg_at_3:>8.4f} "
            f"{row.reciprocal_rank:>7.4f} {row.recall_at_2:>7.4f} {lift_value:>+12.1%}"
        )
    headline = float(lift[lift["model"] == "lambdamart_family"]["hit_at_1"].iloc[0])
    ceiling = realised["accuracy_ceiling"]
    out.append(
        f"  Hit@1 {headline:.4f} against a ceiling of {ceiling:.4f} "
        f"({headline / ceiling:.1%} of what is attainable)"
    )

    out.append("\nWHERE THE MODEL BEATS THE BEST SIMPLE RULE")
    out.append(
        f"  {'segment':16s} {'trips':>6s} {'model N@1':>10s} {'habit rule':>11s} {'lift':>9s}"
    )
    for row in result["segment_comparison"].itertuples(index=False):
        out.append(
            f"  {row.segment:16s} {row.trips:>6d} {row.model_ndcg_at_1:>10.4f} "
            f"{row.habit_rule_ndcg_at_1:>11.4f} {row.lift_ndcg_at_1:>+9.1%}"
        )

    out.append("\nBY SEGMENT")
    for name, key in (("visitor type", "by_visitor_type"), ("city size", "by_city_size")):
        out.append(f"  by {name}")
        frame = result["metrics"][key]
        for row in frame.itertuples(index=False):
            label = str(row[0])
            out.append(
                f"    {label:16s} NDCG@1 {row.ndcg_at_1:.4f}  NDCG@3 {row.ndcg_at_3:.4f}  "
                f"R@2 {row.recall_at_2:.4f}  ({row.trips} trips)"
            )

    out.append("\nBY AVAILABILITY MASK (accuracy where features are missing)")
    by_mask = result["metrics"]["by_mask"]
    for row in by_mask.sort_values("trips", ascending=False).itertuples(index=False):
        out.append(
            f"  {str(row.mask_key):34s} NDCG@1 {row.ndcg_at_1:.4f}  "
            f"NDCG@3 {row.ndcg_at_3:.4f}  ({row.trips} trips)"
        )

    out.append("\nCALIBRATION AND ABSTENTION")
    out.append(f"  expected calibration error {result['ece']:.4f}")
    threshold = result["threshold"]
    out.append(
        f"  threshold {threshold['threshold']:.2f}: answers {threshold['coverage']:.1%} of trips "
        f"with one office, right {threshold['accuracy_covered']:.1%} of the time"
    )
    out.append(
        f"  abstaining on the rest and sending the top two: "
        f"{threshold['accuracy_with_fallback']:.1%} overall "
        f"(against {threshold['accuracy_without_abstention']:.1%} with no abstention)"
    )

    out.append("\nDID THE MODEL RECOVER THE SIMULATOR'S STRUCTURE?")
    out.append(
        f"  {'utility term':18s} {'planted share':>14s} {'recovered share':>16s} "
        f"{'planted rank':>13s} {'recovered rank':>15s}"
    )
    for row in result["comparison"].itertuples(index=False):
        out.append(
            f"  {row.utility_term:18s} {row.planted_share:>14.4f} {row.recovered_share:>16.4f} "
            f"{row.planted_rank:>13d} {row.recovered_rank:>15d}"
        )
    out.append(f"  Spearman rank correlation: {result['rank_correlation']}")
    if result.get("cold_comparison") is not None:
        out.append(
            f"\n  Same comparison on the cold-start model ({result['cold_key']}), which has "
            "no habit feature to lean on:"
        )
        out.append(
            f"  {'utility term':18s} {'planted share':>14s} {'recovered share':>16s} "
            f"{'planted rank':>13s} {'recovered rank':>15s}"
        )
        for row in result["cold_comparison"].itertuples(index=False):
            out.append(
                f"  {row.utility_term:18s} {row.planted_share:>14.4f} "
                f"{row.recovered_share:>16.4f} {row.planted_rank:>13d} "
                f"{row.recovered_rank:>15d}"
            )
        out.append(f"  Spearman rank correlation: {result['cold_rank_correlation']}")

    out.append("\nWORKED EXAMPLES (why this office)")
    for trip_id, frame in result["worked_examples"]:
        out.append(f"  trip {trip_id}")
        for row in frame.head(3).itertuples(index=False):
            marker = "<- actually visited" if row.is_correct else ""
            out.append(f"    {row.office_id}  score {row.score:+.3f}  {row.top_drivers}  {marker}")

    out.append(f"\npipeline completed in {result['elapsed']:.0f}s")
    out.append(rule)
    return "\n".join(out)


def main(charts: bool = True, tune: bool = False) -> int:
    cfg = load_config()
    if not GROUND_TRUTH.exists():
        print("No generated data. Run `make data` first.")
        return 1
    result = run_pipeline(cfg, tune=tune)
    if charts:
        from .charts import render_all

        for path in render_all(result):
            print(f"  wrote {path.relative_to(path.parents[2])}")
    print(render(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(main(tune="--tune" in sys.argv))

"""One end-to-end test asserting the claims the README makes."""

from __future__ import annotations

from visit_ranker.models import evaluate


def test_pipeline_ranks_well_but_below_the_ceiling(scored, truth, candidates):
    ceiling = float(truth["realised"]["accuracy_ceiling"])
    per_trip = evaluate.add_segments(evaluate.trip_level(scored))
    hit_at_1 = float(per_trip["hit_at_1"].mean())

    # 1. Good, but never above what the exploration share makes attainable.
    #    Above the ceiling means a leak, not a better model.
    assert 0.60 < hit_at_1 <= ceiling, f"{hit_at_1} against a ceiling of {ceiling}"

    # 2. Cold start is genuinely harder, and the repo reports it rather than
    #    hiding it in an average.
    by_type = per_trip.groupby("visitor_type")["ndcg_at_1"].mean()
    assert by_type["first visit"] < by_type["repeat visit"]

    # 3. More candidate offices is a harder problem.
    by_size = per_trip.groupby("city_size", observed=True)["ndcg_at_1"].mean()
    assert by_size.iloc[0] > by_size.iloc[-1]

    # 4. Every trip got exactly one prediction, from a model it was allowed
    #    to be scored by.
    assert scored.groupby("trip_id")["label"].sum().eq(1).all()
    assert scored["scored_by"].ne("").all()


def test_ranker_beats_every_simple_baseline(scored, candidates):
    from visit_ranker.models import ranker

    train = candidates[candidates["split"] == "train"]
    test = candidates[candidates["split"] == "test"]
    baselines = ranker.train_baselines(train)

    model_score = float(evaluate.trip_level(scored)["ndcg_at_3"].mean())
    for name in ranker.BASELINES:
        working = test.copy()
        working["score"] = baselines.score(working, name)
        baseline_score = float(evaluate.trip_level(working)["ndcg_at_3"].mean())
        assert model_score >= baseline_score - 1e-6, f"beaten by {name}"


def test_the_model_helps_most_where_habit_is_absent(scored, candidates):
    """The honest headline: overall the model barely beats 'send them where
    they went last time', and the whole gain is in the cold-start segment."""
    from visit_ranker.models import ranker

    train = candidates[candidates["split"] == "train"]
    test = candidates[candidates["split"] == "test"]
    baselines = ranker.train_baselines(train)

    habit = test.copy()
    habit["score"] = baselines.score(habit, "employee_last_choice")

    model_trips = evaluate.add_segments(evaluate.trip_level(scored))
    habit_trips = evaluate.add_segments(evaluate.trip_level(habit))

    first_model = model_trips[model_trips["visitor_type"] == "first visit"]["ndcg_at_1"].mean()
    first_habit = habit_trips[habit_trips["visitor_type"] == "first visit"]["ndcg_at_1"].mean()
    assert first_model > first_habit * 1.10

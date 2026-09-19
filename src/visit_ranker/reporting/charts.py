"""Static charts for the README, written to docs/img/.

Rendered by a script rather than screenshotted, so the README renders without
running anything and every chart is reproducible from the seed.

Colour follows a validated categorical palette in fixed slot order, a single
hue for magnitude, and direct value labels throughout.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import IMG_DIR  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQUENTIAL = ["#cfe0f5", "#9ec2ea", "#6ba3e0", "#2a78d6", "#1a4d8a"]
INK, INK_MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#d8d8d4"

MODEL_LABELS = {
    "lambdamart_family": "LambdaMART family",
    "employee_last_choice": "Their last choice",
    "team_most_frequent": "Team's usual office",
    "logistic_regression": "Logistic regression",
    "leader_most_frequent": "Leader's usual office",
    "most_popular_in_city": "Biggest office in city",
    "nearest_to_centre": "Nearest to centre",
}


def _style(ax, title: str, subtitle: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=12.5, color=INK, pad=28 if subtitle else 10, loc="left")
    if subtitle:
        ax.text(
            0, 1.035, subtitle, transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom"
        )


def _save(fig, name: str):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / name
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_lift(lift: pd.DataFrame, ceiling: float):
    """Every model's NDCG@1 and NDCG@3, against the attainable ceiling."""
    frame = lift.copy()
    frame["label"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    frame = frame.sort_values("ndcg_at_3")

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(frame))
    ax.barh(
        y - 0.2,
        frame["ndcg_at_1"],
        height=0.36,
        color=SERIES[0],
        label="NDCG@1 (= Hit@1)",
        edgecolor=SURFACE,
        linewidth=1.4,
    )
    ax.barh(
        y + 0.2,
        frame["ndcg_at_3"],
        height=0.36,
        color=SERIES[2],
        label="NDCG@3",
        edgecolor=SURFACE,
        linewidth=1.4,
    )
    for i, row in enumerate(frame.itertuples(index=False)):
        ax.text(
            row.ndcg_at_1 + 0.008,
            i - 0.2,
            f"{row.ndcg_at_1:.3f}",
            va="center",
            fontsize=8.5,
            color=INK_MUTED,
        )
        ax.text(
            row.ndcg_at_3 + 0.008,
            i + 0.2,
            f"{row.ndcg_at_3:.3f}",
            va="center",
            fontsize=8.5,
            color=INK_MUTED,
        )

    ax.axvline(ceiling, color=SERIES[1], linewidth=2.2, linestyle="--")
    ax.text(ceiling, len(frame) - 0.4, f"  ceiling {ceiling:.3f}", fontsize=9, color=INK, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["label"])
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", color=GRID, alpha=0.6, linewidth=0.8)
    _style(
        ax,
        "Ranking quality against every baseline",
        "The ceiling is what the simulator's exploration share makes attainable. Nothing "
        "should sit above it.",
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_MUTED)
    return _save(fig, "lift_table.png")


def chart_segment_lift(segment: pd.DataFrame):
    """Where the model beats the best simple rule."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(segment))
    width = 0.36
    ax.bar(
        x - width / 2,
        segment["habit_rule_ndcg_at_1"],
        width,
        label="Their last choice",
        color=SERIES[3],
        edgecolor=SURFACE,
        linewidth=1.5,
    )
    ax.bar(
        x + width / 2,
        segment["model_ndcg_at_1"],
        width,
        label="LambdaMART family",
        color=SERIES[0],
        edgecolor=SURFACE,
        linewidth=1.5,
    )
    for i, row in enumerate(segment.itertuples(index=False)):
        ax.text(
            i - width / 2,
            row.habit_rule_ndcg_at_1 + 0.015,
            f"{row.habit_rule_ndcg_at_1:.3f}",
            ha="center",
            fontsize=9,
            color=INK_MUTED,
        )
        ax.text(
            i + width / 2,
            row.model_ndcg_at_1 + 0.015,
            f"{row.model_ndcg_at_1:.3f}",
            ha="center",
            fontsize=9,
            color=INK_MUTED,
        )
        ax.text(
            i,
            max(row.model_ndcg_at_1, row.habit_rule_ndcg_at_1) + 0.07,
            f"{row.lift_ndcg_at_1:+.0%}",
            ha="center",
            fontsize=11,
            color=INK,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.segment}\n({r.trips} trips)" for r in segment.itertuples(index=False)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("NDCG@1", fontsize=9.5, color=INK_MUTED)
    ax.grid(axis="y", color=GRID, alpha=0.6, linewidth=0.8)
    _style(
        ax,
        "The model earns its keep on first-time visitors",
        "On repeat visitors a one-line rule is as good. The gradient-boosted ranker is "
        "paying for the cold-start case.",
    )
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_MUTED)
    return _save(fig, "segment_lift.png")


def chart_availability(metrics: dict, availability_report: pd.DataFrame):
    """Accuracy per availability tier, sized by how many trips are in it."""
    by_mask = (
        metrics["by_mask"]
        .merge(
            availability_report[["mask", "share", "fallback_steps"]],
            left_on="mask_key",
            right_on="mask",
            how="left",
        )
        .sort_values("trips", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    colours = [
        SERIES[1] if steps and steps > 0 else SERIES[0]
        for steps in by_mask["fallback_steps"].fillna(0)
    ]
    bars = ax.bar(
        range(len(by_mask)), by_mask["ndcg_at_1"], color=colours, edgecolor=SURFACE, linewidth=1.5
    )
    for bar, row in zip(bars, by_mask.itertuples(index=False), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{row.ndcg_at_1:.3f}\n({row.trips} trips)",
            ha="center",
            fontsize=8.5,
            color=INK_MUTED,
        )

    ax.set_xticks(range(len(by_mask)))
    ax.set_xticklabels([str(m).replace("+", "\n+") for m in by_mask["mask_key"]], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("NDCG@1", fontsize=9.5, color=INK_MUTED)
    ax.grid(axis="y", color=GRID, alpha=0.6, linewidth=0.8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SERIES[0]),
        plt.Rectangle((0, 0), 1, 1, color=SERIES[1]),
    ]
    ax.legend(
        handles,
        ["has its own model", "falls back"],
        frameon=False,
        fontsize=9,
        labelcolor=INK_MUTED,
    )
    _style(
        ax,
        "Accuracy by availability tier",
        "Each tier is scored by a model trained on exactly the features that tier has. "
        "Nothing is imputed.",
    )
    return _save(fig, "availability_tiers.png")


def chart_reliability(reliability: pd.DataFrame, ece: float):
    fig, ax = plt.subplots(figsize=(6.6, 6))
    ax.plot(
        [0, 1], [0, 1], color=INK_MUTED, linestyle="--", linewidth=1.3, label="perfect calibration"
    )
    sizes = 40 + 260 * reliability["trips"] / reliability["trips"].max()
    ax.scatter(
        reliability["predicted"],
        reliability["observed"],
        s=sizes,
        color=SERIES[0],
        alpha=0.8,
        edgecolor=SURFACE,
        linewidth=1.4,
        label="observed (sized by trips)",
    )
    ax.plot(reliability["predicted"], reliability["observed"], color=SERIES[0], linewidth=1.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability the top office is right", fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel("observed accuracy", fontsize=9.5, color=INK_MUTED)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.8)
    _style(ax, "Top-1 calibration", f"Isotonic, fitted on validation. ECE {ece:.4f}.")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_MUTED)
    return _save(fig, "reliability.png")


def chart_abstention(curve: pd.DataFrame, chosen: dict):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(
        curve["coverage"],
        curve["accuracy_covered"],
        marker="o",
        markersize=6,
        linewidth=2,
        color=SERIES[0],
        label="accuracy on the trips answered",
    )
    ax.plot(
        curve["coverage"],
        curve["accuracy_with_fallback"],
        marker="s",
        markersize=6,
        linewidth=2,
        color=SERIES[2],
        label="accuracy counting a top-2 fallback",
    )
    ax.scatter(
        [chosen["coverage"]],
        [chosen["accuracy_covered"]],
        s=180,
        color=SERIES[1],
        zorder=4,
        edgecolor=SURFACE,
        linewidth=2,
    )
    ax.annotate(
        f"chosen: threshold {chosen['threshold']:.2f}\n"
        f"{chosen['coverage']:.0%} answered, {chosen['accuracy_covered']:.0%} right",
        (chosen["coverage"], chosen["accuracy_covered"]),
        textcoords="offset points",
        xytext=(-16, -46),
        fontsize=9,
        color=INK,
    )
    ax.set_xlabel(
        "coverage - share of trips answered with a single office", fontsize=9.5, color=INK_MUTED
    )
    ax.set_ylabel("accuracy", fontsize=9.5, color=INK_MUTED)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.8)
    _style(
        ax,
        "What abstention buys",
        "Below the threshold the service sends the top two offices instead of one. The cost "
        "is a vaguer message, not a wrong door.",
    )
    ax.legend(frameon=False, fontsize=9.5, loc="lower left", labelcolor=INK_MUTED)
    return _save(fig, "abstention_curve.png")


def chart_importance_vs_truth(
    comparison: pd.DataFrame, correlation: float, label: str, filename: str
):
    """Recovered SHAP importance against the simulator's own weights."""
    frame = comparison.sort_values("planted_share", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    y = np.arange(len(frame))
    ax.barh(
        y - 0.2,
        frame["planted_share"],
        height=0.36,
        color=SERIES[2],
        label="planted weight (share)",
        edgecolor=SURFACE,
        linewidth=1.4,
    )
    ax.barh(
        y + 0.2,
        frame["recovered_share"],
        height=0.36,
        color=SERIES[0],
        label="recovered SHAP (share)",
        edgecolor=SURFACE,
        linewidth=1.4,
    )
    for i, row in enumerate(frame.itertuples(index=False)):
        ax.text(
            row.planted_share + 0.004,
            i - 0.2,
            f"{row.planted_share:.3f}",
            va="center",
            fontsize=8.5,
            color=INK_MUTED,
        )
        ax.text(
            row.recovered_share + 0.004,
            i + 0.2,
            f"{row.recovered_share:.3f}",
            va="center",
            fontsize=8.5,
            color=INK_MUTED,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(frame["utility_term"])
    ax.grid(axis="x", color=GRID, alpha=0.6, linewidth=0.8)
    _style(
        ax,
        f"Did the model find the simulator's structure? ({label})",
        f"Spearman rank correlation {correlation}. Two different quantities normalised to "
        "shares - the ordering is the comparison, not the numbers.",
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_MUTED)
    return _save(fig, filename)


def chart_shap_summary(importance: pd.DataFrame):
    frame = importance.head(14).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    colours = [
        SEQUENTIAL[min(int(i * len(SEQUENTIAL) / len(frame)), len(SEQUENTIAL) - 1)]
        for i in range(len(frame))
    ]
    bars = ax.barh(
        frame["feature"], frame["mean_abs_shap"], color=colours, edgecolor=SURFACE, linewidth=1.4
    )
    for bar, value in zip(bars, frame["mean_abs_shap"], strict=True):
        ax.text(
            bar.get_width() * 1.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8.5,
            color=INK_MUTED,
        )
    ax.set_xlabel("mean |SHAP value|", fontsize=9.5, color=INK_MUTED)
    ax.grid(axis="x", color=GRID, alpha=0.6, linewidth=0.8)
    _style(ax, "Global feature importance", "TreeSHAP on the richest availability tier.")
    return _save(fig, "shap_summary.png")


def render_all(result: dict) -> list:
    ceiling = result["truth"]["realised"]["accuracy_ceiling"]
    paths = [
        chart_lift(result["lift"], ceiling),
        chart_segment_lift(result["segment_comparison"]),
        chart_availability(result["metrics"], result["availability"]),
        chart_reliability(result["reliability"], result["ece"]),
        chart_abstention(result["abstention_curve"], result["threshold"]),
        chart_shap_summary(result["importance"]),
        chart_importance_vs_truth(
            result["comparison"],
            result["rank_correlation"],
            "richest tier, habit available",
            "importance_vs_truth.png",
        ),
    ]
    if result.get("cold_comparison") is not None:
        paths.append(
            chart_importance_vs_truth(
                result["cold_comparison"],
                result["cold_rank_correlation"],
                "cold-start tier, no habit",
                "importance_vs_truth_coldstart.png",
            )
        )
    return paths

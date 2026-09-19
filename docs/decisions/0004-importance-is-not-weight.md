# 4. Feature importance is not the weight in the world

**Status:** accepted · **Date:** 2025-06-30

## Context

Because the simulator's choice weights are known, the model's recovered SHAP
importance can be compared against them. That is a stronger test than any
accuracy number: a model can rank well while leaning on one dominant feature
and getting the rest of the structure wrong.

The comparison came back **weak**: Spearman 0.24 on the richest availability
tier, with leader gravity ranked 8th of 8 against a planted rank of 2nd.

## What was measured, not assumed

Two mechanisms, both checkable:

**Habit absorbs the gravity terms.** A traveller went to the leader's office
last time *because* of the leader. Once habit is in the model, leader gravity
is nearly redundant and SHAP attributes almost nothing to it. The correlation
between `leader_is_top1` and `habit_last_visit` is 0.26.

**Leader and team gravity are collinear.** The manager is a member of the
team, so their office is the team's office 42% of the time; the two features
correlate at 0.32. A tree attributes to whichever it splits on first.

## Decision

Report the comparison on **two** models: the richest tier, and the cold-start
tier that has no habit feature to lean on. Publish both correlations and both
charts, and state the mechanism with the measured numbers rather than waving
at "multicollinearity".

## What this does not fix

The cold-start comparison is not better in rank correlation — it over-weights
team gravity instead, for the collinearity reason above. So the honest
conclusion is not "here is how to recover the true structure" but:

> Attribution methods answer "what did this model use", which is a different
> question from "what drives the world". When features encode each other's
> downstream effects, no amount of attribution separates them, and the only
> reliable way to recover a causal weight is to vary the feature independently
> — which is an experiment, not an explanation.

A repository that could not check this would have reported the SHAP ranking as
though it were the answer.

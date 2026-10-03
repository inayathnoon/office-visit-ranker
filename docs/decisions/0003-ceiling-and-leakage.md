# 3. The simulator publishes an accuracy ceiling, and it is a blocking check

**Status:** accepted · **Date:** 2025-06-30

## Context

The hardest thing to detect in a ranking pipeline is leakage. A feature that
quietly includes the trip's own outcome produces a model that is excellent in
validation, excellent in test, and useless in production - and every metric
looks better, which is exactly the wrong signal.

## Decision

The simulator decides **12% of choices by exploration**, ignoring utility
entirely. No feature predicts those, by construction. That implies a knowable
ceiling - 0.910 on the demo profile, accounting for an exploring traveller
still being right by luck at 1/n - which is written to `ground_truth.json`.

A Dagster asset check compares the achieved Hit@1 against it and **blocks** if
the model is above. Above the ceiling means a leak, not a better model.

This is the check that other repositories cannot have. Against real data there
is no ceiling to compare with, and a suspiciously good number has to be chased
by hand.

## Supporting defences

* **Features are built by walking trips in date order**, so a trip can only
  ever see history accumulated before it. The guarantee is structural - the
  history object simply does not contain the future - rather than a filter
  somebody has to remember.
* **`tests/test_leakage.py` re-derives each feature by brute force** against
  an independent as-of computation on a sample of trips.
* **No feature may correlate above 0.95 with the label.**
* **The split is by time**, with the boundaries asserted. Two trips by the
  same traveller weeks apart share a habit that is the strongest feature in
  the model; a random split puts one on each side and the score comes back
  flattering.

## Consequences

The headline result reads as 0.740 against a ceiling of 0.910 - 81% of what is
attainable - rather than as a bare 0.740. The second number is what makes the
first interpretable.

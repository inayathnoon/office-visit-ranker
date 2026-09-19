# 2. Missing features are routed around, not imputed

**Status:** accepted · **Date:** 2025-06-30

## Context

Some features cannot exist for some trips:

* a **first-time visitor** to a city has no habit and no personal history
  there — 12% of trips on the demo profile;
* a **contractor or BPO worker** often has no manager in the HR extract, so
  there is no leader gravity — leader is available on only 19% of trips.

The standard move is to impute zero and add an indicator column.

## Why that is wrong here

```
imputed zero  =>  "the manager uses none of these offices"
absent        =>  "we do not know where the manager sits"
```

These are different claims. The first is not supported by the data, and a tree
model will happily split on it. Worse, at scoring time the same zero then
covers two populations that behave completely differently: people whose
manager genuinely prefers other buildings, and people who have no manager at
all. An indicator column lets a model *in principle* separate them; it does
not make it do so, and nothing checks whether it did.

## Decision

A trip carries an **availability mask** over four optional families — personal
history, leader, team, department. A model is trained per mask on exactly the
features that mask has. Masks with fewer than 60 training trips fall back
through a declared order (leader, then department, then team, then personal —
least informative first), terminating at the empty mask, which always has a
model.

The invariant is enforced **at the moment of scoring**, not just at setup:
`RankerFamily.score` raises if a model would read a column the row's mask does
not have. A test deliberately mislabels rich rows as empty and asserts the
raise.

On the demo profile that is six models covering 98% of trips directly; two
rare masks fall back one step.

## Where the line is drawn

`prior_visit_count` is always available and lives in trip context, while
`emp_visit_share` is masked out. A count of zero is a fact — "they have never
been here". A *share* over an empty history is undefined. That distinction is
the whole basis of the contract, and a test caught the original name
(`emp_prior_visits_to_city`) implying the wrong one.

## Consequences

Accuracy is reported per mask, because it differs sharply and an average over
tiers describes none of them. The service returns the tier it used, so a
consumer can see that a first-visit prediction rests on twelve features while
a regular's rests on twenty-seven.

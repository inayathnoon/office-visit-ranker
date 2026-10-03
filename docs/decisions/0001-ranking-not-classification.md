# 1. This is a ranking problem, and the group is the trip

**Status:** accepted · **Date:** 2025-06-30

## Context

The obvious framing is classification: for each (trip, office) pair, will they
walk into this one? It is simpler, every classifier in the world accepts it,
and it is the wrong shape.

## Why

The service sends **one instruction per trip** - grant access here, book a desk
here, tell catering this reception - with the option of naming two buildings
when it is unsure. What matters is the *order* of the offices within a trip,
not the absolute probability of each.

A binary classifier optimises per-row log loss. It is indifferent to whether
the correct office came first or third within a trip, as long as its
probabilities are well calibrated overall. Two models with identical log loss
can differ enormously in how often the right building is at the top.

`lambdarank` optimises NDCG directly over groups, where the group is the trip.
That is the loss the decision actually cares about.

## Decision

One row per (trip, candidate office in the destination city). Group = trip.
Label = 1 for the office visited. LightGBM `lambdarank`.

The logistic-regression baseline is kept precisely so the framing can be
tested rather than asserted: it sees identical features and optimises the
other loss. On the demo profile it reaches NDCG@3 of 0.697 against the
ranker's 0.830.

## A note on the metrics

With exactly one relevant item per group several standard metrics collapse,
and `models/evaluate.py` says so rather than quoting them as independent
evidence:

* NDCG@1 **is** Hit@1 - the discount at rank 1 is 1 and the ideal DCG is 1.
* MRR is the mean of 1/rank of the correct office.
* NDCG@3 is the one that adds information, because it separates "second" from
  "third" when the top-1 is wrong.

## Consequences

Ties have to be broken deterministically, or a baseline that scores every
candidate identically gets whatever order the frame happened to arrive in.
`_ranks` sorts on office id as a tiebreak, and a test checks that reversing
the input frame does not change the result.

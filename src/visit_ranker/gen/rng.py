"""Seeded random streams, one named substream per generator component."""

from __future__ import annotations

import hashlib

import numpy as np


def substream(seed: int, name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def lognormal_from_mean(
    rng: np.random.Generator, mean: float, sigma: float, size: int | tuple[int, ...]
) -> np.ndarray:
    """Lognormal draws whose *arithmetic* mean is ``mean``."""
    return rng.lognormal(np.log(mean) - 0.5 * sigma**2, sigma, size)


def choice_from_mix(rng: np.random.Generator, mix: dict[str, float], size: int) -> np.ndarray:
    labels = list(mix)
    weights = np.array([mix[k] for k in labels], dtype=float)
    return rng.choice(labels, size=size, p=weights / weights.sum())


def gumbel(rng: np.random.Generator, scale: float, size) -> np.ndarray:
    """Gumbel noise - the error term that makes a logit a logit."""
    return rng.gumbel(0.0, scale, size)

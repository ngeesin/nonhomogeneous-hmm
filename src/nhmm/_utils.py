"""Internal numerical and validation helpers."""

from __future__ import annotations

import numbers

import numpy as np
from scipy.special import logsumexp


def check_random_state(seed) -> np.random.Generator:
    """Turn ``seed`` into a :class:`numpy.random.Generator` instance.

    Accepts ``None`` (fresh generator), an integer seed, an existing
    ``Generator`` or a legacy ``RandomState``.
    """
    if seed is None or isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.RandomState):
        # Bridge legacy RandomState into a Generator-compatible interface.
        return np.random.default_rng(seed.randint(0, 2**32 - 1))
    raise ValueError(f"Cannot use {seed!r} to seed a numpy Generator")


def log_normalize(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Normalise so that ``exp`` of the result sums to one along ``axis``."""
    a = np.asarray(a, dtype=float)
    with np.errstate(under="ignore"):
        return a - logsumexp(a, axis=axis, keepdims=True)


def log_softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable ``log(softmax(scores))`` along ``axis``."""
    scores = np.asarray(scores, dtype=float)
    return scores - logsumexp(scores, axis=axis, keepdims=True)


def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores, axis=axis, keepdims=True)
    np.exp(shifted, out=shifted)
    shifted /= shifted.sum(axis=axis, keepdims=True)
    return shifted


def normalize(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Normalise ``a`` to sum to one along ``axis`` (zero-sum rows stay zero)."""
    a = np.asarray(a, dtype=float)
    total = a.sum(axis=axis, keepdims=True)
    total[total == 0.0] = 1.0
    return a / total


def iter_sequences(n_total: int, lengths=None):
    """Yield ``(start, end)`` slices for each sequence.

    When ``lengths`` is ``None`` the whole array is treated as one sequence.
    """
    if lengths is None:
        yield 0, n_total
        return
    lengths = np.asarray(lengths, dtype=int)
    if lengths.sum() != n_total:
        raise ValueError(
            f"lengths sum to {int(lengths.sum())} but data has {n_total} rows"
        )
    end = 0
    for length in lengths:
        if length <= 0:
            raise ValueError("sequence lengths must be positive")
        start, end = end, end + int(length)
        yield start, end

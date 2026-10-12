"""
numcompute.rank
===============

Ranking with tie handling, and percentiles.

Conventions
-----------
- Ranks are 1-based (smallest value gets rank 1), as in most statistics texts.
- ``axis=-1`` default for ranking; ``axis=None`` (flatten) default for percentiles.
- NaN handling is controlled by ``nan_policy`` (see each function).
"""

from __future__ import annotations

import numpy as np

from .sort_search import argsort  # stable argsort is the backbone of ranking

__all__ = [
    "rank",
    "percentile",
    "percentile_rank",
]

_RANK_METHODS = ("average", "min", "max", "dense", "ordinal")
_NAN_POLICIES = ("keep", "omit", "raise")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rank_1d(x: np.ndarray, method: str) -> np.ndarray:
    """Rank a NaN-free 1-D array. Core logic shared by ``rank``."""
    raise NotImplementedError


def _check_q(q) -> np.ndarray:
    """Validate percentiles are in [0, 100]; return as float ndarray."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank(x, method: str = "average", axis: int = -1, *, nan_policy: str = "keep"):
    """Rank values along ``axis`` with configurable tie handling.

    Parameters
    ----------
    x : array_like, shape (..., n, ...)
    method : {"average", "min", "max", "dense", "ordinal"}, default "average"
        How tied values are ranked. For ``[10, 20, 20, 30]``:
        average -> [1, 2.5, 2.5, 4]; min -> [1, 2, 2, 4];
        max -> [1, 3, 3, 4]; dense -> [1, 2, 2, 3]; ordinal -> [1, 2, 3, 4].
    axis : int, default -1
    nan_policy : {"keep", "omit", "raise"}, default "keep"
        keep: NaN inputs get NaN ranks; other values ranked among themselves.
        omit: same output as "keep" (NaNs excluded from ranking).
        raise: raise ``ValueError`` if any NaN is present.

    Returns
    -------
    ndarray of float, same shape as ``x``
        Float so "average" ranks and NaNs are representable.

    Raises
    ------
    ValueError
        If ``method`` or ``nan_policy`` is invalid, or NaNs found with
        ``nan_policy="raise"``.
    np.exceptions.AxisError
        If ``axis`` is out of range.

    Notes
    -----
    Time O(n log n) per slice (dominated by the stable argsort),
    space O(n).
    """
    raise NotImplementedError


def percentile(x, q, axis=None, *, method: str = "linear", nan_policy: str = "propagate"):
    """Compute the q-th percentile(s) of ``x``.

    Parameters
    ----------
    x : array_like
    q : float or array_like of float
        Percentile(s) in ``[0, 100]``.
    axis : int or None, default None
        None flattens ``x`` first.
    method : {"linear", "lower", "higher", "nearest", "midpoint"}, default "linear"
        Interpolation between order statistics; "linear" matches
        ``np.percentile``'s default.
    nan_policy : {"propagate", "omit", "raise"}, default "propagate"

    Returns
    -------
    scalar or ndarray
        Shape is ``q.shape`` + ``x.shape`` with ``axis`` removed.

    Raises
    ------
    ValueError
        If ``x`` is empty, any ``q`` is outside [0, 100], ``method`` is
        invalid, or NaNs found with ``nan_policy="raise"``.

    Notes
    -----
    Time O(n log n) per slice via sorting (O(n) possible with
    partitioning). Space O(n).
    """
    raise NotImplementedError


def percentile_rank(x, score, *, kind: str = "weak"):
    """Percentage of values in ``x`` at or below ``score`` (inverse of percentile).

    Parameters
    ----------
    x : array_like, shape (n,)
    score : scalar or array_like
    kind : {"weak", "strict", "mean"}, default "weak"
        weak: % of values <= score; strict: % of values < score;
        mean: average of weak and strict (fair to ties).

    Returns
    -------
    float or ndarray of float
        Values in [0, 100].

    Raises
    ------
    ValueError
        If ``x`` is empty or ``kind`` is invalid.

    Notes
    -----
    Time O(n log n + m log n) for m scores (sort once, then binary search).
    Space O(n).
    """
    raise NotImplementedError
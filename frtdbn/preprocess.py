"""Preprocessing helpers, including var-sortability defenses.

For synthetic benchmarking we full-sample standardize each regime so the
variance gradient that NOTEARS-style estimators can exploit (Reisach et al.
2021) is flattened. The real-data pipeline will add a rolling, past-only
z-score and a rank (Gaussian-copula) transform here later.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri


def standardize_columns(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Z-score each column to zero mean and unit variance.

    Columns with (near-)zero variance map to all zeros rather than NaNs/inf.
    """

    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")
    arr = np.asarray(x, dtype=float)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    return (arr - mean) / np.where(std < eps, 1.0, std)


def robust_scales(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """MAD-based positive per-column scales."""

    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")
    arr = np.asarray(x, dtype=float)
    med = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - med), axis=0)
    return np.maximum(1.4826 * mad, eps)


def rolling_zscore_past(x: np.ndarray, window: int = 250, min_periods: int = 30, eps: float = 1e-12) -> np.ndarray:
    """Past-only rolling z-score; row t uses rows strictly before t."""

    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")
    if window <= 1:
        raise ValueError("window must be greater than 1.")
    arr = np.asarray(x, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    for t in range(arr.shape[0]):
        start = max(0, t - window)
        hist = arr[start:t]
        if hist.shape[0] < min_periods:
            continue
        mean = np.nanmean(hist, axis=0)
        std = np.nanstd(hist, axis=0)
        out[t] = (arr[t] - mean) / np.where(std < eps, 1.0, std)
    return out


def rank_gaussianize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Columnwise rank transform to normal scores for copula robustness checks."""

    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")
    arr = np.asarray(x, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        valid = np.isfinite(col)
        n = int(valid.sum())
        if n == 0:
            continue
        order = np.argsort(col[valid], kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        probs = np.clip((ranks - 0.5) / n, eps, 1.0 - eps)
        out[valid, j] = ndtri(probs)
    return out


def apply_transform(x: np.ndarray, transform: str = "standardize") -> np.ndarray:
    """Dispatch a column transform by name (for the benchmark's robustness columns).

    ``"standardize"`` z-scores (flattens the variance gradient), ``"rank"`` maps
    each column to normal scores (distribution-free / Gaussian-copula), and
    ``"none"`` is the identity.
    """

    if transform == "standardize":
        return standardize_columns(x)
    if transform == "rank":
        return rank_gaussianize(x)
    if transform == "none":
        return np.asarray(x, dtype=float)
    raise ValueError(f"Unknown transform {transform!r}.")

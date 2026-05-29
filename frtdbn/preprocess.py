"""Preprocessing helpers, including var-sortability defenses.

For synthetic benchmarking we full-sample standardize each regime so the
variance gradient that NOTEARS-style estimators can exploit (Reisach et al.
2021) is flattened. The real-data pipeline will add a rolling, past-only
z-score and a rank (Gaussian-copula) transform here later.
"""

from __future__ import annotations

import numpy as np


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

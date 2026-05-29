"""Graph recovery metrics for synthetic experiments."""

from __future__ import annotations

import numpy as np


def binary_adjacency(weights: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    return np.abs(weights) > threshold


def shd_binary(true_adj: np.ndarray, pred_adj: np.ndarray) -> int:
    """Structural Hamming distance on directed binary adjacencies."""

    if true_adj.shape != pred_adj.shape:
        raise ValueError("Adjacency shapes must match.")
    return int(np.sum(true_adj.astype(bool) != pred_adj.astype(bool)))


def change_scores(W: list[np.ndarray], A: list[np.ndarray] | None = None) -> np.ndarray:
    """Absolute regime-1 minus regime-0 edge-change scores."""

    score = np.abs(W[1] - W[0])
    if A is not None:
        score = score + np.abs(A[1] - A[0])
    return score


def auroc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC without requiring sklearn in scripts."""

    y = labels.astype(bool).ravel()
    s = scores.ravel()
    pos = s[y]
    neg = s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = 0.0
    for value in pos:
        greater += float(np.sum(value > neg))
        greater += 0.5 * float(np.sum(value == neg))
    return greater / float(len(pos) * len(neg))

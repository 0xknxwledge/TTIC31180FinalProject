"""Graph recovery metrics for synthetic experiments."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


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


def acyclicity_numpy(W: np.ndarray) -> float:
    """NOTEARS acyclicity surrogate h(W)=tr(exp(W o W))-d."""

    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix.")
    return float(np.trace(expm(W * W)) - W.shape[0])


def graph_diagnostics(
    W: list[np.ndarray],
    A: list[np.ndarray],
    threshold: float = 1e-12,
) -> dict[str, float]:
    """Basic returned-graph diagnostics for reporting and guardrails."""

    if len(W) != len(A):
        raise ValueError("W and A must have the same number of regimes.")
    h_values = [acyclicity_numpy(w) for w in W]
    d = W[0].shape[0]
    offdiag = ~np.eye(d, dtype=bool)
    out: dict[str, float] = {
        "h_returned_max": float(max(abs(v) for v in h_values)),
        "n_regimes": float(len(W)),
    }
    if len(W) >= 2:
        out["delta_w_nnz"] = float(np.sum(np.abs(W[1] - W[0]) > threshold))
        out["delta_a_nnz"] = float(np.sum(np.abs(A[1] - A[0]) > threshold))
    for k, (wk, ak) in enumerate(zip(W, A)):
        out[f"h_returned_{k}"] = float(h_values[k])
        out[f"w_{k}_nnz"] = float(np.sum(np.abs(wk[offdiag]) > threshold))
        out[f"a_{k}_nnz"] = float(np.sum(np.abs(ak) > threshold))
    return out


def var_sortability(X: np.ndarray, W: np.ndarray, tol: float = 1e-9) -> float:
    """Reisach et al. (2021) var-sortability of data `X` under true DAG `W`.

    `X` is `n x d`; `W[i, j] != 0` encodes a directed edge `i -> j`. Returns the
    fraction of directed paths (of every length) along which marginal variance
    increases from parent to child, with ties counted as one half. A value near
    1.0 means variance order recovers causal order (NOTEARS can cheat); near 0.5
    means it carries no ordering information (e.g. standardized data).
    """

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    d = W.shape[0]
    if W.shape != (d, d) or X.shape[1] != d:
        raise ValueError("W must be d x d and match the number of columns in X.")

    E = (W != 0).astype(float)
    reachable = E.copy()
    var = np.var(X, axis=0, keepdims=True)  # (1, d)
    ratio_base = var / var.T  # ratio_base[i, j] = Var(child j) / Var(parent i)

    n_paths = 0.0
    n_correct = 0.0
    for _ in range(max(d - 1, 1)):
        present = reachable > 0
        n_paths += float(present.sum())
        ratio = np.where(present, ratio_base, 0.0)
        n_correct += float(np.sum(present & (ratio > 1.0 + tol)))
        n_correct += 0.5 * float(np.sum(present & (ratio <= 1.0 + tol) & (ratio >= 1.0 - tol)))
        reachable = reachable @ E
    if n_paths == 0.0:
        return float("nan")
    return n_correct / n_paths


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

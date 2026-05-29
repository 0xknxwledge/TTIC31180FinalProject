"""Synthetic data utilities for fused-regime DBN experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticDBN:
    """Ground-truth DBN parameters and sampled observations by regime."""

    W: list[np.ndarray]
    A: list[list[np.ndarray]]
    X: list[np.ndarray]
    changed_W: np.ndarray
    changed_A: np.ndarray


def _signed_uniform(mask: np.ndarray, rng: np.random.Generator, low: float, high: float) -> np.ndarray:
    values = rng.uniform(low, high, size=mask.shape)
    signs = rng.choice(np.array([-1.0, 1.0]), size=mask.shape)
    return mask.astype(float) * values * signs


def _make_ordered_er_dag(
    d: int,
    mean_degree: float,
    rng: np.random.Generator,
    protected_edges: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Sample a DAG whose natural index order is a valid topological order."""

    protected_edges = protected_edges or set()
    prob = min(mean_degree / max(d - 1, 1), 1.0)
    mask = np.zeros((d, d), dtype=bool)
    for i in range(d):
        for j in range(i + 1, d):
            if (i, j) in protected_edges or rng.random() < prob:
                mask[i, j] = True
    return mask


def _sample_inter_slice(d: int, p: int, mean_degree: float, rng: np.random.Generator) -> list[np.ndarray]:
    prob = min(mean_degree / max(d, 1), 1.0)
    out: list[np.ndarray] = []
    for lag in range(p):
        mask = rng.random((d, d)) < prob
        # Later lags get weaker coefficients to make stable VARs easier.
        scale = 1.0 / (lag + 1)
        out.append(_signed_uniform(mask, rng, 0.05 * scale, 0.18 * scale))
    return out


def _reduced_form_coefficients(W: np.ndarray, A: list[np.ndarray]) -> list[np.ndarray]:
    inv = np.linalg.inv(np.eye(W.shape[0]) - W)
    return [a @ inv for a in A]


def _companion_radius(coeffs: list[np.ndarray]) -> float:
    d = coeffs[0].shape[0]
    p = len(coeffs)
    companion = np.zeros((p * d, p * d))
    companion[:d, : p * d] = np.concatenate(coeffs, axis=1)
    if p > 1:
        companion[d:, :-d] = np.eye((p - 1) * d)
    return float(max(abs(np.linalg.eigvals(companion))))


def _scale_until_stable(W: np.ndarray, A: list[np.ndarray], max_radius: float = 0.92) -> list[np.ndarray]:
    coeffs = _reduced_form_coefficients(W, A)
    radius = _companion_radius(coeffs)
    if radius <= max_radius:
        return A
    shrink = max_radius / max(radius, 1e-12)
    return [a * shrink for a in A]


def simulate_dbn(
    W: np.ndarray,
    A: list[np.ndarray],
    n: int,
    rng: np.random.Generator,
    nu: float = 5.0,
    burn_in: int = 200,
) -> np.ndarray:
    """Simulate a linear structural DBN and return `n + p` rows.

    The returned matrix includes the initial lag rows so callers can build a
    lagged design with exactly `n` target observations.
    """

    d = W.shape[0]
    p = len(A)
    inv = np.linalg.inv(np.eye(d) - W)
    total = n + p + burn_in
    x = np.zeros((total, d), dtype=float)
    for t in range(p, total):
        lagged = np.zeros(d, dtype=float)
        for lag, a_lag in enumerate(A, start=1):
            lagged += x[t - lag] @ a_lag
        noise = rng.standard_t(df=nu, size=d) / np.sqrt(nu / (nu - 2.0))
        x[t] = (lagged + noise) @ inv
    return x[burn_in:]


def prepare_lagged_design(x: np.ndarray, p: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return contemporaneous targets and lag matrices for a DBN fit."""

    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")
    if len(x) <= p:
        raise ValueError("x must contain more rows than the lag order.")
    target = x[p:].astype(float, copy=False)
    lags = [x[p - lag : len(x) - lag].astype(float, copy=False) for lag in range(1, p + 1)]
    return target, lags


def make_regime_pair(
    d: int = 20,
    p: int = 1,
    n_per_regime: int = 200,
    mean_degree: float = 2.0,
    change_edges: int = 4,
    nu: float = 5.0,
    seed: int = 0,
) -> SyntheticDBN:
    """Create two related regimes with sparse regime-specific edge changes."""

    if d < 6:
        raise ValueError("d must be at least 6 so macro/crypto blocks are meaningful.")
    rng = np.random.default_rng(seed)
    macro = np.arange(0, max(2, d // 4))
    crypto = np.arange(max(2, d // 4), max(4, d // 2))

    candidates = [(int(i), int(j)) for i in macro for j in crypto if i < j]
    rng.shuffle(candidates)
    protected_list = candidates[:change_edges]
    protected = set(protected_list)

    base_mask = _make_ordered_er_dag(d, mean_degree, rng, protected_edges=set())
    event_mask = base_mask.copy()
    for i, j in protected:
        event_mask[i, j] = True

    W0 = _signed_uniform(base_mask, rng, 0.25, 0.8)
    W1 = W0.copy()
    new_weights = _signed_uniform(event_mask & ~base_mask, rng, 0.35, 0.9)
    W1 += new_weights

    A0 = _scale_until_stable(W0, _sample_inter_slice(d, p, mean_degree, rng))
    A1 = [a.copy() for a in A0]
    changed_A = np.zeros((p, d, d), dtype=bool)
    for lag in range(p):
        for i, j in protected_list[: max(1, change_edges // 2)]:
            A1[lag][i, j] += rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.2)
            changed_A[lag, i, j] = True
    A1 = _scale_until_stable(W1, A1)

    X0 = simulate_dbn(W0, A0, n=n_per_regime, rng=rng, nu=nu)
    X1 = simulate_dbn(W1, A1, n=n_per_regime, rng=rng, nu=nu)
    return SyntheticDBN(
        W=[W0, W1],
        A=[A0, A1],
        X=[X0, X1],
        changed_W=(event_mask != base_mask),
        changed_A=changed_A.any(axis=0),
    )

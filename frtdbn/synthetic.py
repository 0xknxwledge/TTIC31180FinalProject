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


def _max_stable_step(
    W0: np.ndarray,
    A0: list[np.ndarray],
    dW: np.ndarray,
    dA: list[np.ndarray],
    max_radius: float,
) -> float:
    """Largest step in (0, 1] applying the sparse delta while staying stable.

    A single scalar is applied to the whole delta, so the change support (and
    thus the ground-truth Delta) stays exactly the sampled sparse set — no
    global rescale that would contaminate unchanged edges. Smaller steps are
    monotonically more stable (they shrink toward the stable base), so the first
    stable step on a descending grid is the largest admissible one.
    """

    for step in (1.0, 0.8, 0.6, 0.4, 0.25, 0.15, 0.1, 0.05):
        W1 = W0 + step * dW
        A1 = [a0 + step * da for a0, da in zip(A0, dA)]
        if _companion_radius(_reduced_form_coefficients(W1, A1)) <= max_radius:
            return step
    return 0.05


def _sample_change_delta(
    W0: np.ndarray,
    A0: list[np.ndarray],
    macro: np.ndarray,
    crypto: np.ndarray,
    base_mask: np.ndarray,
    rng: np.random.Generator,
    n_changes: int,
    change_types: tuple[str, ...],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Sample a sparse intra/inter-slice delta concentrated in macro->crypto.

    Supports additions (new edges), removals (zeroing existing edges), and
    reweights (new weight, possibly sign-flipped) of existing edges. Additions
    keep `i < j` so the event graph remains acyclic.
    """

    d = W0.shape[0]
    p = len(A0)
    dW = np.zeros((d, d))
    dA = [np.zeros((d, d)) for _ in range(p)]

    def _block_then_global(in_block_pred, global_pred) -> list[tuple[int, int]]:
        block = [(int(i), int(j)) for i in macro for j in crypto if i < j and in_block_pred(i, j)]
        if len(block) >= n_changes:
            rng.shuffle(block)
            return block
        glob = [(i, j) for i in range(d) for j in range(i + 1, d) if global_pred(i, j)]
        merged = list(dict.fromkeys(block + glob))
        rng.shuffle(merged)
        return merged

    adds = _block_then_global(lambda i, j: not base_mask[i, j], lambda i, j: not base_mask[i, j])
    edges = _block_then_global(lambda i, j: base_mask[i, j], lambda i, j: base_mask[i, j])
    add_ptr = edge_ptr = 0

    for idx in range(n_changes):
        ctype = change_types[idx % len(change_types)]
        if ctype == "add" and add_ptr < len(adds):
            i, j = adds[add_ptr]
            add_ptr += 1
            dW[i, j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 0.9)
        elif ctype == "remove" and edge_ptr < len(edges):
            i, j = edges[edge_ptr]
            edge_ptr += 1
            dW[i, j] = -W0[i, j]
        elif ctype == "reweight" and edge_ptr < len(edges):
            i, j = edges[edge_ptr]
            edge_ptr += 1
            new_weight = rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 0.9)
            dW[i, j] = new_weight - W0[i, j]
        elif add_ptr < len(adds):  # fall back to an addition if the type ran out
            i, j = adds[add_ptr]
            add_ptr += 1
            dW[i, j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 0.9)

    a_candidates = [(int(i), int(j)) for i in macro for j in crypto] or [(0, 1)]
    rng.shuffle(a_candidates)
    for k in range(min(max(1, n_changes // 2), len(a_candidates))):
        i, j = a_candidates[k]
        lag = int(rng.integers(0, p))
        dA[lag][i, j] += rng.choice([-1.0, 1.0]) * rng.uniform(0.1, 0.25)

    return dW, dA


def make_regime_pair(
    d: int = 20,
    p: int = 1,
    n_per_regime: int = 200,
    mean_degree: float = 2.0,
    change_edges: int = 4,
    nu: float = 5.0,
    seed: int = 0,
    change_types: tuple[str, ...] = ("add", "remove", "reweight"),
    max_radius: float = 0.9,
    tol: float = 1e-8,
    n_event: int | None = None,
) -> SyntheticDBN:
    """Create two related regimes whose Delta is a sparse, realistic edge change.

    The ordinary system is stabilized once; the event regime applies a sparse
    delta (additions, removals, reweights) scaled by a single stable step. The
    change masks are read off the *realized* parameter differences, so they are
    correct by construction regardless of stabilization.

    `n_per_regime` sets the ordinary-regime sample size; `n_event` (default equal
    to `n_per_regime`) sets the event regime. The realistic fusion setting is
    `n_event << n_per_regime`.
    """

    if d < 6:
        raise ValueError("d must be at least 6 so macro/crypto blocks are meaningful.")
    rng = np.random.default_rng(seed)
    macro = np.arange(0, max(2, d // 4))
    crypto = np.arange(max(2, d // 4), max(4, d // 2))

    base_mask = _make_ordered_er_dag(d, mean_degree, rng)
    W0 = _signed_uniform(base_mask, rng, 0.25, 0.8)
    A0 = _scale_until_stable(W0, _sample_inter_slice(d, p, mean_degree, rng), max_radius)

    dW, dA = _sample_change_delta(W0, A0, macro, crypto, base_mask, rng, change_edges, change_types)
    step = _max_stable_step(W0, A0, dW, dA, max_radius)
    W1 = W0 + step * dW
    A1 = [a0 + step * da for a0, da in zip(A0, dA)]

    changed_W = np.abs(W1 - W0) > tol
    diff_a = np.stack([np.abs(a1 - a0) for a1, a0 in zip(A1, A0)])
    changed_A = (diff_a > tol).any(axis=0)

    n_event = n_per_regime if n_event is None else n_event
    X0 = simulate_dbn(W0, A0, n=n_per_regime, rng=rng, nu=nu)
    X1 = simulate_dbn(W1, A1, n=n_event, rng=rng, nu=nu)
    return SyntheticDBN(
        W=[W0, W1],
        A=[A0, A1],
        X=[X0, X1],
        changed_W=changed_W,
        changed_A=changed_A,
    )

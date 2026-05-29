"""Robustness helpers for real-data evaluation."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np

from frtdbn.model import FitConfig, FitResult, fit_fr_tdbn


def top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask for the k largest absolute scores."""

    if k <= 0:
        raise ValueError("k must be positive.")
    flat = np.abs(scores).ravel()
    k = min(k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    out = np.zeros(flat.size, dtype=bool)
    out[idx] = True
    return out.reshape(scores.shape)


def top_k_jaccard(score_mats: list[np.ndarray], k: int) -> float:
    """Average pairwise Jaccard overlap among top-k edge sets."""

    if len(score_mats) < 2:
        return float("nan")
    vals = []
    masks = [top_k_mask(s, k) for s in score_mats]
    for a, b in combinations(masks, 2):
        union = np.logical_or(a, b).sum()
        vals.append(float(np.logical_and(a, b).sum() / union) if union else float("nan"))
    return float(np.nanmean(vals))


def fit_restarts(
    targets_by_regime: list[np.ndarray],
    lags_by_regime: list[list[np.ndarray]],
    config: FitConfig,
    seeds: list[int],
) -> list[FitResult]:
    """Run multiple random restarts of the same fit configuration."""

    return [
        fit_fr_tdbn(targets_by_regime, lags_by_regime, replace(config, seed=int(seed)))
        for seed in seeds
    ]


def restart_summary(results: list[FitResult], top_k: int = 10) -> dict[str, float]:
    """Summarize restart stability from Delta_W rankings."""

    scores = [np.abs(r.Delta_W) for r in results if r.Delta_W is not None]
    if not scores:
        raise ValueError("Results must include Delta_W.")
    return {
        "n_restarts": float(len(scores)),
        "top_k": float(top_k),
        "top_k_jaccard": top_k_jaccard(scores, top_k),
        "h_returned_max": float(max(r.diagnostics.get("h_returned_max", np.nan) for r in results)),
    }


def block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap indices with replacement."""

    if n <= 0:
        raise ValueError("n must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    starts = np.arange(max(n - block_size + 1, 1))
    out: list[int] = []
    while len(out) < n:
        start = int(rng.choice(starts))
        out.extend(range(start, min(start + block_size, n)))
    return np.asarray(out[:n], dtype=int)


def resample_lagged_regimes(
    targets_by_regime: list[np.ndarray],
    lags_by_regime: list[list[np.ndarray]],
    block_size: int,
    seed: int,
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    """Block-bootstrap target/lag rows within each regime."""

    rng = np.random.default_rng(seed)
    targets_out: list[np.ndarray] = []
    lags_out: list[list[np.ndarray]] = []
    for target, lags in zip(targets_by_regime, lags_by_regime):
        idx = block_bootstrap_indices(len(target), block_size, rng)
        targets_out.append(target[idx])
        lags_out.append([lag[idx] for lag in lags])
    return targets_out, lags_out


def stability_selection(
    targets_by_regime: list[np.ndarray],
    lags_by_regime: list[list[np.ndarray]],
    config: FitConfig,
    n_bootstrap: int = 25,
    block_size: int = 24,
    threshold: float = 1e-12,
    seed: int = 0,
) -> dict[str, np.ndarray | float]:
    """Block-bootstrap edge frequencies for Delta_W and Delta_A."""

    delta_w_hits = None
    delta_a_hits = None
    for b in range(n_bootstrap):
        bt, bl = resample_lagged_regimes(targets_by_regime, lags_by_regime, block_size, seed + b)
        fit = fit_fr_tdbn(bt, bl, replace(config, seed=seed + b))
        dw = np.abs(fit.Delta_W) > threshold
        da = np.abs(fit.Delta_A) > threshold
        delta_w_hits = dw.astype(float) if delta_w_hits is None else delta_w_hits + dw
        delta_a_hits = da.astype(float) if delta_a_hits is None else delta_a_hits + da
    return {
        "delta_w_frequency": delta_w_hits / n_bootstrap,
        "delta_a_frequency": delta_a_hits / n_bootstrap,
        "n_bootstrap": float(n_bootstrap),
    }


def _block_groups(n: int, block_size: int) -> list[np.ndarray]:
    return [np.arange(start, min(start + block_size, n)) for start in range(0, n, block_size)]


def _assign_permuted_labels(
    labels: np.ndarray,
    rng: np.random.Generator,
    volatility: np.ndarray | None,
    n_bins: int,
    block_size: int | None,
) -> np.ndarray:
    labels = labels.astype(int)
    out = labels.copy()
    if block_size is not None and block_size > 1:
        blocks = _block_groups(len(labels), block_size)
        if volatility is None:
            order = rng.permutation(len(blocks))
            out[np.concatenate(blocks)] = np.concatenate([labels[blocks[i]] for i in order])
            return out
        block_vol = np.asarray([np.nanmean(volatility[idx]) for idx in blocks])
        quantiles = np.nanquantile(block_vol, np.linspace(0.0, 1.0, n_bins + 1))
        bins = np.digitize(block_vol, quantiles[1:-1], right=True)
        for b in range(n_bins):
            block_ids = np.flatnonzero(bins == b)
            if len(block_ids) == 0:
                continue
            shuffled = rng.permutation(block_ids)
            dest = np.concatenate([blocks[i] for i in block_ids])
            vals = np.concatenate([labels[blocks[i]] for i in shuffled])
            out[dest] = vals[: len(dest)]
        return out

    if volatility is None:
        rng.shuffle(out)
        return out
    quantiles = np.nanquantile(volatility, np.linspace(0.0, 1.0, n_bins + 1))
    bins = np.digitize(volatility, quantiles[1:-1], right=True)
    for b in range(n_bins):
        idx = np.flatnonzero(bins == b)
        shuffled = out[idx].copy()
        rng.shuffle(shuffled)
        out[idx] = shuffled
    return out


def permutation_null_delta_norm(
    targets_by_regime: list[np.ndarray],
    lags_by_regime: list[list[np.ndarray]],
    config: FitConfig,
    n_permutations: int = 20,
    seed: int = 0,
    volatility_match: bool = True,
    n_vol_bins: int = 5,
    block_size: int | None = None,
) -> np.ndarray:
    """Permutation null for ||Delta_W||_1 with optional block/volatility matching."""

    if len(targets_by_regime) != 2:
        raise NotImplementedError("Permutation null currently supports K=2.")
    rng = np.random.default_rng(seed)
    targets_all = np.concatenate(targets_by_regime, axis=0)
    lags_all = [np.concatenate([lags_by_regime[0][lag], lags_by_regime[1][lag]], axis=0) for lag in range(config.p)]
    labels = np.concatenate([
        np.zeros(len(targets_by_regime[0]), dtype=int),
        np.ones(len(targets_by_regime[1]), dtype=int),
    ])
    volatility = np.nanstd(targets_all, axis=1) if volatility_match else None
    norms = []
    for _ in range(n_permutations):
        perm = _assign_permuted_labels(labels, rng, volatility, n_vol_bins, block_size)
        perm_targets = [targets_all[perm == k] for k in (0, 1)]
        perm_lags = [[lag_mat[perm == k] for lag_mat in lags_all] for k in (0, 1)]
        fit = fit_fr_tdbn(perm_targets, perm_lags, replace(config, seed=int(rng.integers(0, 1_000_000))))
        norms.append(float(np.sum(np.abs(fit.Delta_W))))
    return np.asarray(norms, dtype=float)

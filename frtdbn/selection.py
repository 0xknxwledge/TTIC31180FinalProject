"""Principled hyperparameter selection for FR-tDBN.

The headline fits hand-pick ``lambda`` and ``gamma``; this module replaces that
with two standard criteria so the reported structure is defensible:

  * **held-out NLL** — fit on a time-ordered train split, score the full
    (constant-inclusive) density on the most-recent test block with
    train-derived scales (no leakage). This is the right criterion for the
    fusion strength ``gamma`` (a predictive question).
  * **BIC** ``= 2 * NLL + k * log(N)`` — an in-sample penalized fit where ``k``
    counts the nonzero free parameters (off-diagonal ``W`` plus all of ``A``).
    The right criterion for the sparsity level ``lambda``.

``select_hyperparameters`` grid-searches ``(lambda, gamma)`` under either
criterion and returns the argmin plus the full trace.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from frtdbn.evaluation import full_nll
from frtdbn.model import FitConfig, FitResult, fit_fr_tdbn
from frtdbn.preprocess import robust_scales


def count_nonzero_params(result: FitResult, threshold: float = 1e-8) -> int:
    """Free-parameter count: nonzero off-diagonal ``W`` plus all nonzero ``A``.

    ``W`` diagonals are structurally zero (no self-loops) so they never count;
    lagged self-edges in ``A`` are genuine parameters and do.
    """

    k = 0
    for W in result.W:
        off = np.array(W, dtype=float, copy=True)
        np.fill_diagonal(off, 0.0)
        k += int(np.count_nonzero(np.abs(off) > threshold))
    for A in result.A:
        k += int(np.count_nonzero(np.abs(np.asarray(A, dtype=float)) > threshold))
    return k


def bic_score(
    targets_by_regime: list[np.ndarray],
    lags_by_regime: list[list[np.ndarray]],
    result: FitResult,
    loss: str = "student_t",
    nu: float = 5.0,
    scales: np.ndarray | None = None,
    threshold: float = 1e-8,
) -> float:
    """BIC = 2 * NLL + k * log(N); lower is better."""

    nll = full_nll(targets_by_regime, lags_by_regime, result, loss=loss, nu=nu, scales=scales)
    n_obs = sum(t.shape[0] for t in targets_by_regime)
    k = count_nonzero_params(result, threshold=threshold)
    return float(2.0 * nll + k * math.log(n_obs))


def select_hyperparameters(
    train_targets: list[np.ndarray],
    train_lags: list[list[np.ndarray]],
    test_targets: list[np.ndarray],
    test_lags: list[list[np.ndarray]],
    base_config: FitConfig,
    lambdas: Sequence[float],
    gammas: Sequence[float],
    criterion: str = "heldout_nll",
    loss: str = "student_t",
    nu: float | None = None,
) -> dict:
    """Grid-search ``(lambda, gamma)`` under ``criterion`` (lower is better).

    For each pair, ``lambda`` sets both intra/inter sparsity and ``gamma`` sets
    both intra/inter fusion. ``"heldout_nll"`` fits on train and scores full NLL
    on test (train-derived scales); ``"bic"`` fits and scores BIC on train.
    """

    if criterion not in ("heldout_nll", "bic"):
        raise ValueError(f"Unknown criterion {criterion!r}.")

    nu_eff = base_config.nu if nu is None else nu
    train_scales = robust_scales(np.concatenate(train_targets, axis=0))
    trace: list[dict] = []
    for lam in lambdas:
        for gam in gammas:
            cfg = replace(
                base_config,
                loss=loss,
                lambda_w=lam,
                lambda_a=lam,
                gamma_w=gam,
                gamma_a=gam,
                nu=nu_eff,
            )
            fit = fit_fr_tdbn(train_targets, train_lags, cfg)
            if criterion == "heldout_nll":
                score = full_nll(test_targets, test_lags, fit, loss=loss, nu=nu_eff, scales=train_scales)
            else:  # bic
                score = bic_score(train_targets, train_lags, fit, loss=loss, nu=nu_eff, scales=train_scales)
            delta_w_l1 = float(np.abs(fit.Delta_W).sum()) if fit.Delta_W is not None else 0.0
            trace.append(
                {
                    "lambda": float(lam),
                    "gamma": float(gam),
                    "score": float(score),
                    "delta_w_l1": delta_w_l1,
                    "nnz_params": count_nonzero_params(fit),
                }
            )

    best = min(trace, key=lambda r: r["score"])
    return {
        "criterion": criterion,
        "loss": loss,
        "lambda": best["lambda"],
        "gamma": best["gamma"],
        "best_score": best["score"],
        "trace": trace,
    }

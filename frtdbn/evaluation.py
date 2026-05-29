"""Evaluation utilities: full likelihoods and simple ablation baselines."""

from __future__ import annotations

import math

import numpy as np

from frtdbn.model import FitConfig, FitResult, fit_fr_tdbn
from frtdbn.preprocess import robust_scales


def residuals_for_regime(target: np.ndarray, lags: list[np.ndarray], W: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Compute DBN residuals for one regime."""

    p = len(lags)
    d = target.shape[1]
    pred = target @ W
    A_lags = A.reshape(p, d, d)
    for lag in range(p):
        pred = pred + lags[lag] @ A_lags[lag]
    return target - pred


def student_t_nll_full(resid: np.ndarray, scales: np.ndarray, nu: float) -> float:
    """Full Student-t negative log likelihood, including constants."""

    r = np.asarray(resid, dtype=float)
    s = np.asarray(scales, dtype=float)
    if np.any(s <= 0):
        raise ValueError("scales must be positive.")
    const = math.lgamma(nu / 2.0) + 0.5 * math.log(nu * math.pi) - math.lgamma((nu + 1.0) / 2.0)
    nll = np.log(s) + const + 0.5 * (nu + 1.0) * np.log1p((r / s) ** 2 / nu)
    return float(np.sum(nll))


def gaussian_nll_full(resid: np.ndarray, scales: np.ndarray) -> float:
    """Full Gaussian negative log likelihood, including constants."""

    r = np.asarray(resid, dtype=float)
    s = np.asarray(scales, dtype=float)
    if np.any(s <= 0):
        raise ValueError("scales must be positive.")
    nll = np.log(s) + 0.5 * math.log(2.0 * math.pi) + 0.5 * (r / s) ** 2
    return float(np.sum(nll))


def full_nll(
    targets_by_regime: list[np.ndarray],
    lags_by_regime: list[list[np.ndarray]],
    result: FitResult,
    loss: str = "student_t",
    nu: float = 5.0,
    scales: np.ndarray | None = None,
) -> float:
    """Full (constant-inclusive) NLL of a fitted result on the *given* dataset.

    This evaluates whatever data is passed; pass a held-out test split (and
    train-derived ``scales``) for a genuine out-of-sample number — see
    ``svar_vs_dag_oos``.
    """

    scale_vec = robust_scales(np.concatenate(targets_by_regime, axis=0)) if scales is None else scales
    total = 0.0
    for k, target in enumerate(targets_by_regime):
        resid = residuals_for_regime(target, lags_by_regime[k], result.W[k], result.A[k])
        if loss == "student_t":
            total += student_t_nll_full(resid, scale_vec, nu)
        elif loss == "gaussian":
            total += gaussian_nll_full(resid, scale_vec)
        else:
            raise ValueError(f"Unknown loss {loss!r}.")
    return total


def svar_vs_dag_oos(
    train_targets: list[np.ndarray],
    train_lags: list[list[np.ndarray]],
    test_targets: list[np.ndarray],
    test_lags: list[list[np.ndarray]],
    config: FitConfig,
    loss: str = "student_t",
    nu: float | None = None,
) -> dict[str, float | bool]:
    """Out-of-sample W=0 ablation: does the contemporaneous DAG pay its way?

    Fits FR-tDBN and the W=0 SVAR baseline on the *train* split, then compares
    their full NLL on the *test* split using train-derived scales (no leakage).
    A lower DAG test NLL is the only honest evidence that ``W`` carries
    predictive weight; an in-sample comparison would favor the DAG mechanically
    because it has strictly more parameters.
    """

    nu_eff = config.nu if nu is None else nu
    scales = robust_scales(np.concatenate(train_targets, axis=0))
    dag = fit_fr_tdbn(train_targets, train_lags, config)
    svar = fit_svar_only(train_targets, train_lags)
    dag_nll = full_nll(test_targets, test_lags, dag, loss=loss, nu=nu_eff, scales=scales)
    svar_nll = full_nll(test_targets, test_lags, svar, loss=loss, nu=nu_eff, scales=scales)
    return {
        "dag_test_nll": float(dag_nll),
        "svar_test_nll": float(svar_nll),
        "dag_minus_svar": float(dag_nll - svar_nll),
        "dag_better": bool(dag_nll < svar_nll),
    }


def fit_svar_only(
    targets_by_regime: list[np.ndarray],
    lags_by_regime: list[list[np.ndarray]],
    ridge: float = 1e-6,
) -> FitResult:
    """Fit a W=0 structural-VAR-only baseline by ridge least squares."""

    W: list[np.ndarray] = []
    A: list[np.ndarray] = []
    for target, lags in zip(targets_by_regime, lags_by_regime):
        d = target.shape[1]
        design = np.concatenate(lags, axis=1)
        gram = design.T @ design + ridge * np.eye(design.shape[1])
        coef = np.linalg.solve(gram, design.T @ target)
        W.append(np.zeros((d, d)))
        A.append(coef)
    delta_w = (W[1] - W[0]).copy() if len(W) >= 2 else None
    delta_a = (A[1] - A[0]).copy() if len(A) >= 2 else None
    return FitResult(W=W, A=A, Delta_W=delta_w, Delta_A=delta_a)

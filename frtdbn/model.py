"""Fixed-nu fused-regime Student-t DBN estimator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class FitConfig:
    p: int
    lambda_w: float = 0.02
    lambda_a: float = 0.02
    gamma_w: float = 0.05
    gamma_a: float = 0.05
    nu: float = 5.0
    smooth_eps: float = 1e-6
    dtype: torch.dtype = torch.float64
    lbfgs_max_iter: int = 80
    outer_max_iter: int = 8
    h_tol: float = 1e-6
    rho_init: float = 1.0
    rho_multiplier: float = 10.0
    rho_max: float = 1e12
    seed: int = 0


@dataclass(frozen=True)
class FitResult:
    W: list[np.ndarray]
    A: list[np.ndarray]
    history: list[dict[str, float]] = field(default_factory=list)


def _as_tensor_list(values: list[np.ndarray], dtype: torch.dtype) -> list[torch.Tensor]:
    return [torch.as_tensor(v, dtype=dtype) for v in values]


def _robust_scales(targets: list[np.ndarray]) -> np.ndarray:
    pooled = np.concatenate(targets, axis=0)
    med = np.median(pooled, axis=0)
    mad = np.median(np.abs(pooled - med), axis=0)
    scales = 1.4826 * mad
    return np.maximum(scales, 1e-3)


def _acyclicity(W: torch.Tensor) -> torch.Tensor:
    d = W.shape[-1]
    return torch.trace(torch.matrix_exp(W * W)) - d


def _smooth_l1(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(x * x + eps).sum()


def _student_t_nll(resid: torch.Tensor, scales: torch.Tensor, nu: float) -> torch.Tensor:
    scaled_sq = (resid / scales) ** 2
    nll = torch.log(scales) + 0.5 * (nu + 1.0) * torch.log1p(scaled_sq / nu)
    return nll.sum()


def _pack_A(A_param: torch.Tensor, p: int) -> list[torch.Tensor]:
    return [A_param[:, lag, :, :] for lag in range(p)]


def fit_fr_tdbn(
    targets_by_regime: list[np.ndarray],
    lags_by_regime: list[list[np.ndarray]],
    config: FitConfig,
    scales: np.ndarray | None = None,
) -> FitResult:
    """Fit a fused-regime Student-t DBN with fixed `nu` and fixed scales.

    Parameters
    ----------
    targets_by_regime:
        List of `n_k x d` contemporaneous matrices.
    lags_by_regime:
        For each regime, a list of `p` lag matrices, each `n_k x d`.
    config:
        Optimization and penalty settings.
    scales:
        Optional positive per-variable scales. Robust MAD scales are used by
        default and shared across regimes.
    """

    if len(targets_by_regime) < 2:
        raise ValueError("At least two regimes are required for fusion.")
    K = len(targets_by_regime)
    d = targets_by_regime[0].shape[1]
    if any(x.shape[1] != d for x in targets_by_regime):
        raise ValueError("All regimes must have the same number of variables.")
    if any(len(lags) != config.p for lags in lags_by_regime):
        raise ValueError("Each regime must provide exactly p lag matrices.")

    torch.manual_seed(config.seed)
    X = _as_tensor_list(targets_by_regime, config.dtype)
    Y = [_as_tensor_list(lags, config.dtype) for lags in lags_by_regime]
    scale_arr = _robust_scales(targets_by_regime) if scales is None else np.asarray(scales, dtype=float)
    if scale_arr.shape != (d,) or np.any(scale_arr <= 0):
        raise ValueError("scales must be a positive vector with length d.")
    scale_t = torch.as_tensor(scale_arr, dtype=config.dtype)
    offdiag = (torch.ones((d, d), dtype=config.dtype) - torch.eye(d, dtype=config.dtype))

    W_param = torch.nn.Parameter(0.01 * torch.randn((K, d, d), dtype=config.dtype))
    A_param = torch.nn.Parameter(0.01 * torch.randn((K, config.p, d, d), dtype=config.dtype))

    rho = config.rho_init
    alpha = torch.zeros(K, dtype=config.dtype)
    history: list[dict[str, float]] = []
    prev_h = float("inf")

    for outer in range(config.outer_max_iter):
        optimizer = torch.optim.LBFGS(
            [W_param, A_param],
            max_iter=config.lbfgs_max_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            W = W_param * offdiag
            A_lags = _pack_A(A_param, config.p)
            nll = torch.zeros((), dtype=config.dtype)
            sparsity = torch.zeros((), dtype=config.dtype)
            h_penalty = torch.zeros((), dtype=config.dtype)
            h_values = []

            for k in range(K):
                pred = X[k] @ W[k]
                for lag in range(config.p):
                    pred = pred + Y[k][lag] @ A_lags[lag][k]
                resid = X[k] - pred
                nll = nll + _student_t_nll(resid, scale_t, config.nu)
                sparsity = sparsity + config.lambda_w * _smooth_l1(W[k], config.smooth_eps)
                for lag in range(config.p):
                    sparsity = sparsity + config.lambda_a * _smooth_l1(A_lags[lag][k], config.smooth_eps)
                h_k = _acyclicity(W[k])
                h_values.append(h_k)
                h_penalty = h_penalty + 0.5 * rho * h_k * h_k + alpha[k] * h_k

            fusion = torch.zeros((), dtype=config.dtype)
            for k in range(K):
                for ell in range(k + 1, K):
                    fusion = fusion + config.gamma_w * _smooth_l1(W[k] - W[ell], config.smooth_eps)
                    for lag in range(config.p):
                        fusion = fusion + config.gamma_a * _smooth_l1(
                            A_lags[lag][k] - A_lags[lag][ell],
                            config.smooth_eps,
                        )

            total_n = sum(x.shape[0] for x in X)
            loss = (nll + sparsity + fusion) / total_n + h_penalty
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        with torch.no_grad():
            W_now = W_param * offdiag
            h = torch.stack([_acyclicity(W_now[k]) for k in range(K)])
            max_h = float(torch.max(torch.abs(h)).detach().cpu())
            history.append(
                {
                    "outer": float(outer),
                    "loss": float(loss.detach().cpu()),
                    "max_h": max_h,
                    "rho": float(rho),
                }
            )
            if max_h <= config.h_tol:
                break
            if max_h > 0.25 * prev_h and rho < config.rho_max:
                rho *= config.rho_multiplier
            else:
                alpha = alpha + rho * h
                prev_h = max_h

    with torch.no_grad():
        W_final = (W_param * offdiag).detach().cpu().numpy()
        A_final = A_param.detach().cpu().numpy()
    W_out = [W_final[k].copy() for k in range(K)]
    A_out = [A_final[k].reshape(config.p * d, d).copy() for k in range(K)]
    return FitResult(W=W_out, A=A_out, history=history)

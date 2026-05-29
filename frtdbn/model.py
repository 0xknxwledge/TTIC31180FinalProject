"""Fixed-nu fused-regime Student-t DBN estimator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from frtdbn.metrics import graph_diagnostics
from frtdbn.prox import fused_lasso_prox_pair


@dataclass(frozen=True)
class FitConfig:
    p: int
    lambda_w: float = 0.02
    lambda_a: float = 0.02
    gamma_w: float = 0.05
    gamma_a: float = 0.05
    nu: float = 5.0
    loss: str = "student_t"
    huber_delta: float = 1.345
    nig_alpha: float = 1.0
    solver: str = "lbfgs_smooth"
    fusion: str = "uniform"
    adaptive_pilot: str = "uniform"
    adaptive_eps: float = 0.05
    smooth_eps: float = 1e-6
    dtype: torch.dtype = torch.float64
    lbfgs_max_iter: int = 80
    outer_max_iter: int = 8
    h_tol: float = 1e-6
    rho_init: float = 1.0
    rho_multiplier: float = 10.0
    rho_max: float = 1e12
    rho_admm: float = 1.0
    t_admm: int = 3
    seed: int = 0


@dataclass(frozen=True)
class FitResult:
    W: list[np.ndarray]
    A: list[np.ndarray]
    history: list[dict[str, float]] = field(default_factory=list)
    Delta_W: np.ndarray | None = None
    Delta_A: np.ndarray | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)


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


def _gaussian_nll(resid: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood up to an additive constant.

    Drops the 0.5*log(2*pi) term, which does not affect the argmin at fixed
    scales; used as the squared-loss (DYNOTEARS-equivalent) arm of the grid.
    """

    return (0.5 * (resid / scales) ** 2 + torch.log(scales)).sum()


def _pseudo_huber_nll(resid: torch.Tensor, scales: torch.Tensor, delta: float) -> torch.Tensor:
    """Pseudo-Huber data term on scale-normalized residuals.

    ``delta`` is the quadratic-to-linear crossover in robust-scale (sigma)
    units: quadratic for ``|z| << delta``, linear for ``|z| >> delta`` with a
    *bounded* (non-redescending) influence, unlike the Student-t. This is a
    robust loss, not a normalized log-density, so it is used only for the
    recovery ablation (AUROC), not the held-out density comparisons. The
    ``log(scales)`` term is constant at fixed scales and does not affect the
    argmin; it is kept for consistency with the other arms.
    """

    z = resid / scales
    huber = (delta * delta) * (torch.sqrt(1.0 + (z / delta) ** 2) - 1.0)
    return (huber + torch.log(scales)).sum()


class _LogBesselK1(torch.autograd.Function):
    """Numerically stable, differentiable ``log K_1(x)`` for ``x > 0``.

    ``torch.special.modified_bessel_k1`` has no autograd backward, and ``K_1``
    underflows for large ``x``; we evaluate ``log K_1`` via the exponentially
    scaled Bessel (``e^x K_1(x)``) and supply the analytic derivative
    ``d/dx log K_1(x) = -K_0(x)/K_1(x) - 1/x`` (verified against finite
    differences), using the ratio of scaled Bessels so it stays stable.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.log(torch.special.scaled_modified_bessel_k1(x)) - x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        sk0 = torch.special.scaled_modified_bessel_k0(x)
        sk1 = torch.special.scaled_modified_bessel_k1(x)
        return grad_output * (-(sk0 / sk1) - 1.0 / x)


def _nig_nll(resid: torch.Tensor, scales: torch.Tensor, alpha: float) -> torch.Tensor:
    """Symmetric Normal-Inverse-Gaussian negative log-likelihood.

    A normal variance-mean mixture with an inverse-Gaussian mixing density
    (Barndorff-Nielsen 1997): heavy-tailed, real-support, and a *proper*
    density (unlike pseudo-Huber). We fit the standardized, symmetric case
    (beta=0, mu=0, scale delta=alpha so the variance is exactly 1), with
    ``alpha`` the single tail-steepness knob (small = heavier; alpha -> inf
    recovers Gaussian). Residuals are scaled by ``sigma`` with the matching
    ``log sigma`` Jacobian term, as for the Student-t arm.
    """

    z = resid / scales
    r = torch.sqrt(alpha * alpha + z * z)            # sqrt(delta^2 + z^2), delta=alpha
    log_k1 = _LogBesselK1.apply(alpha * r)
    const = math.log(math.pi) - 2.0 * math.log(alpha) - alpha * alpha
    nll_std = const + torch.log(r) - log_k1          # -log f_z(z; alpha)
    return (nll_std + torch.log(scales)).sum()


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

    if config.loss not in ("student_t", "gaussian", "pseudo_huber", "nig"):
        raise ValueError(
            f"Unknown loss {config.loss!r}; expected 'student_t', 'gaussian', "
            "'pseudo_huber', or 'nig'."
        )
    if config.solver not in ("lbfgs_smooth", "admm"):
        raise ValueError(f"Unknown solver {config.solver!r}; expected 'lbfgs_smooth' or 'admm'.")
    if config.fusion not in ("uniform", "adaptive"):
        raise ValueError(f"Unknown fusion {config.fusion!r}; expected 'uniform' or 'adaptive'.")
    if config.adaptive_pilot not in ("uniform", "independent"):
        raise ValueError(
            f"Unknown adaptive_pilot {config.adaptive_pilot!r}; expected 'uniform' or 'independent'."
        )
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

    if config.solver == "admm":
        return _fit_admm(X, Y, scale_t, offdiag, K, d, config)

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
                if config.loss == "student_t":
                    nll = nll + _student_t_nll(resid, scale_t, config.nu)
                elif config.loss == "pseudo_huber":
                    nll = nll + _pseudo_huber_nll(resid, scale_t, config.huber_delta)
                elif config.loss == "nig":
                    nll = nll + _nig_nll(resid, scale_t, config.nig_alpha)
                else:
                    nll = nll + _gaussian_nll(resid, scale_t)
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
    return _make_result(W_out, A_out, history)


def _adaptive_weights(delta: np.ndarray, gamma0: float, eps: float) -> np.ndarray:
    """Adaptive-lasso fusion weights: gamma0 / (|delta_pilot| + eps).

    Large pilot changes get a small penalty (preserved, ~unbiased); near-zero
    pilot changes get the maximum penalty gamma0/eps (shrunk to exactly zero).
    """

    return gamma0 / (np.abs(delta) + eps)


def _gamma_for_lag(gamma_a, lag: int, rho: float):
    g = np.asarray(gamma_a)
    return (g if g.ndim == 0 else g[lag]) / rho


def _make_result(W_out: list[np.ndarray], A_out: list[np.ndarray], history: list[dict[str, float]]) -> FitResult:
    Delta_W = (W_out[1] - W_out[0]).copy() if len(W_out) >= 2 else None
    Delta_A = (A_out[1] - A_out[0]).copy() if len(A_out) >= 2 else None
    diagnostics = graph_diagnostics(W_out, A_out)
    if history:
        diagnostics["theta_h_logged_last"] = float(history[-1].get("max_h", np.nan))
        diagnostics["primal_res_last"] = float(history[-1].get("primal_res", np.nan))
        diagnostics["dual_res_last"] = float(history[-1].get("dual_res", np.nan))
    return FitResult(
        W=W_out,
        A=A_out,
        history=history,
        Delta_W=Delta_W,
        Delta_A=Delta_A,
        diagnostics=diagnostics,
    )


def _run_admm(
    X: list[torch.Tensor],
    Y: list[list[torch.Tensor]],
    scale_t: torch.Tensor,
    offdiag: torch.Tensor,
    K: int,
    d: int,
    config: FitConfig,
    gamma_w,
    gamma_a,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """One consensus-ADMM run with given fusion weights (scalar or per-edge).

    Smooth data + acyclicity terms are handled by L-BFGS (the Theta-update); the
    sparsity + fusion penalties are applied by the closed-form fused prox (the
    Z-update), which yields exactly sparse `W^k`. `gamma_w` is a scalar or (d,d)
    array; `gamma_a` is a scalar or (p,d,d) array.
    """

    if K != 2:
        raise NotImplementedError("The ADMM fused solver currently supports K=2 regimes.")
    p = config.p
    dtype = config.dtype
    N = float(sum(x.shape[0] for x in X))
    offdiag_np = offdiag.detach().cpu().numpy()
    gamma_w_arr = np.asarray(gamma_w, dtype=float)

    torch.manual_seed(config.seed)
    W_param = torch.nn.Parameter(0.01 * torch.randn((K, d, d), dtype=dtype))
    A_param = torch.nn.Parameter(0.01 * torch.randn((K, p, d, d), dtype=dtype))

    with torch.no_grad():
        Vw = (W_param * offdiag).detach().cpu().numpy().copy()
        Va = A_param.detach().cpu().numpy().copy()
    Uw = np.zeros_like(Vw)
    Ua = np.zeros_like(Va)

    rho = float(config.rho_admm)
    rho_h = config.rho_init
    alpha = torch.zeros(K, dtype=dtype)
    history: list[dict[str, float]] = []
    prev_h = float("inf")

    def nll_term(resid: torch.Tensor) -> torch.Tensor:
        if config.loss == "student_t":
            return _student_t_nll(resid, scale_t, config.nu)
        if config.loss == "pseudo_huber":
            return _pseudo_huber_nll(resid, scale_t, config.huber_delta)
        if config.loss == "nig":
            return _nig_nll(resid, scale_t, config.nig_alpha)
        return _gaussian_nll(resid, scale_t)

    primal = dual = 0.0
    for outer in range(config.outer_max_iter):
        for _ in range(max(1, config.t_admm)):
            Vw_t = torch.as_tensor(Vw, dtype=dtype)
            Va_t = torch.as_tensor(Va, dtype=dtype)
            Uw_t = torch.as_tensor(Uw, dtype=dtype)
            Ua_t = torch.as_tensor(Ua, dtype=dtype)

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
                A_lags = _pack_A(A_param, p)
                nll = torch.zeros((), dtype=dtype)
                h_penalty = torch.zeros((), dtype=dtype)
                for k in range(K):
                    pred = X[k] @ W[k]
                    for lag in range(p):
                        pred = pred + Y[k][lag] @ A_lags[lag][k]
                    resid = X[k] - pred
                    nll = nll + nll_term(resid)
                    h_k = _acyclicity(W[k])
                    h_penalty = h_penalty + 0.5 * rho_h * h_k * h_k + alpha[k] * h_k
                quad = 0.5 * rho * (
                    (W - Vw_t + Uw_t).pow(2).sum() + (A_param - Va_t + Ua_t).pow(2).sum()
                )
                loss = nll / N + h_penalty + quad
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                W_now = (W_param * offdiag).detach().cpu().numpy()
                A_now = A_param.detach().cpu().numpy()

            Vw_prev, Va_prev = Vw, Va
            aw0, aw1 = W_now[0] + Uw[0], W_now[1] + Uw[1]
            vw0, vw1 = fused_lasso_prox_pair(aw0, aw1, lam=config.lambda_w / rho, gam=gamma_w_arr / rho)
            Vw = np.stack([vw0 * offdiag_np, vw1 * offdiag_np])
            Va = np.empty_like(Va_prev)
            for lag in range(p):
                aa0, aa1 = A_now[0, lag] + Ua[0, lag], A_now[1, lag] + Ua[1, lag]
                va0, va1 = fused_lasso_prox_pair(
                    aa0, aa1, lam=config.lambda_a / rho, gam=_gamma_for_lag(gamma_a, lag, rho)
                )
                Va[0, lag], Va[1, lag] = va0, va1

            Uw = Uw + W_now - Vw
            Ua = Ua + A_now - Va

            primal = float(np.sqrt(((W_now - Vw) ** 2).sum() + ((A_now - Va) ** 2).sum()))
            dual = float(rho * np.sqrt(((Vw - Vw_prev) ** 2).sum() + ((Va - Va_prev) ** 2).sum()))

        with torch.no_grad():
            W_masked = W_param * offdiag
            h = torch.stack([_acyclicity(W_masked[k]) for k in range(K)])
            max_h = float(torch.max(torch.abs(h)).detach().cpu())

        history.append(
            {
                "outer": float(outer),
                "max_h": max_h,
                "primal_res": primal,
                "dual_res": dual,
                "rho": rho,
                "rho_h": float(rho_h),
            }
        )
        if max_h <= config.h_tol:
            break
        if max_h > 0.25 * prev_h and rho_h < config.rho_max:
            rho_h *= config.rho_multiplier
        else:
            alpha = alpha + rho_h * h
            prev_h = max_h

    return Vw, Va, history


def _fit_admm(
    X: list[torch.Tensor],
    Y: list[list[torch.Tensor]],
    scale_t: torch.Tensor,
    offdiag: torch.Tensor,
    K: int,
    d: int,
    config: FitConfig,
) -> FitResult:
    """Dispatch uniform vs adaptive fused ADMM; the adaptive run reweights the
    fusion penalty per edge using an independent (gamma=0) pilot fit."""

    if config.fusion == "uniform":
        Vw, Va, history = _run_admm(X, Y, scale_t, offdiag, K, d, config, config.gamma_w, config.gamma_a)
    else:  # adaptive: stage-1 pilot, then reweight the fusion penalty and refit
        if config.adaptive_pilot == "uniform":
            pilot_gw, pilot_ga = config.gamma_w, config.gamma_a  # Candes-style reweighting
        else:
            pilot_gw, pilot_ga = 0.0, 0.0  # independent (gamma=0) initial estimator
        pilot_w, pilot_a, _ = _run_admm(X, Y, scale_t, offdiag, K, d, config, pilot_gw, pilot_ga)
        gamma_w_mat = _adaptive_weights(pilot_w[1] - pilot_w[0], config.gamma_w, config.adaptive_eps)
        gamma_a_mat = _adaptive_weights(pilot_a[1] - pilot_a[0], config.gamma_a, config.adaptive_eps)
        Vw, Va, history = _run_admm(X, Y, scale_t, offdiag, K, d, config, gamma_w_mat, gamma_a_mat)

    p = config.p
    W_out = [Vw[k].copy() for k in range(K)]
    A_out = [Va[k].reshape(p * d, d).copy() for k in range(K)]
    return _make_result(W_out, A_out, history)

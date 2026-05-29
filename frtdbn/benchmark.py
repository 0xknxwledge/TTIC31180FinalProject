"""Var-sortability-honest synthetic benchmark for FR-tDBN.

Runs the {Gaussian, Student-t} x {independent, fused} grid in a single
optimizer so gains can be attributed to the heavy-tail loss, the fusion
penalty, or both. Each regime is full-sample standardized before fitting so
recovery cannot ride the variance gradient (Reisach et al. 2021); the
var-sortability of the data is reported both raw and as fitted.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from frtdbn.metrics import auroc_binary, change_scores, var_sortability
from frtdbn.model import FitConfig, fit_fr_tdbn
from frtdbn.preprocess import standardize_columns
from frtdbn.synthetic import make_regime_pair, prepare_lagged_design


def _lagged_change_score(A: list[np.ndarray], p: int, d: int) -> np.ndarray:
    """Max-over-lags absolute regime change of inter-slice coefficients."""

    a0 = A[0].reshape(p, d, d)
    a1 = A[1].reshape(p, d, d)
    return np.max(np.abs(a1 - a0), axis=0)


def run_one(
    seed: int,
    n: int,
    d: int,
    p: int,
    loss: str,
    gamma: float,
    gamma_a: float | None = None,
    solver: str = "lbfgs_smooth",
    standardize: bool = True,
    lambda_reg: float = 0.03,
    nu: float = 5.0,
    change_edges: int = 4,
    lbfgs_max_iter: int = 20,
    outer_max_iter: int = 5,
) -> dict[str, float | str | bool]:
    """Fit one cell of the benchmark grid and score change-edge recovery."""

    synth = make_regime_pair(d=d, p=p, n_per_regime=n, nu=nu, seed=seed, change_edges=change_edges)
    varsort_raw = var_sortability(synth.X[0], synth.W[0])

    series = [standardize_columns(x) if standardize else x for x in synth.X]
    varsort_fit = var_sortability(series[0], synth.W[0])

    targets, lags = [], []
    for x in series:
        target, lagged = prepare_lagged_design(x, p=p)
        targets.append(target)
        lags.append(lagged)

    gamma_a = gamma if gamma_a is None else gamma_a
    cfg = FitConfig(
        p=p,
        loss=loss,
        solver=solver,
        lambda_w=lambda_reg,
        lambda_a=lambda_reg,
        gamma_w=gamma,
        gamma_a=gamma_a,
        nu=nu,
        lbfgs_max_iter=lbfgs_max_iter,
        outer_max_iter=outer_max_iter,
        h_tol=1e-5,
        seed=seed,
    )
    start = time.perf_counter()
    fit = fit_fr_tdbn(targets, lags, cfg)
    seconds = time.perf_counter() - start

    return {
        "seed": int(seed),
        "n": int(n),
        "d": int(d),
        "p": int(p),
        "nu": float(nu),
        "change_edges": int(change_edges),
        "loss": loss,
        "solver": solver,
        "gamma": float(gamma),
        "gamma_w": float(gamma),
        "gamma_a": float(gamma_a),
        "standardize": bool(standardize),
        "varsort_raw": float(varsort_raw),
        "varsort_fit": float(varsort_fit),
        "auroc_change_w": float(auroc_binary(synth.changed_W, change_scores(fit.W))),
        "auroc_change_a": float(auroc_binary(synth.changed_A, _lagged_change_score(fit.A, p, d))),
        "delta_w_l1": float(np.abs(fit.Delta_W).sum()) if fit.Delta_W is not None
        else float(np.abs(fit.W[1] - fit.W[0]).sum()),
        "max_h": float(fit.history[-1]["max_h"]),
        "seconds": float(seconds),
    }


def _mean_se(values: list[float]) -> tuple[float, float]:
    """Mean and standard error of the mean (se = 0 for a single value)."""

    mean = statistics.mean(values)
    se = statistics.stdev(values) / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return float(mean), float(se)


def summarize_grid(rows: list[dict]) -> list[dict]:
    """Aggregate per-fit rows into per-(solver, loss, fusion) mean +/- se cells."""

    cells: dict[tuple[int, float, str, str, str, float, float], list[dict]] = {}
    for row in rows:
        gamma_w = float(row.get("gamma_w", row["gamma"]))
        gamma_a = float(row.get("gamma_a", row["gamma"]))
        key = (
            int(row["p"]),
            float(row.get("nu", float("nan"))),
            row["solver"],
            row["loss"],
            "fused" if (gamma_w > 0 or gamma_a > 0) else "indep",
            gamma_w,
            gamma_a,
        )
        cells.setdefault(key, []).append(row)

    out: list[dict] = []
    for (p, nu, solver, loss, fusion, gamma_w, gamma_a), group in cells.items():
        w_mean, w_se = _mean_se([r["auroc_change_w"] for r in group])
        a_mean, a_se = _mean_se([r["auroc_change_a"] for r in group])
        out.append(
            {
                "solver": solver,
                "loss": loss,
                "fusion": fusion,
                "p": p,
                "nu": nu,
                "gamma_w": gamma_w,
                "gamma_a": gamma_a,
                "n_fits": len(group),
                "auroc_w_mean": w_mean,
                "auroc_w_se": w_se,
                "auroc_a_mean": a_mean,
                "auroc_a_se": a_se,
                "max_h_mean": float(statistics.mean(r["max_h"] for r in group)),
                "seconds_mean": float(statistics.mean(r["seconds"] for r in group)),
            }
        )
    return out


def run_grid(
    d: int,
    p: int,
    n_values: list[int],
    seeds: list[int],
    losses: tuple[str, ...] = ("gaussian", "student_t"),
    gammas: tuple[float, ...] = (0.0, 0.04),
    gamma_as: tuple[float, ...] | None = None,
    solvers: tuple[str, ...] = ("lbfgs_smooth", "admm"),
    change_edges_values: tuple[int, ...] = (4,),
    standardize: bool = True,
    lambda_reg: float = 0.03,
    nu: float = 5.0,
    lbfgs_max_iter: int = 20,
    outer_max_iter: int = 5,
) -> list[dict[str, float | str | bool]]:
    """Run the {solver} x {loss} x {fusion} x {change_edges} grid over n and seeds."""

    gamma_as = gammas if gamma_as is None else gamma_as
    rows: list[dict[str, float | str | bool]] = []
    for n in n_values:
        for seed in seeds:
            for solver in solvers:
                for loss in losses:
                    for gamma in gammas:
                        for gamma_a in gamma_as:
                            for change_edges in change_edges_values:
                                rows.append(
                                    run_one(
                                        seed=seed,
                                        n=n,
                                        d=d,
                                        p=p,
                                        loss=loss,
                                        gamma=gamma,
                                        gamma_a=gamma_a,
                                        solver=solver,
                                        standardize=standardize,
                                        lambda_reg=lambda_reg,
                                        nu=nu,
                                        change_edges=change_edges,
                                        lbfgs_max_iter=lbfgs_max_iter,
                                        outer_max_iter=outer_max_iter,
                                    )
                                )
    return rows

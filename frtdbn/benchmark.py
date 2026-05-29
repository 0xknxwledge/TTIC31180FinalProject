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
    solver: str = "lbfgs_smooth",
    standardize: bool = True,
    lambda_reg: float = 0.03,
    nu: float = 5.0,
    lbfgs_max_iter: int = 20,
    outer_max_iter: int = 5,
) -> dict[str, float | str | bool]:
    """Fit one cell of the benchmark grid and score change-edge recovery."""

    synth = make_regime_pair(d=d, p=p, n_per_regime=n, nu=nu, seed=seed)
    varsort_raw = var_sortability(synth.X[0], synth.W[0])

    series = [standardize_columns(x) if standardize else x for x in synth.X]
    varsort_fit = var_sortability(series[0], synth.W[0])

    targets, lags = [], []
    for x in series:
        target, lagged = prepare_lagged_design(x, p=p)
        targets.append(target)
        lags.append(lagged)

    cfg = FitConfig(
        p=p,
        loss=loss,
        solver=solver,
        lambda_w=lambda_reg,
        lambda_a=lambda_reg,
        gamma_w=gamma,
        gamma_a=gamma,
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
        "loss": loss,
        "solver": solver,
        "gamma": float(gamma),
        "standardize": bool(standardize),
        "varsort_raw": float(varsort_raw),
        "varsort_fit": float(varsort_fit),
        "auroc_change_w": float(auroc_binary(synth.changed_W, change_scores(fit.W))),
        "auroc_change_a": float(auroc_binary(synth.changed_A, _lagged_change_score(fit.A, p, d))),
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

    cells: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["solver"], row["loss"], "fused" if row["gamma"] > 0 else "indep")
        cells.setdefault(key, []).append(row)

    out: list[dict] = []
    for (solver, loss, fusion), group in cells.items():
        w_mean, w_se = _mean_se([r["auroc_change_w"] for r in group])
        a_mean, a_se = _mean_se([r["auroc_change_a"] for r in group])
        out.append(
            {
                "solver": solver,
                "loss": loss,
                "fusion": fusion,
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
    solvers: tuple[str, ...] = ("lbfgs_smooth", "admm"),
    standardize: bool = True,
    lambda_reg: float = 0.03,
    nu: float = 5.0,
    lbfgs_max_iter: int = 20,
    outer_max_iter: int = 5,
) -> list[dict[str, float | str | bool]]:
    """Run the full {solver} x {loss} x {fusion} grid over the requested n and seeds."""

    rows: list[dict[str, float | str | bool]] = []
    for n in n_values:
        for seed in seeds:
            for solver in solvers:
                for loss in losses:
                    for gamma in gammas:
                        rows.append(
                            run_one(
                                seed=seed,
                                n=n,
                                d=d,
                                p=p,
                                loss=loss,
                                gamma=gamma,
                                solver=solver,
                                standardize=standardize,
                                lambda_reg=lambda_reg,
                                nu=nu,
                                lbfgs_max_iter=lbfgs_max_iter,
                                outer_max_iter=outer_max_iter,
                            )
                        )
    return rows

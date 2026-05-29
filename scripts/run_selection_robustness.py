"""Principled hyperparameter selection + rank-transform robustness column.

Three deliverables for the paper's rigor section, composing tested library
functions only:

  (A) Synthetic (lambda, gamma) selection by held-out NLL and by BIC — shows the
      recovery result does not depend on hand-picked penalties.
  (B) Rank-transform robustness column — re-runs the headline recovery
      comparison under a distribution-free (Gaussian-copula) transform; the
      FR-tDBN advantage should survive.
  (C) Real-panel (lambda, gamma) selection by held-out NLL — reports the
      data-driven penalties for the d=23 fit (replaces the hand-set values).

  python scripts/run_selection_robustness.py

Writes outputs/selection_synthetic.csv, outputs/rank_robustness.csv,
outputs/selection_real.csv. CPU-only; no network.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.benchmark import run_one
from frtdbn.events import load_event_calendar
from frtdbn.model import FitConfig
from frtdbn.panel import DEFAULT_PANEL, assemble_regime_design
from frtdbn.selection import select_hyperparameters
from frtdbn.splitting import train_test_split_regimes
from frtdbn.synthetic import make_regime_pair, prepare_lagged_design

OUT = Path("outputs")
LAMBDAS = (0.01, 0.03, 0.05, 0.1)
GAMMAS = (0.0, 0.02, 0.05, 0.1)


def _prep(synth, p=1):
    targets, lags = [], []
    for x in synth.X:
        target, lagged = prepare_lagged_design(x, p=p)
        targets.append(target)
        lags.append(lagged)
    return targets, lags


def synthetic_selection(seeds=range(5), d=12, n=140) -> pd.DataFrame:
    """(A) Select (lambda, gamma) on synthetic ground truth, both criteria."""

    base = FitConfig(p=1, solver="admm", lbfgs_max_iter=20, outer_max_iter=5, t_admm=3, h_tol=1e-5, seed=0)
    rows = []
    for seed in seeds:
        synth = make_regime_pair(d=d, p=1, n_per_regime=n, nu=5.0, seed=seed, change_edges=4)
        targets, lags = _prep(synth)
        tr_t, tr_l, te_t, te_l = train_test_split_regimes(targets, lags, test_fraction=0.25)
        for criterion in ("heldout_nll", "bic"):
            sel = select_hyperparameters(
                tr_t, tr_l, te_t, te_l, replace(base, seed=seed),
                lambdas=LAMBDAS, gammas=GAMMAS, criterion=criterion, loss="student_t",
            )
            rows.append({"seed": seed, "criterion": criterion,
                         "lambda": sel["lambda"], "gamma": sel["gamma"], "best_score": sel["best_score"]})
    df = pd.DataFrame(rows)
    print("\n=== (A) Synthetic (lambda, gamma) selection ===")
    for criterion in ("heldout_nll", "bic"):
        sub = df[df["criterion"] == criterion]
        lam_mode = Counter(sub["lambda"]).most_common(1)[0][0]
        gam_mode = Counter(sub["gamma"]).most_common(1)[0][0]
        print(f"  {criterion:<12} modal lambda={lam_mode}  gamma={gam_mode}  "
              f"(gamma>0 in {int((sub['gamma'] > 0).sum())}/{len(sub)} seeds)")
    return df


def rank_robustness(seeds=range(8), d=12, n=140) -> pd.DataFrame:
    """(B) Headline recovery comparison under standardize vs rank transform."""

    methods = {
        "FR-tDBN (admm, student-t, fused)": dict(solver="admm", loss="student_t", gamma=0.04),
        "DYNOTEARS-equiv (lbfgs, gaussian, indep)": dict(solver="lbfgs_smooth", loss="gaussian", gamma=0.0),
    }
    rows = []
    for transform in ("standardize", "rank"):
        for name, cfg in methods.items():
            for seed in seeds:
                r = run_one(seed=seed, n=n, d=d, p=1, transform=transform,
                            lbfgs_max_iter=20, outer_max_iter=5, change_edges=4, **cfg)
                rows.append({"transform": transform, "method": name, "seed": seed,
                             "auroc_change_w": r["auroc_change_w"], "varsort_fit": r["varsort_fit"]})
    df = pd.DataFrame(rows)
    print("\n=== (B) Rank-transform robustness (change-W AUROC, mean +/- se) ===")
    summary = []
    for transform in ("standardize", "rank"):
        line = {"transform": transform}
        for name in methods:
            vals = df[(df["transform"] == transform) & (df["method"] == name)]["auroc_change_w"].to_numpy()
            mean, se = float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))
            line[name] = mean
            print(f"  {transform:<12} {name:<42} {mean:.3f} +/- {se:.3f}")
        gap = line["FR-tDBN (admm, student-t, fused)"] - line["DYNOTEARS-equiv (lbfgs, gaussian, indep)"]
        print(f"  {'':<12} {'>>> FR-tDBN advantage':<42} {gap:+.3f}")
        summary.append(line)
    return df


def real_selection() -> pd.DataFrame:
    """(C) Select (lambda, gamma) on the real d=23 panel by held-out NLL."""

    frames = {s: pd.read_parquet(f"data/raw/yahoo/{s}_1h.parquet") for s in DEFAULT_PANEL}
    events = load_event_calendar("data/events.csv")
    tbr, lbr, names, labels = assemble_regime_design(frames, events)
    tr_t, tr_l, te_t, te_l = train_test_split_regimes(tbr, lbr, test_fraction=0.2)
    base = FitConfig(p=1, solver="admm", lbfgs_max_iter=25, outer_max_iter=8, t_admm=2, h_tol=1e-4, seed=0)

    print(f"\n=== (C) Real-panel (lambda, gamma) selection (d={len(names)}) ===")
    out_rows = []
    for criterion in ("heldout_nll", "bic"):
        sel = select_hyperparameters(
            tr_t, tr_l, te_t, te_l, base,
            lambdas=LAMBDAS, gammas=GAMMAS, criterion=criterion, loss="student_t",
        )
        print(f"  {criterion:<12} selected lambda={sel['lambda']}  gamma={sel['gamma']}  "
              f"best_score={sel['best_score']:.1f}")
        for t in sel["trace"]:
            out_rows.append({"criterion": criterion, **t})
    return pd.DataFrame(out_rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    synthetic_selection().to_csv(OUT / "selection_synthetic.csv", index=False)
    rank_robustness().to_csv(OUT / "rank_robustness.csv", index=False)
    real_selection().to_csv(OUT / "selection_real.csv", index=False)
    print("\nWrote outputs/selection_synthetic.csv, outputs/rank_robustness.csv, outputs/selection_real.csv")


if __name__ == "__main__":
    main()

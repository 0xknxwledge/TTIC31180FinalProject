"""Baseline comparison: FR-tDBN vs DYNOTEARS variants vs W=0 SVAR.

Two parts, written to outputs/baseline_compare.csv:

PART 1 -- Real-data out-of-sample held-out NLL (lower = better).
  Fit each model on the TRAIN split of the regime-labeled panel, then score the
  full-density NLL on the held-out TEST split using train-derived MAD scales
  (shared across all models -> fair, no leakage). Every model is scored under
  BOTH a Student-t(nu=5) and a Gaussian density. Scoring under Student-t favors
  heavy-tailed-fit models; we show the Gaussian column too so the comparison is
  honest.

  Models:
    - FR-tDBN              : ADMM, student_t, fused (gamma_w>0), exact fused prox.
    - DYNOTEARS-per-regime : lbfgs_smooth, gaussian, no fusion, independent graphs.
    - DYNOTEARS-pooled     : lbfgs_smooth, gaussian, no fusion, ONE pooled graph
                             (regime labels ignored: train pooled & duplicated to K=2).
    - SVAR (W=0)           : ridge least-squares lag-only baseline (no contemporaneous DAG).

PART 2 -- Synthetic ground-truth recovery (already computed; read + summarize).
  Reads outputs/headline_nu{5,3}_summary.csv and extracts the change-W AUROC for
  the 2x2x2 of {solver: admm vs lbfgs_smooth} x {loss: student_t vs gaussian} x
  {fusion: fused vs indep}. smooth-L1/gaussian IS the DYNOTEARS-equivalent arm,
  so admm-vs-smooth and student_t-vs-gaussian deltas are the ground-truth evidence.

  python scripts/compare_baselines.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.evaluation import fit_svar_only, full_nll
from frtdbn.events import load_event_calendar
from frtdbn.model import FitConfig, fit_fr_tdbn
from frtdbn.panel import DEFAULT_PANEL, assemble_regime_design
from frtdbn.preprocess import robust_scales
from frtdbn.splitting import train_test_split_regimes

OUT_CSV = Path("outputs/baseline_compare.csv")
NU = 5.0

# FR-tDBN: exact-prox fused ADMM, Student-t loss, fusion on W (gamma_w>0).
FRTDBN_CFG = FitConfig(
    p=1, solver="admm", loss="student_t",
    gamma_w=0.03, gamma_a=0.0, lambda_w=0.05, lambda_a=0.05,
    lbfgs_max_iter=25, outer_max_iter=8, t_admm=2, h_tol=1e-4, seed=0,
)

# DYNOTEARS-equivalent: smooth-L1 acyclicity, Gaussian squared loss, NO fusion.
DYNOTEARS_CFG = FitConfig(
    p=1, solver="lbfgs_smooth", loss="gaussian",
    gamma_w=0.0, gamma_a=0.0, lambda_w=0.05, lambda_a=0.05,
    lbfgs_max_iter=25, outer_max_iter=8, seed=0,
)


def fit_pooled(tr_t, tr_l):
    """DYNOTEARS ignoring regimes: pool train rows, duplicate to K=2, fit one graph.

    The duplicate-as-2-regimes trick (see scripts/make_figures.py) yields W[0]==W[1]
    = the single pooled fit, since both regimes carry identical data and gamma=0.
    """
    pooled_t = np.concatenate(tr_t, axis=0)
    p = len(tr_l[0])
    pooled_l = [np.concatenate([tr_l[k][lag] for k in range(len(tr_l))], axis=0) for lag in range(p)]
    return fit_fr_tdbn([pooled_t, pooled_t], [pooled_l, pooled_l], DYNOTEARS_CFG)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- PART 1: real-data OOS held-out NLL ----------------
    frames = {s: pd.read_parquet(f"data/raw/yahoo/{s}_1h.parquet") for s in DEFAULT_PANEL}
    events = load_event_calendar("data/events.csv")
    tbr, lbr, names, labels = assemble_regime_design(frames, events)
    n_evt = int(labels.sum())
    print(f"panel d={len(names)} | rows={len(labels)} | "
          f"ordinary={len(labels) - n_evt} event={n_evt} ({100 * n_evt / len(labels):.1f}%)")

    tr_t, tr_l, te_t, te_l = train_test_split_regimes(tbr, lbr, test_fraction=0.2)
    print(f"train: ordinary={len(tr_t[0])} event={len(tr_t[1])} | "
          f"test: ordinary={len(te_t[0])} event={len(te_t[1])}")

    # Train-derived MAD scales shared by ALL models (fair, no leakage).
    scales = robust_scales(np.concatenate(tr_t, axis=0))

    print("\n=== fitting models on TRAIN ===")
    fits = {
        "FR-tDBN (admm, student_t, fused)": fit_fr_tdbn(tr_t, tr_l, FRTDBN_CFG),
        "DYNOTEARS-per-regime (smooth, gaussian, indep)": fit_fr_tdbn(tr_t, tr_l, DYNOTEARS_CFG),
        "DYNOTEARS-pooled (smooth, gaussian, 1 graph)": fit_pooled(tr_t, tr_l),
        "SVAR (W=0)": fit_svar_only(tr_t, tr_l),
    }
    for name, fit in fits.items():
        print(f"  fit: {name}")

    # Score full NLL on the held-out TEST split under both densities.
    rows_p1 = []
    for name, fit in fits.items():
        nll_t = full_nll(te_t, te_l, fit, loss="student_t", nu=NU, scales=scales)
        nll_g = full_nll(te_t, te_l, fit, loss="gaussian", scales=scales)
        rows_p1.append({"model": name, "test_nll_studentt": nll_t, "test_nll_gaussian": nll_g})

    print("\n=== PART 1: real-data OOS held-out NLL (lower = better) ===")
    print(f"  {'model':<48} {'studentt':>12} {'gaussian':>12}")
    for r in sorted(rows_p1, key=lambda r: r["test_nll_studentt"]):
        print(f"  {r['model']:<48} {r['test_nll_studentt']:>12.1f} {r['test_nll_gaussian']:>12.1f}")

    # ---------------- PART 2: synthetic ground-truth AUROC ----------------
    rows_p2 = []
    for nu_tag in (5, 3):
        df = pd.read_csv(f"outputs/headline_nu{nu_tag}_summary.csv")
        for _, row in df.iterrows():
            rows_p2.append({
                "nu_data": nu_tag,
                "solver": row["solver"],
                "loss": row["loss"],
                "fusion": row["fusion"],
                "auroc_w_mean": float(row["auroc_w_mean"]),
                "auroc_w_se": float(row["auroc_w_se"]),
            })

    def auroc(nu_tag, solver, loss, fusion):
        for r in rows_p2:
            if (r["nu_data"] == nu_tag and r["solver"] == solver
                    and r["loss"] == loss and r["fusion"] == fusion):
                return r["auroc_w_mean"], r["auroc_w_se"]
        return float("nan"), float("nan")

    print("\n=== PART 2: synthetic change-W AUROC (2x2x2) ===")
    print(f"  {'nu':>3} {'solver':>13} {'loss':>10} {'fusion':>6} {'auroc_w':>9} {'se':>7}")
    for nu_tag in (5, 3):
        for solver in ("admm", "lbfgs_smooth"):
            for loss in ("student_t", "gaussian"):
                for fusion in ("fused", "indep"):
                    m, se = auroc(nu_tag, solver, loss, fusion)
                    print(f"  {nu_tag:>3} {solver:>13} {loss:>10} {fusion:>6} {m:>9.3f} {se:>7.3f}")

    # ---------------- write combined CSV ----------------
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["part", "key", "metric", "value", "value2"])
        # Part 1: model -> (studentt nll, gaussian nll)
        for r in rows_p1:
            w.writerow(["1_real_oos_nll", r["model"], "test_nll_studentt|test_nll_gaussian",
                        f"{r['test_nll_studentt']:.6f}", f"{r['test_nll_gaussian']:.6f}"])
        # Part 2: 2x2x2 synthetic change-W AUROC
        for r in rows_p2:
            key = f"nu{r['nu_data']}|{r['solver']}|{r['loss']}|{r['fusion']}"
            w.writerow(["2_synthetic_auroc_w", key, "auroc_w_mean|auroc_w_se",
                        f"{r['auroc_w_mean']:.6f}", f"{r['auroc_w_se']:.6f}"])
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()

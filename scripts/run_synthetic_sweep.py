"""Broadened synthetic benchmark to harden the recovery claim: FR-tDBN (admm,
student-t, fused) vs DYNOTEARS-equivalent (smooth-L1, gaussian, independent),
swept one axis at a time over change-edge count, sample size, and tail heaviness.

  python scripts/run_synthetic_sweep.py            # full (12 seeds, ~1k fits)
  python scripts/run_synthetic_sweep.py --smoke    # fast sanity
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.benchmark import run_grid  # noqa: E402

D, P = 12, 1
SOLVERS = ("admm", "lbfgs_smooth")
LOSSES = ("student_t", "gaussian")
GAMMAS = (0.0, 0.05)  # independent vs fused
# FR-tDBN = (admm, student_t, fused);  DYNOTEARS-equiv = (lbfgs_smooth, gaussian, indep)


def _mean_se(values):
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
    return statistics.mean(vals), se


def _cell(rows, solver, loss, fused, key="auroc_change_w"):
    return _mean_se([r[key] for r in rows
                     if r["solver"] == solver and r["loss"] == loss and (r["gamma"] > 0) == fused])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seeds = list(range(2 if args.smoke else 12))
    lbfgs, outer = (8, 2) if args.smoke else (25, 6)

    def grid(**kw):
        return run_grid(d=D, p=P, seeds=seeds, solvers=SOLVERS, losses=LOSSES, gammas=GAMMAS,
                        lbfgs_max_iter=lbfgs, outer_max_iter=outer, **kw)

    change_vals = (0, 2) if args.smoke else (0, 1, 2, 3, 5)
    n_vals = [50, 100] if args.smoke else [50, 100, 200]
    nu_vals = (5.0,) if args.smoke else (3.0, 5.0, 30.0)

    rows = []
    for r in grid(n_values=[100], nu=5.0, change_edges_values=change_vals):
        r["axis"], r["axis_value"] = "change_edges", r["change_edges"]
        rows.append(r)
    for r in grid(n_values=n_vals, nu=5.0, change_edges_values=(4,)):
        r["axis"], r["axis_value"] = "n", r["n"]
        rows.append(r)
    for nu in nu_vals:
        for r in grid(n_values=[100], nu=nu, change_edges_values=(4,)):
            r["axis"], r["axis_value"] = "nu", nu
            rows.append(r)

    out = Path("outputs/synthetic_sweep.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}\n")

    # FR-tDBN vs DYNOTEARS-equivalent across each axis
    print(f"{'axis':<13}{'value':<8}{'FR-tDBN AUROC':<18}{'DYNOTEARS AUROC':<18}{'gap':<8}{'FR Δ_W L1':<10}")
    for axis in ("change_edges", "n", "nu"):
        for v in sorted({r["axis_value"] for r in rows if r["axis"] == axis}):
            rs = [r for r in rows if r["axis"] == axis and r["axis_value"] == v]
            fr_m, fr_s = _cell(rs, "admm", "student_t", True)
            dy_m, dy_s = _cell(rs, "lbfgs_smooth", "gaussian", False)
            fr_l1, _ = _cell(rs, "admm", "student_t", True, key="delta_w_l1")
            gap = fr_m - dy_m if not (np.isnan(fr_m) or np.isnan(dy_m)) else float("nan")
            print(f"{axis:<13}{str(v):<8}{f'{fr_m:.3f} ({fr_s:.3f})':<18}"
                  f"{f'{dy_m:.3f} ({dy_s:.3f})':<18}{f'{gap:+.3f}':<8}{fr_l1:<10.3f}")

    # figure: AUROC vs change-edges (the headline robustness panel)
    ce = [r for r in rows if r["axis"] == "change_edges"]
    xs = sorted({r["change_edges"] for r in ce if r["change_edges"] > 0})
    fr = [_cell([r for r in ce if r["change_edges"] == x], "admm", "student_t", True) for x in xs]
    dy = [_cell([r for r in ce if r["change_edges"] == x], "lbfgs_smooth", "gaussian", False) for x in xs]
    fig, axp = plt.subplots(figsize=(7, 5))
    axp.errorbar(xs, [m for m, _ in fr], yerr=[s for _, s in fr], marker="o", capsize=4, label="FR-tDBN (admm, t, fused)")
    axp.errorbar(xs, [m for m, _ in dy], yerr=[s for _, s in dy], marker="s", capsize=4, label="DYNOTEARS-equiv (smooth, gaussian, indep)")
    axp.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
    axp.set_xlabel("# change edges in Δ"); axp.set_ylabel("change-W AUROC (mean ± se)")
    axp.set_title(f"Change-graph recovery vs Δ cardinality (d={D}, n=100, ν=5, {len(seeds)} seeds)")
    axp.legend(); axp.set_ylim(0.4, 1.0)
    figpath = Path("outputs/figures/synthetic_sweep_change_edges.png")
    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {figpath}")


if __name__ == "__main__":
    main()

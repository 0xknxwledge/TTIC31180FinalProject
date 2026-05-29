"""Appendix robustness ablation: pseudo-Huber data term vs Gaussian / Student-t.

Runs the headline synthetic grid (var-sortability-controlled, ADMM exact-prox
solver) with three data terms -- Gaussian, Student-t, and pseudo-Huber -- at the
same settings used for Table 1, so the Gaussian/Student-t cells reproduce the
main table and pseudo-Huber slots in directly. Question: is the change-graph
recovery advantage specific to the Student-t form, or does any robust loss get
most of it?

The pseudo-Huber arm is scored on recovery (change-W AUROC) only; it is a robust
loss, not a normalized density, so it is excluded from the held-out NLL
comparisons by construction.

  python scripts/run_huber_ablation.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.benchmark import run_grid, summarize_grid  # noqa: E402

# Match the Table 1 headline grid exactly (ADMM solver, lambda=0.03, gamma=0.04,
# d=12, 12 seeds; nu=5 uses n in {50,100}, nu=3 uses n in {20,35}).
SETTINGS = [
    {"nu": 5.0, "n_values": [50, 100]},
    {"nu": 3.0, "n_values": [20, 35]},
]
LOSSES = ("gaussian", "student_t", "pseudo_huber", "nig")
SEEDS = list(range(12))
OUT = Path("outputs/huber_ablation.csv")


def main() -> None:
    summary_rows: list[dict] = []
    for s in SETTINGS:
        rows = run_grid(
            d=12,
            p=1,
            n_values=s["n_values"],
            seeds=SEEDS,
            losses=LOSSES,
            gammas=(0.0, 0.04),
            gamma_as=(0.0,),  # fusion on W only (gamma_A=0), matching the headline FR-tDBN
            solvers=("admm",),
            lambda_reg=0.03,
            nu=s["nu"],
            lbfgs_max_iter=20,
            outer_max_iter=5,
        )
        for c in summarize_grid(rows):
            c["nu_data"] = s["nu"]
            summary_rows.append(c)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["nu_data", "solver", "loss", "fusion", "gamma_w", "n_fits",
            "auroc_w_mean", "auroc_w_se", "seconds_mean"]
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {OUT}\n")

    # Pretty table: rows = loss, cols = (nu, fusion) -> change-W AUROC (se).
    def cell(nu, loss, fusion):
        for r in summary_rows:
            if r["nu_data"] == nu and r["loss"] == loss and r["fusion"] == fusion:
                return f"{r['auroc_w_mean']:.3f} ({r['auroc_w_se']:.3f})"
        return "  --  "

    print(f"change-W AUROC, ADMM solver, {len(SEEDS)*2} fits/cell\n")
    print(f"{'loss':>13} | {'nu5 indep':>14} {'nu5 fused':>14} | {'nu3 indep':>14} {'nu3 fused':>14}")
    print("-" * 78)
    for loss in LOSSES:
        print(f"{loss:>13} | {cell(5.0, loss, 'indep'):>14} {cell(5.0, loss, 'fused'):>14} | "
              f"{cell(3.0, loss, 'indep'):>14} {cell(3.0, loss, 'fused'):>14}")


if __name__ == "__main__":
    main()

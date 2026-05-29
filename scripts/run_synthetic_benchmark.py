"""Var-sortability-honest synthetic benchmark: {loss} x {fusion} grid.

Each regime is full-sample standardized before fitting (unless --no-standardize)
so recovery cannot exploit the variance gradient. Reports change-edge AUROC for
both W and A, plus var-sortability (raw vs fitted), acyclicity violation, and
wall-clock per fit.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.benchmark import run_grid, summarize_grid

COLUMNS = [
    "seed", "n", "d", "p", "loss", "solver", "gamma", "standardize",
    "varsort_raw", "varsort_fit", "auroc_change_w", "auroc_change_a",
    "max_h", "seconds",
]

SUMMARY_COLUMNS = [
    "solver", "loss", "fusion", "n_fits",
    "auroc_w_mean", "auroc_w_se", "auroc_a_mean", "auroc_a_se",
    "max_h_mean", "seconds_mean",
]


def _print_summary(rows: list[dict]) -> list[dict]:
    print("\nvar-sortability: raw={:.3f}  fitted={:.3f}  (mean +/- se over {} fits/cell)".format(
        statistics.mean(r["varsort_raw"] for r in rows),
        statistics.mean(r["varsort_fit"] for r in rows),
        max((c["n_fits"] for c in summarize_grid(rows)), default=0),
    ))
    summary = summarize_grid(rows)
    summary.sort(key=lambda c: (c["solver"], c["loss"], c["fusion"]))
    print(f"{'solver':<14}{'loss':<11}{'fusion':<9}{'AUROC W (se)':<18}{'AUROC A (se)':<18}{'max_h':<11}{'sec/fit':<8}")
    for c in summary:
        print("{:<14}{:<11}{:<9}{:<18}{:<18}{:<11.2e}{:<8.2f}".format(
            c["solver"], c["loss"], c["fusion"],
            f"{c['auroc_w_mean']:.3f} ({c['auroc_w_se']:.3f})",
            f"{c['auroc_a_mean']:.3f} ({c['auroc_a_se']:.3f})",
            c["max_h_mean"], c["seconds_mean"],
        ))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=12)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--n", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--nu", type=float, default=5.0)
    parser.add_argument("--lambda-reg", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=0.04)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--outer-iter", type=int, default=5)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--output", default="outputs/synthetic_benchmark.csv")
    args = parser.parse_args()

    rows = run_grid(
        d=args.d,
        p=args.p,
        n_values=args.n,
        seeds=args.seeds,
        gammas=(0.0, args.gamma),
        standardize=not args.no_standardize,
        lambda_reg=args.lambda_reg,
        nu=args.nu,
        lbfgs_max_iter=args.max_iter,
        outer_max_iter=args.outer_iter,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")

    summary = _print_summary(rows)
    summary_path = out.with_name(out.stem + "_summary.csv")
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()

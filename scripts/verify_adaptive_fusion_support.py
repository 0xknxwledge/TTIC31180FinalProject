"""Verify exact-support F1 for the adaptive-fusion negative ablation.

This is intentionally small and aligned to the paper's heavy-tail/small-sample
synthetic setting: d=12, p=1, nu=3, n in {20, 35}, and 12 seeds per n.
It writes per-fit rows plus a grouped summary used to support the paper text.

  python scripts/verify_adaptive_fusion_support.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.model import FitConfig, fit_fr_tdbn
from frtdbn.preprocess import apply_transform
from frtdbn.synthetic import make_regime_pair, prepare_lagged_design


def _support_metrics(true_support: np.ndarray, pred_support: np.ndarray) -> dict[str, float]:
    true = true_support.astype(bool).ravel()
    pred = pred_support.astype(bool).ravel()
    tp = int((true & pred).sum())
    fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "pred_nnz": float(pred.sum()),
    }


def _fit_one(n: int, seed: int, fusion: str, adaptive_pilot: str, args: argparse.Namespace) -> dict:
    synth = make_regime_pair(
        d=args.d,
        p=args.p,
        n_per_regime=n,
        nu=args.nu,
        seed=seed,
        change_edges=args.change_edges,
    )
    targets, lags = [], []
    for x in [apply_transform(xx, "standardize") for xx in synth.X]:
        target, lagged = prepare_lagged_design(x, p=args.p)
        targets.append(target)
        lags.append(lagged)

    cfg = FitConfig(
        p=args.p,
        solver="admm",
        loss="student_t",
        fusion=fusion,
        adaptive_pilot=adaptive_pilot,
        lambda_w=args.lambda_reg,
        lambda_a=args.lambda_reg,
        gamma_w=args.gamma,
        gamma_a=args.gamma,
        lbfgs_max_iter=args.max_iter,
        outer_max_iter=args.outer_iter,
        h_tol=1e-5,
        seed=seed,
    )
    fit = fit_fr_tdbn(targets, lags, cfg)
    pred_support = np.abs(fit.Delta_W) > args.threshold
    row = {
        "n": n,
        "seed": seed,
        "fusion": fusion,
        "adaptive_pilot": adaptive_pilot,
        "delta_w_l1": float(np.abs(fit.Delta_W).sum()),
    }
    row.update(_support_metrics(synth.changed_W, pred_support))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=12)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--n", type=int, nargs="+", default=[20, 35])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--nu", type=float, default=3.0)
    parser.add_argument("--change-edges", type=int, default=4)
    parser.add_argument("--lambda-reg", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=0.08)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--outer-iter", type=int, default=5)
    parser.add_argument("--output", default="outputs/adaptive_fusion_support.csv")
    args = parser.parse_args()

    variants = [
        ("uniform", "uniform"),
        ("adaptive", "uniform"),
        ("adaptive", "independent"),
    ]
    rows = []
    for n in args.n:
        for seed in args.seeds:
            for fusion, pilot in variants:
                rows.append(_fit_one(n, seed, fusion, pilot, args))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    summary = (
        df.groupby(["fusion", "adaptive_pilot"], as_index=False)
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            pred_nnz_mean=("pred_nnz", "mean"),
            pred_nnz_std=("pred_nnz", "std"),
            delta_w_l1_mean=("delta_w_l1", "mean"),
            delta_w_l1_std=("delta_w_l1", "std"),
        )
    )
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path)

    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()

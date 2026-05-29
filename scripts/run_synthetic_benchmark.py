"""Run a small synthetic comparison of independent vs fused FR-tDBN."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.metrics import auroc_binary, change_scores
from frtdbn.model import FitConfig, fit_fr_tdbn
from frtdbn.synthetic import make_regime_pair, prepare_lagged_design


def _fit_one(seed: int, n: int, d: int, p: int, gamma: float, max_iter: int) -> dict[str, float]:
    synth = make_regime_pair(d=d, p=p, n_per_regime=n, seed=seed)
    targets = []
    lags = []
    for x in synth.X:
        target, lagged = prepare_lagged_design(x, p=p)
        targets.append(target)
        lags.append(lagged)
    cfg = FitConfig(
        p=p,
        lambda_w=0.03,
        lambda_a=0.03,
        gamma_w=gamma,
        gamma_a=gamma,
        lbfgs_max_iter=max_iter,
        outer_max_iter=5,
        h_tol=1e-5,
        seed=seed,
    )
    fit = fit_fr_tdbn(targets, lags, cfg)
    return {
        "seed": float(seed),
        "n": float(n),
        "d": float(d),
        "p": float(p),
        "gamma": float(gamma),
        "auroc_change_w": auroc_binary(synth.changed_W, change_scores(fit.W)),
        "max_h": fit.history[-1]["max_h"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=12)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--n", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--output", default="outputs/synthetic_benchmark.csv")
    args = parser.parse_args()

    rows = []
    for n in args.n:
        for seed in args.seeds:
            rows.append(_fit_one(seed, n, args.d, args.p, gamma=0.0, max_iter=args.max_iter))
            rows.append(_fit_one(seed, n, args.d, args.p, gamma=0.04, max_iter=args.max_iter))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

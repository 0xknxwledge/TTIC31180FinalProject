"""Run a small fixed-nu FR-tDBN synthetic smoke benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.metrics import auroc_binary, binary_adjacency, change_scores, shd_binary
from frtdbn.model import FitConfig, fit_fr_tdbn
from frtdbn.synthetic import make_regime_pair, prepare_lagged_design


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=12)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=35)
    args = parser.parse_args()

    synth = make_regime_pair(d=args.d, p=args.p, n_per_regime=args.n, seed=args.seed)
    targets = []
    lags = []
    for x in synth.X:
        x_target, x_lags = prepare_lagged_design(x, p=args.p)
        targets.append(x_target)
        lags.append(x_lags)

    cfg = FitConfig(
        p=args.p,
        lambda_w=0.03,
        lambda_a=0.03,
        gamma_w=0.04,
        gamma_a=0.04,
        lbfgs_max_iter=args.max_iter,
        outer_max_iter=5,
        h_tol=1e-5,
        seed=args.seed,
    )
    fit = fit_fr_tdbn(targets, lags, cfg)

    true_change = synth.changed_W
    pred_change_score = change_scores(fit.W)
    true_W0 = binary_adjacency(synth.W[0], threshold=1e-12)
    pred_W0 = binary_adjacency(fit.W[0], threshold=0.1)
    true_W1 = binary_adjacency(synth.W[1], threshold=1e-12)
    pred_W1 = binary_adjacency(fit.W[1], threshold=0.1)

    print("FR-tDBN synthetic smoke")
    print(f"d={args.d} p={args.p} n_per_regime={args.n} seed={args.seed}")
    print(f"outer_history={fit.history}")
    print(f"SHD(W ordinary)={shd_binary(true_W0, pred_W0)}")
    print(f"SHD(W event)={shd_binary(true_W1, pred_W1)}")
    print(f"AUROC(change W)={auroc_binary(true_change, pred_change_score):.3f}")
    print(f"estimated_nonzero_W={[int(np.sum(binary_adjacency(w, 0.1))) for w in fit.W]}")


if __name__ == "__main__":
    main()

"""Real-data robustness: block-bootstrap stability selection + permutation null.

Reports which change edges (Delta_W) are stable across resamples, and whether the
observed ||Delta_W||_1 exceeds a volatility/block-matched regime-label null.

  python scripts/run_real_robustness.py --n-bootstrap 25 --n-permutations 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.events import load_event_calendar
from frtdbn.model import FitConfig, fit_fr_tdbn
from frtdbn.panel import DEFAULT_PANEL, assemble_regime_design
from frtdbn.robustness import (
    bh_rejected,
    edgewise_permutation_test,
    permutation_null_delta_norm,
    stability_selection,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=25)
    ap.add_argument("--n-permutations", type=int, default=20)
    ap.add_argument("--n-edgewise", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=6)
    ap.add_argument("--gamma-w", type=float, default=0.03)
    ap.add_argument("--freq-threshold", type=float, default=0.7)
    args = ap.parse_args()

    frames = {s: pd.read_parquet(f"data/raw/yahoo/{s}_1h.parquet") for s in DEFAULT_PANEL}
    events = load_event_calendar("data/events.csv")
    tbr, lbr, names, labels = assemble_regime_design(frames, events)
    print(f"panel d={len(names)} | ordinary={int((labels == 0).sum())} event={int(labels.sum())}")

    cfg = FitConfig(
        p=1, solver="admm", loss="student_t",
        lambda_w=0.05, lambda_a=0.05, gamma_w=args.gamma_w, gamma_a=0.0,
        lbfgs_max_iter=20, outer_max_iter=5, t_admm=1, h_tol=1e-4, seed=0,
    )

    d = len(names)
    offdiag = ~np.eye(d, dtype=bool)
    Path("outputs").mkdir(parents=True, exist_ok=True)

    if args.n_bootstrap > 0:
        print(f"\n=== stability selection ({args.n_bootstrap} block bootstraps) ===")
        stab = stability_selection(tbr, lbr, cfg, n_bootstrap=args.n_bootstrap, block_size=24, seed=0)
        freq = stab["delta_w_frequency"]
        n_stable = int((freq[offdiag] >= args.freq_threshold).sum())
        print(f"  edges with freq >= {args.freq_threshold}: {n_stable} (of {int(offdiag.sum())} off-diagonal)")
        print("  top stable change edges (freq):")
        shown = 0
        for flat in np.argsort(freq.ravel())[::-1]:
            i, j = divmod(int(flat), d)
            if i == j:
                continue
            print(f"    {names[i]:>9} -> {names[j]:<9}  freq={freq[i, j]:.2f}")
            shown += 1
            if shown >= 15:
                break

    if args.n_permutations > 0:
        print(f"\n=== global permutation null on ||Delta_W||_1 ({args.n_permutations} perms) ===")
        observed = float(np.abs(fit_fr_tdbn(tbr, lbr, cfg).Delta_W).sum())
        null = permutation_null_delta_norm(tbr, lbr, cfg, n_permutations=args.n_permutations,
                                           seed=0, volatility_match=True, block_size=args.block_size)
        p_value = (1 + int((null >= observed).sum())) / (1 + len(null))
        print(f"  observed={observed:.3f}  null mean={null.mean():.3f} max={null.max():.3f}  p={p_value:.3f}")

    if args.n_edgewise > 0:
        print(f"\n=== edge-wise permutation test ({args.n_edgewise} perms) ===")
        res = edgewise_permutation_test(tbr, lbr, cfg, n_permutations=args.n_edgewise,
                                        seed=0, volatility_match=True, block_size=args.block_size)
        pv, obs = res["pvalues"], res["observed_abs"]
        pv_off = pv[offdiag]
        n_edges = int(offdiag.sum())
        raw_sig = int((pv_off < 0.05).sum())
        n_bh = int(bh_rejected(pv_off, alpha=0.05).sum())
        print(f"  edges p<0.05 (raw): {raw_sig}  vs ~{0.05 * n_edges:.0f} expected by chance  |  BH-FDR survivors: {n_bh}")
        edges = sorted(((i, j) for i in range(d) for j in range(d) if i != j),
                       key=lambda ij: (pv[ij], -obs[ij]))
        print("  lowest-p change edges (p, |Delta|):")
        for i, j in edges[:15]:
            print(f"    {names[i]:>9} -> {names[j]:<9}  p={pv[i, j]:.3f}  |Delta|={obs[i, j]:.3f}")
        rows = [{"source": names[i], "target": names[j], "pvalue": float(pv[i, j]), "abs_delta": float(obs[i, j])}
                for i in range(d) for j in range(d) if i != j]
        pd.DataFrame(rows).sort_values(["pvalue", "abs_delta"], ascending=[True, False]).to_csv(
            "outputs/real_edgewise_pvalues.csv", index=False)
        print("  wrote outputs/real_edgewise_pvalues.csv")


if __name__ == "__main__":
    main()

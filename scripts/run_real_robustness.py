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
from frtdbn.robustness import permutation_null_delta_norm, stability_selection


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=25)
    ap.add_argument("--n-permutations", type=int, default=20)
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

    # --- stability selection ---
    print(f"\n=== stability selection ({args.n_bootstrap} block bootstraps) ===")
    stab = stability_selection(tbr, lbr, cfg, n_bootstrap=args.n_bootstrap, block_size=24, seed=0)
    freq = stab["delta_w_frequency"]
    d = freq.shape[0]
    offdiag = ~np.eye(d, dtype=bool)
    n_stable = int((freq[offdiag] >= args.freq_threshold).sum())
    print(f"  edges with freq >= {args.freq_threshold}: {n_stable} (of {int(offdiag.sum())} off-diagonal)")
    order = np.argsort(freq.ravel())[::-1]
    shown = 0
    print("  top stable change edges (freq):")
    for flat in order:
        i, j = divmod(int(flat), d)
        if i == j:
            continue
        print(f"    {names[i]:>9} -> {names[j]:<9}  freq={freq[i, j]:.2f}")
        shown += 1
        if shown >= 15:
            break

    # --- permutation null on ||Delta_W||_1 ---
    print(f"\n=== permutation null ({args.n_permutations} vol/block-matched perms) ===")
    observed = float(np.abs(fit_fr_tdbn(tbr, lbr, cfg).Delta_W).sum())
    null = permutation_null_delta_norm(
        tbr, lbr, cfg, n_permutations=args.n_permutations, seed=0,
        volatility_match=True, block_size=args.block_size,
    )
    p_value = (1 + int((null >= observed).sum())) / (1 + len(null))
    print(f"  observed ||Delta_W||_1 = {observed:.3f}")
    print(f"  null mean = {null.mean():.3f}  null max = {null.max():.3f}  (n={len(null)})")
    print(f"  smoothed empirical p-value = {p_value:.3f}")

    out = Path("outputs/real_robustness_stability.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"source": names[i], "target": names[j], "delta_w_freq": float(freq[i, j])}
            for i in range(d) for j in range(d) if i != j]
    pd.DataFrame(rows).sort_values("delta_w_freq", ascending=False).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

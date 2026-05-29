"""Lagged (inter-slice) structure analysis at p=1: the 1-hour lead-lag graph A,
how it changes around events (Delta_A), whether that change beats a matched null,
and how the lag-1 lead-lag edges evolve over time.

Lag-1 is the only lag carrying weight (see run_lag_selection.py). Edge i ->(1h) j
means asset i at t-1 predicts asset j at t.

  python scripts/run_lagged_analysis.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.events import label_event_regime, load_event_calendar  # noqa: E402
from frtdbn.model import FitConfig, fit_fr_tdbn  # noqa: E402
from frtdbn.panel import DEFAULT_PANEL, build_lagged_design  # noqa: E402
from frtdbn.robustness import bh_rejected, edgewise_permutation_test, permutation_null_delta_norm  # noqa: E402
from frtdbn.splitting import split_panel_by_regime, time_block_indices  # noqa: E402
from frtdbn.viz import top_edges  # noqa: E402

P = 1
CFG = dict(p=P, solver="admm", loss="student_t", lambda_w=0.05, lambda_a=0.05,
           gamma_w=0.03, gamma_a=0.0, lbfgs_max_iter=25, outer_max_iter=8, t_admm=2, h_tol=1e-4, seed=0)


def main() -> None:
    frames = {s: pd.read_parquet(f"data/raw/yahoo/{s}_1h.parquet") for s in DEFAULT_PANEL}
    events = load_event_calendar("data/events.csv")
    target, lags, names, ts = build_lagged_design(frames, DEFAULT_PANEL, p=P)
    d = len(names)
    labels = label_event_regime(ts, events["event_time_utc"], window_hours=2.0)
    tbr, lbr = split_panel_by_regime(target, lags, labels)

    cfg = FitConfig(**CFG)
    fit = fit_fr_tdbn(tbr, lbr, cfg)
    dA = np.asarray(fit.Delta_A).reshape(P, d, d)[0]   # lag-1 event-minus-ordinary
    A_ord = np.asarray(fit.A[0]).reshape(P, d, d)[0]

    print("Top lag-1 lead-lag edges (ordinary regime), i ->(1h) j:")
    for s, t, w in top_edges(A_ord, names, 10):
        print(f"    {s:>9} ->(1h) {t:<9}  a={w:+.3f}")
    print("\nTop lag-1 lead-lag edges that CHANGE around events (|Delta_A|):")
    for s, t, w in top_edges(dA, names, 12):
        print(f"    {s:>9} ->(1h) {t:<9}  Delta={w:+.3f}")

    # --- significance of the lagged change ---
    obs = float(np.abs(fit.Delta_A).sum())
    null = permutation_null_delta_norm(tbr, lbr, cfg, n_permutations=20, seed=0,
                                       volatility_match=True, block_size=6, delta="A")
    p_glob = (1 + int((null >= obs).sum())) / (1 + len(null))
    print(f"\nGlobal null ||Delta_A||_1: obs={obs:.3f}  null mean={null.mean():.3f}  p={p_glob:.3f}")

    res = edgewise_permutation_test(tbr, lbr, cfg, n_permutations=100, seed=0,
                                    volatility_match=True, block_size=6, delta="A")
    pv = res["pvalues"]
    raw_sig = int((pv < 0.05).sum())
    n_bh = int(bh_rejected(pv, 0.05).sum())
    print(f"edge-wise Delta_A: {raw_sig} edges p<0.05 vs ~{0.05 * pv.size:.0f} by chance | BH survivors: {n_bh}")

    # --- lag-1 lead-lag persistence over time ---
    cfg0 = replace(cfg, gamma_w=0.0, gamma_a=0.0, outer_max_iter=6, t_admm=1)
    blocks = time_block_indices(target.shape[0], 8)
    block_A, block_labels = [], []
    for idx in blocks:
        tb, lb = target[idx], [lag[idx] for lag in lags]
        fb = fit_fr_tdbn([tb, tb], [lb, lb], cfg0)
        block_A.append(np.asarray(fb.A[0]).reshape(P, d, d)[0])
        block_labels.append(f"{pd.Timestamp(ts[idx[0]]).date()}")
    # union of each block's top-6 cross-asset lead-lag edges
    edges = sorted({(names.index(s), names.index(t)) for A in block_A for s, t, _ in top_edges(A, names, 6)})
    mat = np.array([[A[i, j] for A in block_A] for i, j in edges])
    order = np.argsort(-np.abs(mat).mean(axis=1))
    mat, edges = mat[order], [edges[k] for k in order]
    lim = float(np.abs(mat).max()) or 1.0
    fig, ax = plt.subplots(figsize=(11, max(4, 0.32 * len(edges))))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(edges)))
    ax.set_yticklabels([f"{names[i]} ->(1h) {names[j]}" for i, j in edges], fontsize=7)
    ax.set_xticks(range(len(block_labels)))
    ax.set_xticklabels(block_labels, rotation=45, ha="right", fontsize=7)
    ax.set_title("Lag-1 lead-lag edge weight (A) across time blocks")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    out = Path("outputs/figures/lagged_a_persistence.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

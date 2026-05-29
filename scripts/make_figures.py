"""Generate paper figures from the real panel: regime heatmaps, change network,
and contemporaneous structure over time.

  python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.events import label_event_regime, load_event_calendar
from frtdbn.model import FitConfig, fit_fr_tdbn
from frtdbn.panel import DEFAULT_PANEL, build_lagged_design
from frtdbn.splitting import split_panel_by_regime, time_block_indices
from frtdbn.viz import plot_change_network, plot_regime_heatmaps, plot_structure_over_time

FIGDIR = Path("outputs/figures")


def main() -> None:
    frames = {s: pd.read_parquet(f"data/raw/yahoo/{s}_1h.parquet") for s in DEFAULT_PANEL}
    events = load_event_calendar("data/events.csv")
    target, lags, names, ts = build_lagged_design(frames, DEFAULT_PANEL)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    base = FitConfig(p=1, solver="admm", loss="student_t", lambda_w=0.05, lambda_a=0.05,
                     gamma_w=0.03, gamma_a=0.0, lbfgs_max_iter=25, outer_max_iter=8, t_admm=2,
                     h_tol=1e-4, seed=0)

    # --- regime graphs (ordinary / event / Delta) ---
    labels = label_event_regime(ts, events["event_time_utc"], window_hours=2.0)
    tbr, lbr = split_panel_by_regime(target, lags, labels)
    fit = fit_fr_tdbn(tbr, lbr, base)
    plot_regime_heatmaps(fit.W[0], fit.W[1], names, FIGDIR / "regime_heatmaps.png")
    plot_change_network(fit.Delta_W, names, FIGDIR / "change_network.png", k=20)
    print("wrote regime_heatmaps.png, change_network.png")

    # --- contemporaneous structure over time (single-W per contiguous block) ---
    cfg0 = replace(base, gamma_w=0.0, gamma_a=0.0, outer_max_iter=6)
    W_blocks, titles = [], []
    for idx in time_block_indices(target.shape[0], n_blocks=6):
        tb, lb = target[idx], [lag[idx] for lag in lags]
        fb = fit_fr_tdbn([tb, tb], [lb, lb], cfg0)  # duplicate-as-2-regimes -> single W
        W_blocks.append(fb.W[0])
        titles.append(f"{pd.Timestamp(ts[idx[0]]).date()} .. {pd.Timestamp(ts[idx[-1]]).date()}")
    plot_structure_over_time(W_blocks, titles, names, FIGDIR / "structure_over_time.png")
    print("wrote structure_over_time.png")


if __name__ == "__main__":
    main()

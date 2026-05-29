"""Lag-order selection on the real panel: does adding lags (1h..6h) help OOS, and
how much weight does each lag carry? Guides p for the lagged-edge analysis.

  python scripts/run_lag_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.evaluation import full_nll
from frtdbn.events import load_event_calendar
from frtdbn.model import FitConfig, fit_fr_tdbn
from frtdbn.panel import DEFAULT_PANEL, assemble_regime_design
from frtdbn.preprocess import robust_scales
from frtdbn.splitting import train_test_split_regimes


def main() -> None:
    frames = {s: pd.read_parquet(f"data/raw/yahoo/{s}_1h.parquet") for s in DEFAULT_PANEL}
    events = load_event_calendar("data/events.csv")

    print(f"{'p':<4}{'A_params':<10}{'OOS NLL(t)':<13}{'h_max':<10}{'||W||1':<9}{'per-lag ||A_lag||1 (ordinary)'}")
    for p in [1, 2, 3, 6]:
        tbr, lbr, names, labels = assemble_regime_design(frames, events, p=p)
        tr_t, tr_l, te_t, te_l = train_test_split_regimes(tbr, lbr, test_fraction=0.2)
        scales = robust_scales(np.concatenate(tr_t, axis=0))
        cfg = FitConfig(p=p, solver="admm", loss="student_t", lambda_w=0.05, lambda_a=0.05,
                        gamma_w=0.03, gamma_a=0.0, lbfgs_max_iter=20, outer_max_iter=5,
                        t_admm=1, h_tol=1e-4, seed=0)
        fit = fit_fr_tdbn(tr_t, tr_l, cfg)
        nll = full_nll(te_t, te_l, fit, loss="student_t", nu=5.0, scales=scales)
        d = len(names)
        a0 = np.asarray(fit.A[0]).reshape(p, d, d)
        per_lag = [round(float(np.abs(a0[lag]).sum()), 2) for lag in range(p)]
        wnorm = float(np.abs(fit.W[0]).sum())
        print(f"{p:<4}{2 * (p + 1) * d * d:<10}{nll:<13.1f}{fit.diagnostics['h_returned_max']:<10.2e}"
              f"{wnorm:<9.2f}{per_lag}")


if __name__ == "__main__":
    main()

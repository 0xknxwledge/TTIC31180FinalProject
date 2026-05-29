"""End-to-end real-data run: assemble regime design -> OOS W=0 pre-check ->
first FR-tDBN fit. Composes the (separately tested) library functions.

  python scripts/run_real_panel.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.evaluation import svar_vs_dag_oos
from frtdbn.events import load_event_calendar
from frtdbn.model import FitConfig, fit_fr_tdbn
from frtdbn.panel import DEFAULT_PANEL, assemble_regime_design
from frtdbn.splitting import train_test_split_regimes


def main() -> None:
    frames = {s: pd.read_parquet(f"data/raw/yahoo/{s}_1h.parquet") for s in DEFAULT_PANEL}
    events = load_event_calendar("data/events.csv")
    tbr, lbr, names, labels = assemble_regime_design(frames, events)
    n_evt = int(labels.sum())
    print(f"panel d={len(names)} | rows={len(labels)} | "
          f"ordinary={len(labels) - n_evt} event={n_evt} ({100 * n_evt / len(labels):.1f}%)")

    tr_t, tr_l, te_t, te_l = train_test_split_regimes(tbr, lbr, test_fraction=0.2)
    print(f"train: ordinary={len(tr_t[0])} event={len(tr_t[1])} | "
          f"test: ordinary={len(te_t[0])} event={len(te_t[1])}")

    base = FitConfig(p=1, solver="admm", lambda_w=0.05, lambda_a=0.05, gamma_w=0.03, gamma_a=0.0,
                     lbfgs_max_iter=25, outer_max_iter=8, t_admm=2, h_tol=1e-4, seed=0)

    print("\n=== OOS W=0 pre-check (does the contemporaneous DAG pay its way?) ===")
    for loss in ("gaussian", "student_t"):
        res = svar_vs_dag_oos(tr_t, tr_l, te_t, te_l, replace(base, loss=loss), loss=loss)
        verdict = "DAG better" if res["dag_better"] else "SVAR better"
        print(f"  {loss:<10} dag={res['dag_test_nll']:.1f} svar={res['svar_test_nll']:.1f} "
              f"delta={res['dag_minus_svar']:+.1f} -> {verdict}")

    print("\n=== first FR-tDBN fit (student-t, fused, ADMM) on train; top |Delta_W| edges ===")
    fit = fit_fr_tdbn(tr_t, tr_l, replace(base, loss="student_t"))
    dw = fit.Delta_W
    print(f"  h_returned_max={fit.diagnostics['h_returned_max']:.2e}  Delta_W nnz={int((np.abs(dw) > 1e-8).sum())}")
    order = np.argsort(np.abs(dw).ravel())[::-1]
    shown = 0
    for flat in order:
        i, j = divmod(int(flat), dw.shape[0])
        if i == j or abs(dw[i, j]) < 1e-8:
            continue
        print(f"    {names[i]:>9} -> {names[j]:<9}  Delta={dw[i, j]:+.3f}")
        shown += 1
        if shown >= 12:
            break


if __name__ == "__main__":
    main()

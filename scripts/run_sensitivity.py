"""Sensitivity sweep on the macro-event regime null for ||Delta_W||_1.

The global d=23 panel result is NULL (permutation p~0.81): macro events do not
restructure the dependency graph in aggregate. This script asks whether that null
is *robust* by sweeping (a) per-event-type regimes (FOMC / CPI / NFP) and
(b) the event-window width (pooled events: 1h / 2h / 4h). For each setting it
relabels the SAME lagged design, splits into [ordinary, event], fits the FR-tDBN,
records the observed ||Delta_W||_1, and compares it to a volatility/block-matched
permutation null. Writes outputs/sensitivity.csv.

  python scripts/run_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.events import label_event_regime, load_event_calendar
from frtdbn.model import FitConfig, fit_fr_tdbn
from frtdbn.panel import DEFAULT_PANEL, build_lagged_design
from frtdbn.robustness import permutation_null_delta_norm
from frtdbn.splitting import split_panel_by_regime

N_PERMUTATIONS = 20
BLOCK_SIZE = 6
MIN_EVENT_ROWS = 10

CONFIG = FitConfig(
    p=1, solver="admm", loss="student_t",
    lambda_w=0.05, lambda_a=0.05, gamma_w=0.03, gamma_a=0.0,
    lbfgs_max_iter=20, outer_max_iter=5, t_admm=1, h_tol=1e-4, seed=0,
)


def evaluate(label: str, target, lags, labels):
    """Split -> fit -> observed ||Delta_W||_1 -> matched permutation-null p-value."""

    n_event = int(np.asarray(labels).sum())
    if n_event < MIN_EVENT_ROWS:
        print(f"  [SKIP] {label}: n_event={n_event} < {MIN_EVENT_ROWS}")
        return {
            "setting": label, "n_event_rows": n_event, "observed_norm": np.nan,
            "null_mean": np.nan, "p_value": np.nan, "status": "skipped_few_event_rows",
        }

    tbr, lbr = split_panel_by_regime(target, lags, labels)
    observed = float(np.abs(fit_fr_tdbn(tbr, lbr, CONFIG).Delta_W).sum())
    null = permutation_null_delta_norm(
        tbr, lbr, CONFIG, n_permutations=N_PERMUTATIONS,
        seed=0, volatility_match=True, block_size=BLOCK_SIZE,
    )
    p_value = (1 + int((null >= observed).sum())) / (1 + len(null))
    null_mean = float(null.mean())
    print(f"  {label:<22} n_event={n_event:<5} observed={observed:.3f}  "
          f"null_mean={null_mean:.3f}  p={p_value:.3f}")
    return {
        "setting": label, "n_event_rows": n_event, "observed_norm": observed,
        "null_mean": null_mean, "p_value": p_value, "status": "ok",
    }


def main() -> None:
    frames = {s: pd.read_parquet(f"data/raw/yahoo/{s}_1h.parquet") for s in DEFAULT_PANEL}
    target, lags, names, ts = build_lagged_design(frames, DEFAULT_PANEL)
    print(f"panel d={len(names)} | target rows={len(target)}")

    rows = []

    # (a) Per-event-type, window_hours=2.0 (only that type's windows are the event regime).
    print("\n=== (a) per-event-type (window=2.0h) ===")
    for etype in ["FOMC", "CPI", "NFP"]:
        events = load_event_calendar("data/events.csv", event_types=[etype])
        labels = label_event_regime(ts, events["event_time_utc"], window_hours=2.0)
        rows.append(evaluate(f"type={etype}", target, lags, labels))

    # (b) Window width, pooled across all events.
    print("\n=== (b) window width (pooled all events) ===")
    events_all = load_event_calendar("data/events.csv")
    for wh in [1.0, 2.0, 4.0]:
        labels = label_event_regime(ts, events_all["event_time_utc"], window_hours=wh)
        rows.append(evaluate(f"window={wh:g}h", target, lags, labels))

    out = pd.DataFrame(rows)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    out_path = "outputs/sensitivity.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    ok = out[out["status"] == "ok"]
    rejecters = ok[ok["p_value"] < 0.10]
    print("\n=== SUMMARY (p<0.10 rejects the null) ===")
    print(out.to_string(index=False))
    if rejecters.empty:
        print("\nCONCLUSION: null is ROBUST -- no event type or window has p<0.10.")
    else:
        labs = ", ".join(f"{r.setting} (p={r.p_value:.3f})" for r in rejecters.itertuples())
        print(f"\nCONCLUSION: null REJECTED by: {labs}")


if __name__ == "__main__":
    main()

"""Macro event calendar and regime labeling.

The calendar is hand-curated as `date,time_et,event_type` (release times are
published in ET); `load_event_calendar` converts to **tz-naive UTC** (DST-aware
via America/New_York) to match the panel. `label_event_regime` tags panel bars
within +/- a window of any release as the event regime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_event_calendar(path: str | Path, event_types: list[str] | None = None) -> pd.DataFrame:
    """Load `date,time_et,event_type` into `event_time_utc` (tz-naive UTC) + type."""

    raw = pd.read_csv(path, comment="#")
    et = pd.to_datetime(raw["date"].astype(str).str.strip() + " " + raw["time_et"].astype(str).str.strip())
    utc = (
        et.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    out = pd.DataFrame({"event_time_utc": utc, "event_type": raw["event_type"].astype(str).str.strip()})
    if event_types is not None:
        out = out[out["event_type"].isin(event_types)]
    return out.dropna(subset=["event_time_utc"]).sort_values("event_time_utc").reset_index(drop=True)


def label_event_regime(timestamps, event_times, window_hours: float = 2.0) -> np.ndarray:
    """Return 0/1 labels (1 = event) for bars within +/- window_hours of an event."""

    ts = pd.DatetimeIndex(timestamps).asi8
    ev = pd.DatetimeIndex(pd.to_datetime(list(event_times))).asi8
    labels = np.zeros(len(ts), dtype=int)
    if ev.size == 0 or ts.size == 0:
        return labels
    window_ns = int(window_hours * 3_600 * 1_000_000_000)
    nearest = np.abs(ts[:, None] - ev[None, :]).min(axis=1)
    return (nearest <= window_ns).astype(int)

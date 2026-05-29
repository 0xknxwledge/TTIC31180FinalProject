"""Assemble a regime-labeled lagged design from cached OHLCV frames.

Composes the (separately tested) data / preprocess / events / splitting steps so
the real-data scripts share one pipeline: returns -> rolling past-only z-score ->
lagged design -> event-regime labels -> [ordinary, event] partition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from frtdbn.data import build_return_panel
from frtdbn.events import label_event_regime
from frtdbn.preprocess import rolling_zscore_past
from frtdbn.splitting import split_panel_by_regime
from frtdbn.synthetic import prepare_lagged_design

# Canonical d=23 crypto-macro panel (CNY=X dropped: sparse US-RTH coverage;
# ^TNX dropped: redundant with IEF + grid-limiting :20 stamp).
DEFAULT_PANEL = [
    "SPY", "QQQ", "IWM", "NVDA", "COIN", "MSTR",          # equity / crypto-equity
    "TLT", "IEF", "SHY", "HYG",                            # rates / credit
    "GLD", "SLV", "USO",                                   # commodities
    "DX-Y.NYB", "EURUSD=X", "GBPUSD=X", "JPY=X",          # FX
    "^VIX",                                                # vol
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LINK-USD",  # crypto
]


def assemble_regime_design(
    frames_by_symbol: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    panel_symbols: list[str] | None = None,
    p: int = 1,
    zscore_window: int = 250,
    min_periods: int = 60,
    event_window_h: float = 2.0,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], list[str], np.ndarray]:
    """Return (targets_by_regime, lags_by_regime, column_names, labels).

    `events` must have an `event_time_utc` column. Regimes are [ordinary, event].
    """

    if panel_symbols is not None:
        frames_by_symbol = {s: frames_by_symbol[s] for s in panel_symbols}
    panel = build_return_panel(frames_by_symbol)
    names = list(panel.columns)

    z = rolling_zscore_past(panel.to_numpy(), window=zscore_window, min_periods=min_periods)
    keep = np.isfinite(z).all(axis=1)
    z, ts = z[keep], panel.index[keep]

    target, lags = prepare_lagged_design(z, p=p)
    ts_target = ts[p:]
    labels = label_event_regime(ts_target, events["event_time_utc"], window_hours=event_window_h)
    targets_by_regime, lags_by_regime = split_panel_by_regime(target, lags, labels)
    return targets_by_regime, lags_by_regime, names, labels

import numpy as np
import pandas as pd

from frtdbn.data import build_return_panel
from frtdbn.panel import assemble_regime_design


def _frame(ts, closes):
    return pd.DataFrame(
        {"timestamp": ts, "open": closes, "high": closes, "low": closes,
         "close": closes, "volume": 0.0}
    )


def test_build_return_panel_aligns_and_log_returns():
    ts = pd.date_range("2025-01-01 14:00", periods=4, freq="h")
    a = _frame(ts, [100.0, 110.0, 121.0, 133.1])   # +10% per bar
    b = _frame(ts, [50.0, 55.0, 60.5, 66.55])
    panel = build_return_panel({"A": a, "B": b})

    assert list(panel.columns) == ["A", "B"]
    assert len(panel) == 3                          # first (NaN) return row dropped
    assert np.isclose(panel.iloc[0]["A"], np.log(1.1))
    assert np.isclose(panel.iloc[0]["B"], np.log(1.1))
    assert panel.index.is_monotonic_increasing


def test_build_return_panel_aligns_subhour_offsets_to_hourly_grid():
    # Yahoo stamps equities at :30 and crypto/FX/VIX at :00 — must align to the hour
    eq = pd.date_range("2025-01-01 14:30", periods=4, freq="h")   # 14:30..17:30
    cr = pd.date_range("2025-01-01 14:00", periods=5, freq="h")   # 14:00..18:00
    a = _frame(eq, [100.0, 110.0, 121.0, 133.1])
    b = _frame(cr, [50.0, 55.0, 60.5, 66.55, 73.205])
    panel = build_return_panel({"EQ": a, "CR": b})
    assert len(panel) >= 2
    assert not panel.isna().any().any()


def test_assemble_regime_design_produces_two_aligned_regimes():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2025-01-02 14:00", periods=40, freq="h")
    frames = {}
    for sym in ["AAA", "BBB", "CCC"]:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=40)))
        frames[sym] = pd.DataFrame(
            {"timestamp": ts, "open": prices, "high": prices, "low": prices,
             "close": prices, "volume": 0.0}
        )
    events = pd.DataFrame({"event_time_utc": [pd.Timestamp("2025-01-03 12:00")]})

    tbr, lbr, names, labels = assemble_regime_design(
        frames, events, p=1, zscore_window=6, min_periods=3, event_window_h=2.0
    )

    assert names == ["AAA", "BBB", "CCC"]
    assert len(tbr) == 2 and len(lbr) == 2
    assert tbr[0].shape[1] == 3 and lbr[0][0].shape[1] == 3
    assert labels.shape[0] == tbr[0].shape[0] + tbr[1].shape[0]  # all rows partitioned
    assert tbr[1].shape[0] >= 1                                   # event regime non-empty


def test_build_return_panel_keeps_only_common_grid():
    ts_a = pd.date_range("2025-01-01 14:00", periods=4, freq="h")     # 14,15,16,17
    ts_b = pd.date_range("2025-01-01 15:00", periods=4, freq="h")     # 15,16,17,18
    a = _frame(ts_a, [100.0, 101.0, 102.0, 103.0])
    b = _frame(ts_b, [10.0, 11.0, 12.0, 13.0])
    panel = build_return_panel({"A": a, "B": b})

    # overlap timestamps are 15,16,17; returns need a prior bar, so rows kept = 16,17
    assert set(panel.columns) == {"A", "B"}
    assert (panel.index >= pd.Timestamp("2025-01-01 16:00")).all()
    assert not panel.isna().any().any()             # no NaNs in the returned panel

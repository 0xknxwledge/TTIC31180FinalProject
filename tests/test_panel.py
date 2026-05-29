import numpy as np
import pandas as pd

from frtdbn.data import build_return_panel


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

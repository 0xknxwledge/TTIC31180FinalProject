import numpy as np
import pandas as pd

from frtdbn.data import close_to_return_panel


def test_close_to_return_panel_builds_wide_log_returns():
    ts = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    frames = [
        pd.DataFrame({"timestamp": ts, "symbol": "BTC/USD", "close": [100.0, 110.0, 121.0]}),
        pd.DataFrame({"timestamp": ts, "symbol": "ETH/USD", "close": [50.0, 55.0, 55.0]}),
    ]

    panel = close_to_return_panel(frames)

    assert panel.shape == (3, 2)
    assert np.isnan(panel.loc[ts[0], "BTC/USD"])
    assert np.isclose(panel.loc[ts[1], "BTC/USD"], np.log(1.1))
    assert np.isclose(panel.loc[ts[2], "ETH/USD"], 0.0)

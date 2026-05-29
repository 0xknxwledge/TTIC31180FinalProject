import pandas as pd
import pytest

from frtdbn.data import _retry, _yahoo_session, normalize_yahoo_hourly

OUT = ["timestamp", "open", "high", "low", "close", "volume"]


def _raw_multiindex():
    idx = pd.DatetimeIndex(
        ["2025-01-02 14:00", "2025-01-02 15:00"], tz="UTC", name="Datetime"
    )
    cols = pd.MultiIndex.from_tuples(
        [("Open", "^VIX"), ("High", "^VIX"), ("Low", "^VIX"),
         ("Close", "^VIX"), ("Adj Close", "^VIX"), ("Volume", "^VIX")]
    )
    return pd.DataFrame(
        [[10, 11, 9, 10.5, 10.5, 0], [10.5, 12, 10, 11.5, 11.5, 0]],
        index=idx, columns=cols,
    )


def test_normalize_flattens_columns_and_unwraps_utc():
    out = normalize_yahoo_hourly(_raw_multiindex())
    assert list(out.columns) == OUT
    assert out["timestamp"].dt.tz is None                          # tz-naive UTC
    assert out["timestamp"].iloc[0] == pd.Timestamp("2025-01-02 14:00")
    assert out["close"].iloc[1] == 11.5
    assert len(out) == 2


def test_normalize_handles_flat_columns():
    idx = pd.DatetimeIndex(["2025-01-02 14:00"], tz="UTC", name="Datetime")
    raw = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5], "Volume": [0]},
        index=idx,
    )
    out = normalize_yahoo_hourly(raw)
    assert list(out.columns) == OUT
    assert out["close"].iloc[0] == 10.5


def test_normalize_handles_empty():
    raw = pd.DataFrame(columns=pd.MultiIndex.from_tuples([("Close", "^VIX")]))
    out = normalize_yahoo_hourly(raw)
    assert out.empty
    assert list(out.columns) == OUT


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}
    delays: list[float] = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Too Many Requests")
        return "ok"

    out = _retry(flaky, retries=5, base_delay=1.0, sleep=delays.append)
    assert out == "ok"
    assert calls["n"] == 3
    assert delays == [1.0, 2.0]  # exponential backoff between the two failures


def test_retry_raises_after_exhausting():
    def always():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        _retry(always, retries=2, base_delay=0.0, sleep=lambda s: None)


def test_yahoo_session_never_raises():
    # returns a curl_cffi session if installed, else None — but must not raise
    _yahoo_session()

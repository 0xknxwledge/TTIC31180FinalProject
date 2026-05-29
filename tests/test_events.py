import pandas as pd

from frtdbn.events import label_event_regime, load_event_calendar


def test_load_event_calendar_converts_et_to_utc_with_dst(tmp_path):
    p = tmp_path / "events.csv"
    p.write_text("date,time_et,event_type\n2024-07-11,08:30,CPI\n2024-01-11,08:30,CPI\n")
    cal = load_event_calendar(p)
    assert list(cal.columns) == ["event_time_utc", "event_type"]
    times = set(cal["event_time_utc"])
    assert pd.Timestamp("2024-07-11 12:30") in times   # EDT (UTC-4): 08:30 -> 12:30
    assert pd.Timestamp("2024-01-11 13:30") in times   # EST (UTC-5): 08:30 -> 13:30
    assert cal["event_time_utc"].dt.tz is None          # tz-naive UTC, matches the panel


def test_load_event_calendar_filters_event_types(tmp_path):
    p = tmp_path / "events.csv"
    p.write_text("date,time_et,event_type\n2025-03-12,08:30,CPI\n2025-03-19,14:00,FOMC\n")
    cal = load_event_calendar(p, event_types=["CPI"])
    assert cal["event_type"].tolist() == ["CPI"]


def test_label_event_regime_tags_within_window():
    idx = pd.date_range("2025-03-12 10:00", periods=8, freq="h")  # 10..17 UTC
    events = pd.to_datetime(["2025-03-12 13:30"])
    labels = label_event_regime(idx, events, window_hours=2.0)
    # within [11:30, 15:30] -> bars 12,13,14,15
    assert labels.tolist() == [0, 0, 1, 1, 1, 1, 0, 0]


def test_label_event_regime_no_events_is_all_ordinary():
    idx = pd.date_range("2025-03-12 10:00", periods=4, freq="h")
    labels = label_event_regime(idx, pd.to_datetime([]), window_hours=2.0)
    assert labels.tolist() == [0, 0, 0, 0]

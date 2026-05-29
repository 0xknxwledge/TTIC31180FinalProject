import pandas as pd

from frtdbn.data import audit_symbol_coverage, find_stooq_files, load_stooq_txt

HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _write(path, rows):
    path.write_text(HEADER + "".join(rows))


def test_load_stooq_txt_parses_timestamp_and_close(tmp_path):
    p = tmp_path / "spy.us.txt"
    _write(p, [
        "SPY.US,60,20210104,143000,100,101,99,100.5,1000,0\n",
        "SPY.US,60,20210104,153000,100.5,102,100,101.5,1200,0\n",
    ])
    df = load_stooq_txt(p)
    assert list(df.columns) == COLUMNS
    assert len(df) == 2
    assert df["close"].iloc[0] == 100.5
    assert df["timestamp"].iloc[0] == pd.Timestamp("2021-01-04 14:30:00")


def test_load_stooq_txt_handles_header_only(tmp_path):
    p = tmp_path / "empty.us.txt"
    p.write_text(HEADER)
    df = load_stooq_txt(p)
    assert df.empty
    assert list(df.columns) == COLUMNS


def test_find_stooq_files_matches_us_and_world_tickers(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "spy.us.txt").write_text(HEADER)
    (tmp_path / "a" / "b" / "^vix.txt").write_text(HEADER)
    assert [p.name for p in find_stooq_files(tmp_path, "spy")] == ["spy.us.txt"]
    assert [p.name for p in find_stooq_files(tmp_path, "^vix")] == ["^vix.txt"]
    assert find_stooq_files(tmp_path, "nope") == []


def test_audit_symbol_coverage_reports_span_and_bad_closes(tmp_path):
    p = tmp_path / "x.us.txt"
    _write(p, [
        "X.US,60,20210104,143000,1,1,1,10,5,0\n",
        "X.US,60,20210104,153000,1,1,1,0,5,0\n",     # nonpositive close
        "X.US,60,20210105,143000,1,1,1,11,5,0\n",
    ])
    audit = audit_symbol_coverage(load_stooq_txt(p), "X")
    assert audit["symbol"] == "X"
    assert audit["n_rows"] == 3
    assert audit["n_days"] == 2
    assert audit["close_nonpositive"] == 1
    assert audit["first_ts"] == "2021-01-04 14:30:00"

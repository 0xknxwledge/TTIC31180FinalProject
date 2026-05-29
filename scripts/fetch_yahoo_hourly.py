"""Fetch + cache + audit Yahoo Finance hourly bars (needs network; run locally).

Yahoo serves ~730 days of hourly intraday history, which matches the usable Stooq
window (2024-05 → 2026-05). Primary use: source hourly `^VIX` (absent from the
Stooq dump). Yahoo timestamps are explicit UTC, so this also anchors the Stooq
time zone.

Example (note the quotes around ^VIX):
  python scripts/fetch_yahoo_hourly.py --symbols '^VIX' --period 730d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.data import audit_symbol_coverage, fetch_yahoo_hourly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["^VIX"])
    parser.add_argument("--period", default="730d")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--cache-dir", default="data/raw/yahoo")
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    print(f"{'symbol':<10}{'rows':<8}{'first (UTC)':<21}{'last (UTC)':<21}{'days':<7}{'hours'}")
    for symbol in args.symbols:
        try:
            frame = fetch_yahoo_hourly(symbol, period=args.period, start=args.start, end=args.end)
        except Exception as exc:  # network / yahoo errors surface here
            print(f"{symbol:<10}FAILED: {exc}")
            continue
        if frame.empty:
            print(f"{symbol:<10}0       (no rows returned)")
            continue
        path = cache / f"{symbol.replace('/', '_').replace(':', '_')}_1h.parquet"
        frame.to_parquet(path, index=False)
        a = audit_symbol_coverage(frame, symbol)
        print("{:<10}{:<8}{:<21}{:<21}{:<7}{}".format(
            symbol, a["n_rows"], str(a["first_ts"])[:19], str(a["last_ts"])[:19],
            a["n_days"], a["hours"],
        ))
        print(f"           cached -> {path}")


if __name__ == "__main__":
    main()

"""Audit hourly coverage of candidate panel symbols in the local Stooq dump.

First pipeline deliverable: answers "what is actually available hourly, over what
span, with what session/timezone" empirically, before we commit to a panel.

Example:
  python scripts/audit_data_coverage.py \
    --symbols spy qqq tlt ief gld shy ^vix \
    --output outputs/coverage_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.data import audit_symbol_coverage, find_stooq_files, load_stooq_txt

DEFAULT_ROOTS = ["data/stooq_index_etf", "data/stooq_bonds_crypto_FX"]
COLUMNS = [
    "symbol", "n_files", "n_rows", "first_ts", "last_ts", "span_days", "n_days",
    "rows_per_day_median", "n_distinct_hours", "close_nulls", "close_nonpositive", "hours",
]


def audit_one(symbol: str, roots: list[str], start: str | None, end: str | None) -> dict:
    files: list[Path] = []
    for root in roots:
        if Path(root).exists():
            files.extend(find_stooq_files(root, symbol))
    if not files:
        return {"symbol": symbol, "n_files": 0, "n_rows": 0, "first_ts": None, "last_ts": None,
                "span_days": 0, "n_days": 0, "rows_per_day_median": 0.0, "n_distinct_hours": 0,
                "close_nulls": 0, "close_nonpositive": 0, "hours": "[]"}
    frame = pd.concat([load_stooq_txt(f) for f in files], ignore_index=True)
    if start is not None:
        frame = frame[frame["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["timestamp"] < pd.Timestamp(end)]
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
    audit = audit_symbol_coverage(frame, symbol)
    audit["n_files"] = len(files)
    audit["hours"] = str(audit["hours"])
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="outputs/coverage_audit.csv")
    args = parser.parse_args()

    rows = [audit_one(s, args.roots, args.start, args.end) for s in args.symbols]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows([{k: r.get(k) for k in COLUMNS} for r in rows])

    print(f"{'symbol':<10}{'files':<6}{'rows':<8}{'first':<21}{'last':<21}{'days':<7}{'bad_close':<10}{'hours'}")
    for r in rows:
        print("{:<10}{:<6}{:<8}{:<21}{:<21}{:<7}{:<10}{}".format(
            r["symbol"], r["n_files"], r["n_rows"], str(r["first_ts"])[:19], str(r["last_ts"])[:19],
            r["n_days"], r["close_nonpositive"], r["hours"],
        ))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

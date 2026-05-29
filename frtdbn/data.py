"""Data-ingestion helpers for the crypto-macro DBN panel."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CORE_CRYPTO_PERPS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
    "DOGE/USDT:USDT",
    "SOL/USDT:USDT",
    "LINK/USDT:USDT",
    "LTC/USDT:USDT",
    "BCH/USDT:USDT",
]

CORE_CRYPTO_SPOT = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "ADA/USD",
    "DOGE/USD",
    "LINK/USD",
    "LTC/USD",
    "BCH/USD",
]


def safe_symbol_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def fetch_ccxt_ohlcv(
    exchange_id: str,
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    timeframe: str = "1h",
    market_type: str | None = None,
    limit: int = 1000,
    pause_seconds: float = 0.05,
) -> pd.DataFrame:
    """Fetch OHLCV bars through a ccxt exchange adapter."""

    import ccxt  # type: ignore

    options = {"defaultType": market_type} if market_type is not None else {}
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True, "options": options})
    start_ts = _as_utc_timestamp(start)
    end_ts = _as_utc_timestamp(end)
    since = int(start_ts.timestamp() * 1000)
    until = int(end_ts.timestamp() * 1000)

    rows: list[list[float]] = []
    while since < until:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        next_since = int(batch[-1][0]) + 1
        if next_since <= since:
            break
        since = next_since
        if int(batch[-1][0]) >= until:
            break
        time.sleep(pause_seconds)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])

    df = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df.pop("timestamp_ms"), unit="ms", utc=True)
    df = df[df["timestamp"].between(start_ts, end_ts, inclusive="left")]
    df["symbol"] = symbol
    return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]].drop_duplicates(
        ["timestamp", "symbol"]
    )


def fetch_binance_perp_ohlcv(
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    timeframe: str = "1h",
    limit: int = 1000,
    pause_seconds: float = 0.05,
) -> pd.DataFrame:
    """Fetch Binance USDT-margined perpetual OHLCV bars through ccxt."""

    return fetch_ccxt_ohlcv(
        "binance",
        symbol,
        start=start,
        end=end,
        timeframe=timeframe,
        market_type="future",
        limit=limit,
        pause_seconds=pause_seconds,
    )


def cache_ccxt_ohlcv(
    exchange_id: str,
    symbols: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: str | Path,
    timeframe: str = "1h",
    market_type: str | None = None,
) -> dict[str, Path]:
    """Fetch and cache one OHLCV parquet file per ccxt symbol."""

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for symbol in symbols:
        path = out_dir / f"{safe_symbol_name(symbol)}_{timeframe}.parquet"
        existing = pd.read_parquet(path) if path.exists() else None
        fetched = fetch_ccxt_ohlcv(
            exchange_id,
            symbol,
            start=start,
            end=end,
            timeframe=timeframe,
            market_type=market_type,
        )
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, fetched], ignore_index=True)
        else:
            combined = fetched
        combined = combined.drop_duplicates(["timestamp", "symbol"]).sort_values("timestamp")
        combined.to_parquet(path, index=False)
        written[symbol] = path
    return written


def cache_crypto_ohlcv(
    symbols: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: str | Path = "data/raw/binance_perps",
    timeframe: str = "1h",
) -> dict[str, Path]:
    """Fetch and cache one OHLCV parquet file per symbol."""

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for symbol in symbols:
        path = out_dir / f"{safe_symbol_name(symbol)}_{timeframe}.parquet"
        existing = pd.read_parquet(path) if path.exists() else None
        fetched = fetch_binance_perp_ohlcv(symbol, start=start, end=end, timeframe=timeframe)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, fetched], ignore_index=True)
        else:
            combined = fetched
        combined = combined.drop_duplicates(["timestamp", "symbol"]).sort_values("timestamp")
        combined.to_parquet(path, index=False)
        written[symbol] = path
    return written


def _as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


STOOQ_OUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_stooq_txt(path: str | Path) -> pd.DataFrame:
    """Load one Stooq hourly `.txt` file into a tidy OHLCV frame.

    Stooq format: `<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,
    <OPENINT>`, DATE=YYYYMMDD, TIME=HHMMSS. Timestamps are returned **tz-naive**;
    the Stooq time zone is not self-describing, so infer it from the hour
    histogram (see `audit_symbol_coverage`) before aligning to event times.
    """

    path = Path(path)
    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame(columns=STOOQ_OUT_COLUMNS)
    raw.columns = [c.strip("<>").lower() for c in raw.columns]
    date = raw["date"].astype(int).astype(str)
    time = raw["time"].astype(int).astype(str).str.zfill(6)
    ts = pd.to_datetime(date + time, format="%Y%m%d%H%M%S", errors="coerce")
    out = pd.DataFrame(
        {
            "timestamp": ts,
            "open": raw["open"].astype(float),
            "high": raw["high"].astype(float),
            "low": raw["low"].astype(float),
            "close": raw["close"].astype(float),
            "volume": raw["vol"].astype(float),
        }
    )
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def find_stooq_files(root: str | Path, ticker: str) -> list[Path]:
    """Find Stooq files for a ticker (matches `<ticker>.txt` or `<ticker>.us.txt`)."""

    wanted = {f"{ticker.lower()}.txt", f"{ticker.lower()}.us.txt"}
    return sorted(p for p in Path(root).rglob("*.txt") if p.name.lower() in wanted)


def audit_symbol_coverage(df: pd.DataFrame, symbol: str) -> dict:
    """Coverage diagnostics for one symbol's hourly frame (tz-agnostic)."""

    if df.empty:
        return {"symbol": symbol, "n_rows": 0, "first_ts": None, "last_ts": None,
                "span_days": 0, "n_days": 0, "rows_per_day_median": 0.0,
                "n_distinct_hours": 0, "hours": [], "close_nulls": 0, "close_nonpositive": 0}
    ts = df["timestamp"]
    close = df["close"]
    days = ts.dt.normalize()
    hours = sorted(int(h) for h in ts.dt.hour.unique())
    return {
        "symbol": symbol,
        "n_rows": int(len(df)),
        "first_ts": str(ts.min()),
        "last_ts": str(ts.max()),
        "span_days": int((ts.max() - ts.min()).days),
        "n_days": int(days.nunique()),
        "rows_per_day_median": float(df.groupby(days).size().median()),
        "n_distinct_hours": len(hours),
        "hours": hours,
        "close_nulls": int(close.isna().sum()),
        "close_nonpositive": int((close <= 0).sum()),
    }


def close_to_return_panel(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Convert cached OHLCV frames to a wide hourly log-return panel."""

    stacked = pd.concat(list(frames), ignore_index=True)
    prices = (
        stacked.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
        .sort_index()
        .astype(float)
    )
    return np.log(prices).diff()

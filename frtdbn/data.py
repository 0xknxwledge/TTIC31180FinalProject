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
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def normalize_yahoo_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a `yfinance` hourly download to the tidy OHLCV schema.

    Handles yfinance's MultiIndex `(field, ticker)` columns and its tz-aware
    (UTC) DatetimeIndex. Timestamps are returned **tz-naive UTC** to match the
    rest of the pipeline; unlike Stooq, Yahoo's index is an explicit UTC anchor,
    so it can also be used to calibrate the Stooq time zone.
    """

    if raw is None or raw.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    ts = pd.to_datetime(df.iloc[:, 0])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    cols = {str(c).lower(): c for c in df.columns}
    out = pd.DataFrame(
        {
            "timestamp": ts.to_numpy(),
            "open": df[cols["open"]].astype(float).to_numpy(),
            "high": df[cols["high"]].astype(float).to_numpy(),
            "low": df[cols["low"]].astype(float).to_numpy(),
            "close": df[cols["close"]].astype(float).to_numpy(),
            "volume": (df[cols["volume"]].astype(float).to_numpy() if "volume" in cols else 0.0),
        }
    )
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _retry(fn, retries: int = 4, base_delay: float = 3.0, max_delay: float = 60.0, sleep=time.sleep):
    """Call `fn`, retrying on any exception with exponential backoff + cap."""

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # transient network / Yahoo 429s
            last_exc = exc
            if attempt == retries:
                break
            sleep(min(base_delay * (2 ** attempt), max_delay))
    assert last_exc is not None
    raise last_exc


def _yahoo_session():
    """A curl_cffi browser-impersonating session if available, else None.

    Yahoo's crumb/cookie handshake (and thus the 429 rate-limit) is reliably
    satisfied by a real browser TLS fingerprint, which `curl_cffi` provides.
    """

    try:
        from curl_cffi import requests as cffi_requests

        return cffi_requests.Session(impersonate="chrome")
    except Exception:
        return None


def fetch_yahoo_hourly(
    ticker: str,
    period: str = "730d",
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = False,
    session=None,
    retries: int = 4,
    base_delay: float = 3.0,
) -> pd.DataFrame:
    """Fetch hourly bars from Yahoo Finance (needs network). ~730d intraday cap.

    Uses a curl_cffi browser-impersonating session (install `curl_cffi`) plus
    exponential backoff to get past Yahoo's rate limiter. An empty response is
    treated as a (likely rate-limited) failure and retried. Pass `period` OR
    `start`/`end`. Returns the tidy OHLCV schema.
    """

    import yfinance as yf  # imported lazily so the module loads without network deps

    if session is None:
        session = _yahoo_session()
    kwargs = {"interval": "1h", "auto_adjust": auto_adjust, "progress": False}
    if start is not None or end is not None:
        kwargs.update({"start": start, "end": end})
    else:
        kwargs["period"] = period

    def _download() -> pd.DataFrame:
        if session is not None:
            try:
                raw = yf.download(ticker, session=session, **kwargs)
            except TypeError:  # yfinance build manages curl_cffi internally
                raw = yf.download(ticker, **kwargs)
        else:
            raw = yf.download(ticker, **kwargs)
        if raw is None or raw.empty:
            raise RuntimeError(f"empty/rate-limited Yahoo response for {ticker!r}")
        return raw

    return normalize_yahoo_hourly(_retry(_download, retries=retries, base_delay=base_delay))


def cache_yahoo_hourly(
    symbols: Iterable[str],
    cache_dir: str | Path = "data/raw/yahoo",
    period: str = "730d",
    start: str | None = None,
    end: str | None = None,
    pause_seconds: float = 2.0,
) -> dict[str, Path]:
    """Fetch and cache one parquet per Yahoo symbol, throttled between symbols."""

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _yahoo_session()  # reuse one session across symbols
    written: dict[str, Path] = {}
    for i, symbol in enumerate(symbols):
        if i > 0:
            time.sleep(pause_seconds)
        frame = fetch_yahoo_hourly(symbol, period=period, start=start, end=end, session=session)
        path = out_dir / f"{safe_symbol_name(symbol)}_1h.parquet"
        frame.to_parquet(path, index=False)
        written[symbol] = path
    return written


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


def build_return_panel(
    frames_by_symbol: dict[str, pd.DataFrame],
    method: str = "log",
) -> pd.DataFrame:
    """Assemble a wide return panel from per-symbol OHLCV frames.

    Closes are aligned on the union of timestamps (outer join); returns are
    computed per column, then rows with any missing return are dropped — which
    naturally restricts the panel to the common trading grid (e.g. equity RTH,
    where 24/7 crypto is also active). Feed `timestamp`-bearing frames already on
    a single time zone (e.g. all UTC).
    """

    closes: dict[str, pd.Series] = {}
    for symbol, df in frames_by_symbol.items():
        # Yahoo stamps asset classes at different minute offsets (equities :30,
        # crypto/FX :00, ^TNX :20); floor to the hour so they share one grid.
        idx = pd.to_datetime(df["timestamp"]).dt.floor("h")
        s = pd.Series(df["close"].astype(float).to_numpy(), index=idx)
        closes[symbol] = s[~s.index.duplicated(keep="last")].sort_index()
    prices = pd.DataFrame(closes).sort_index().sort_index(axis=1)
    rets = np.log(prices).diff() if method == "log" else prices.pct_change()
    return rets.dropna(how="any")


def close_to_return_panel(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Convert cached OHLCV frames to a wide hourly log-return panel."""

    stacked = pd.concat(list(frames), ignore_index=True)
    prices = (
        stacked.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
        .sort_index()
        .astype(float)
    )
    return np.log(prices).diff()

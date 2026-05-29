"""Code carried over from qts-final-project/mst_strategy.py.

Only the genuinely reusable bits — data ingestion + caching, rolling correlations,
the Mantegna correlation→distance metric (useful for the optional HRP dessert),
and the performance-metrics helper — are kept. The MST / Laplacian / effective-
resistance / Kruskal / pair-weight code is intentionally NOT carried over: MST
is undirected and unweighted-tree-only, which is the wrong object for a DAG
structure-learning project.

Treat this file as a placeholder; rename and split into modules
(`data.py`, `eval.py`, etc.) as the project crystallizes.

What is here:
  - fetch_etf_prices              : Nasdaq Data Link QUOTEMEDIA fetch
  - load_or_fetch_prices          : per-ticker CSV cache layer
  - compute_rolling_correlations  : rolling Pearson corr matrices
  - mantegna_distance             : d_ij = sqrt(2(1-rho)) — used by HRP
  - compute_performance_metrics   : Sharpe/Sortino/MDD/VaR — for HRP dessert

What is NOT here (deliberately dropped):
  - build_mst_kruskal, compute_graph_laplacian, compute_effective_resistance,
    compute_fiedler_value, generate_trading_signals, compute_pair_weights,
    simulate_strategy, run_weekly_mst_pipeline.

Likely additions in week 1 that do NOT live here yet:
  - hourly bar fetchers for Binance perps (klines + funding endpoints)
  - hourly bar fetcher for Polygon equities / ETFs / single names
  - rates / FX / commodities fetcher (FRED or Polygon)
  - RTH-alignment + per-regime label join (event-window labels from CPI / NFP /
    FOMC release calendars)
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# nasdaqdatalink + dotenv imports kept inside fetch_etf_prices so the module
# imports cleanly even if those packages are not installed yet in this project.


def fetch_etf_prices(
    tickers: Sequence[str],
    start_date,
    end_date,
) -> pd.DataFrame:
    """Fetch adjusted close prices from NASDAQ Data Link QUOTEMEDIA/PRICES.

    REUSE NOTES: works for equity tickers / sector ETFs. Will need to be
    paired with a separate fetcher for crypto perps (Binance) and rates / FX
    (Polygon or FRED). Daily granularity only — see TODO at bottom of file
    about an hourly variant.
    """
    import nasdaqdatalink  # type: ignore
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
    api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")
    if api_key is None:
        raise ValueError("NASDAQ_DATA_LINK_API_KEY not found in environment.")
    setattr(nasdaqdatalink.ApiConfig, "api_key", api_key)

    start_str = pd.to_datetime(start_date, errors="raise").date().isoformat()
    end_str = pd.to_datetime(end_date, errors="raise").date().isoformat()

    collected: List[pd.DataFrame] = []
    for ticker in tickers:
        df = nasdaqdatalink.get_table(
            "QUOTEMEDIA/PRICES",
            ticker=ticker,
            date={"gte": start_str, "lte": end_str},
            paginate=True,
        )
        if df.empty:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        small = df[["date", "ticker", price_col]].rename(columns={price_col: "close"})
        collected.append(small)

    if not collected:
        raise ValueError("No data fetched from NASDAQ Data Link for requested tickers and date range.")

    stacked = pd.concat(collected, ignore_index=True)
    prices = (
        stacked.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        .sort_index()
        .sort_index(axis=1)
    )
    return prices


def load_or_fetch_prices(
    tickers: Sequence[str],
    start_date,
    end_date,
    data_dir: str,
) -> pd.DataFrame:
    """Load cached CSVs from data_dir; fetch and cache missing tickers.

    REUSE NOTES: the per-ticker-CSV cache pattern is exactly what we want
    for the new project — fetch is slow, refits are frequent. Will extend
    to a per-ticker Parquet cache once we move to hourly bars.
    """
    os.makedirs(data_dir, exist_ok=True)
    start_ts = pd.Timestamp(pd.to_datetime(start_date, errors="raise"))
    end_ts = pd.Timestamp(pd.to_datetime(end_date, errors="raise"))

    cached_frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for ticker in tickers:
        cache_path = os.path.join(data_dir, f"{ticker}.csv")
        if os.path.exists(cache_path):
            cached = pd.read_csv(cache_path, parse_dates=["date"])
            cached = cached[(cached["date"] >= start_ts) & (cached["date"] <= end_ts)]
            if not cached.empty:
                cached_frames.append(cached.assign(ticker=ticker))
                continue
        missing.append(ticker)

    if missing:
        fetched = fetch_etf_prices(missing, start_date, end_date)
        for ticker in missing:
            if ticker not in fetched.columns:
                continue
            single = fetched[[ticker]].dropna().reset_index().rename(columns={ticker: "close"})
            single["ticker"] = ticker
            single.to_csv(os.path.join(data_dir, f"{ticker}.csv"), index=False)
            cached_frames.append(single)

    if not cached_frames:
        raise ValueError("No cached or fetched data available.")

    pooled = pd.concat(cached_frames, ignore_index=True)
    prices = (
        pooled.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        .sort_index()
        .sort_index(axis=1)
    )
    prices = prices[(prices.index >= start_ts) & (prices.index <= end_ts)]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    return prices


def compute_rolling_correlations(
    returns: pd.DataFrame,
    window: int = 252,
) -> dict[object, pd.DataFrame]:
    """Compute rolling Pearson correlation matrix for each date.

    REUSE NOTES: used for descriptive plots and the HRP dessert. NOT used
    inside the FR-tDBN fit itself (that operates on raw residuals via a
    Student-t likelihood). Will likely re-key by Timestamp not str.
    """
    clean = returns.dropna(how="all")
    out: dict[object, pd.DataFrame] = {}
    for idx in range(window - 1, len(clean)):
        date_raw = pd.to_datetime(clean.index[idx], errors="coerce")
        if pd.isna(date_raw):
            continue
        date = str(date_raw)
        sample = clean.iloc[idx - window + 1 : idx + 1]
        out[date] = sample.corr()
    return out


def mantegna_distance(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute Mantegna distance: d_ij = sqrt(2 * (1 - rho_ij)).

    REUSE NOTES: this is exactly the HRP distance metric (López de Prado 2016)
    up to a factor of 2 inside the sqrt — HRP uses sqrt((1-rho)/2). Will need
    to either rescale or just use this function with a note. Used only by
    the optional HRP-on-structured-covariance figure.
    """
    rho = correlation_matrix.clip(-1.0, 1.0)
    dist = np.sqrt(2.0 * (1.0 - rho))
    np.fill_diagonal(dist.values, 0.0)
    return dist


def compute_performance_metrics(
    returns: pd.Series,
    rf_rate: float = 0.02,
    trade_count: Optional[int] = None,
    average_holding_period: Optional[float] = None,
) -> pd.Series:
    """Compute strategy performance and risk metrics.

    REUSE NOTES: used only by the optional HRP dessert. Trade-count and
    holding-period args carry over from MST pair-trading and can be ignored
    (or removed) when this is repurposed for a long-only HRP backtest.
    """
    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("Return series is empty.")

    ann_factor = 252.0
    total_periods = len(r)
    cum_return = float((1.0 + r).prod() - 1.0)
    ann_return = float((1.0 + cum_return) ** (ann_factor / total_periods) - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(ann_factor))

    rf_daily = rf_rate / ann_factor
    excess = r - rf_daily
    sharpe = float(excess.mean() / r.std(ddof=1) * np.sqrt(ann_factor)) if ann_vol > 0 else np.nan

    downside = r[r < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(ann_factor)) if len(downside) > 1 else np.nan
    sortino = float((excess.mean() * ann_factor) / downside_vol) if downside_vol and downside_vol > 0 else np.nan

    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    var_95 = float(np.quantile(r, 0.05))
    cvar_95 = float(r[r <= var_95].mean())
    win_rate = float((r > 0).mean())

    metrics = pd.Series(
        {
            "Cumulative Return": cum_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Maximum Drawdown": max_drawdown,
            "VaR 95%": var_95,
            "CVaR 95%": cvar_95,
            "Win Rate": win_rate,
            "Number of Trades": float(trade_count) if trade_count is not None else np.nan,
            "Average Holding Period": float(average_holding_period)
            if average_holding_period is not None
            else np.nan,
        }
    )
    return metrics


# -----------------------------------------------------------------------------
# TODO (week 1) — new fetchers / utilities that DO NOT exist in the QTS repo:
#
#   - fetch_binance_perp_klines(symbol, interval='1h', start, end) -> DataFrame
#   - fetch_binance_funding(symbol, start, end) -> DataFrame
#   - fetch_polygon_aggs(ticker, multiplier=1, timespan='hour', start, end)
#   - fetch_fred_series(series_id, start, end)
#   - align_to_rth_grid(panel, tz='America/New_York')
#   - build_event_window_labels(release_calendar, hours_before, hours_after)
#   - per-regime z-score / rank-transform helpers (for var-sortability defense)
# -----------------------------------------------------------------------------

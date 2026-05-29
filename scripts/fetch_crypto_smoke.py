"""Fetch a tiny Binance perp sample to validate the crypto data path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frtdbn.data import CORE_CRYPTO_SPOT, cache_ccxt_ohlcv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="coinbase")
    parser.add_argument("--market-type", default=None)
    parser.add_argument("--start", default="2026-05-01")
    parser.add_argument("--end", default="2026-05-03")
    parser.add_argument("--symbols", nargs="*", default=CORE_CRYPTO_SPOT[:2])
    parser.add_argument("--cache-dir", default="data/raw/crypto_spot")
    args = parser.parse_args()

    written = cache_ccxt_ohlcv(
        args.exchange,
        args.symbols,
        args.start,
        args.end,
        cache_dir=args.cache_dir,
        market_type=args.market_type,
    )
    for symbol, path in written.items():
        print(f"{symbol}: {path}")


if __name__ == "__main__":
    main()

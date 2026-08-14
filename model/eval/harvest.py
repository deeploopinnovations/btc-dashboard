"""
eval/harvest.py
=====================================================================
Build hourly history bundles for assets NOCTUA has never seen.

WHY THIS RUNS IN CI AND NOT LOCALLY

Every exchange API is unreachable from the development container -- the egress
proxy returns 403 on bitstamp.net and api.exchange.coinbase.com alike. That is
a property of THIS container, not of the internet: the repository's own
fetch-data cron has been pulling Bitstamp successfully every hour for days
from a GitHub runner. So the runner is the data channel. This script is meant
to be executed by .github/workflows/harvest-assets.yml, which commits what it
produces; the evaluation then reads the committed bundles offline.

No proxy is bypassed and no restriction is worked around -- the fetch simply
happens where the network already permits it.

WHY THESE ASSETS

The test that matters is ZERO-SHOT TRANSFER. NOCTUA was trained exclusively on
Bitcoin, and the synthetic battery already showed it recovers sigma on
processes it has never seen. Real altcoins are a harder and more honest test
than synthetic paths: they have genuine microstructure, real jumps, real
illiquidity, and excursion distributions that are systematically fatter than
Bitcoin's. If the model has learned first-passage structure it should transfer;
if it has memorised Bitcoin's specific excursion shape it will be miscalibrated
in a direction the benchmark can name.

USD pairs throughout. A USDT pair would fold stablecoin depeg risk into the
excursion distribution and confound the comparison.

    python -m model.eval.harvest --symbols eth,sol,xrp,ltc --days 400
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.fetch import SYMBOLS, fetch_bitstamp, fetch_coinbase, validate_bars  # noqa: E402
from serve.history import hours_from_bars                                       # noqa: E402

OUT_DIR = Path("data/assets")
CHUNK_HOURS = 800          # ~1000 five-minute bars, Bitstamp's per-call ceiling


def harvest_symbol(symbol: str, days: int, verbose: bool = True) -> pd.DataFrame:
    """Page backwards through a long history, one venue, oldest-first result."""
    total_hours = days * 24
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for fn in (fetch_bitstamp, fetch_coinbase):
        frames.clear()
        errors.clear()
        try:
            remaining = total_hours
            while remaining > 0:
                take = min(CHUNK_HOURS, remaining)
                # `tail_hours` counts back from now, so ask for an increasing
                # window and keep the oldest slice of each response.
                got = fn(tail_hours=int(total_hours - remaining + take), symbol=symbol)
                frames.append(got)
                remaining -= take
                if verbose:
                    print(f"    {symbol}: {total_hours - remaining}/{total_hours}h", flush=True)
                time.sleep(0.3)
            break
        except Exception as e:                                   # noqa: BLE001
            errors.append(f"{fn.__name__}: {type(e).__name__}: {e}")
            continue

    if not frames:
        raise RuntimeError(f"{symbol}: every venue failed -> {' | '.join(errors)}")

    bars = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("timestamp")
            .sort_values("timestamp", ignore_index=True))
    bars = validate_bars(bars, tail_hours=24)
    hours = hours_from_bars(bars)
    if verbose:
        span = (hours.hour_ts.max() - hours.hour_ts.min()) / 86400
        print(f"  {symbol}: {len(bars):,} bars -> {len(hours):,} hours "
              f"({span:.0f} days), source={bars.source.iloc[0]}")
    return hours


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Harvest hourly bundles for other assets")
    p.add_argument("--symbols", default="eth,sol,xrp,ltc")
    p.add_argument("--days", type=int, default=400)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)

    syms = [s.strip().lower() for s in a.symbols.split(",") if s.strip()]
    unknown = [s for s in syms if s not in SYMBOLS]
    if unknown:
        print(f"unknown symbols: {unknown}; known: {sorted(SYMBOLS)}")
        return 2

    a.out_dir.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []
    for s in syms:
        print(f"[harvest] {s} ...", flush=True)
        try:
            hours = harvest_symbol(s, a.days)
        except Exception as e:                                   # noqa: BLE001
            # One dead pair must not lose the assets that did work.
            print(f"  {s}: FAILED -- {e}")
            failed.append(s)
            continue
        path = a.out_dir / f"{s}_history.parquet"
        hours.to_parquet(path, index=False, compression="zstd")
        print(f"  {s}: wrote {path} ({path.stat().st_size/1024:.0f} KB)")
        ok.append(s)

    print(f"\nharvested: {ok}")
    if failed:
        print(f"failed:    {failed}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

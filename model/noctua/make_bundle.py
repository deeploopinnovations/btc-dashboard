"""
noctua/make_bundle.py
=====================================================================
Seed `data/noctua_history.parquet` from the training data.

The served model needs 365 days of hourly history for `reg_rv_vs_year` (and
100 days for `mom_dist_ma100`, 90 for `mom_drawdown_90d`). Fetching that from
a public API every 30 minutes would be ~105 paginated requests; instead the
history ships with the repo and the cron only tops up the tail.

Crucially the bundle is built by the SAME `build_hourly` used in training, so
the served features are constructed identically to the trained ones rather than
merely similarly.

    python -m noctua.make_bundle --artifacts model/artifacts
    python -m noctua.make_bundle --artifacts model/artifacts --backfill

--backfill EXTENDS the committed bundle backwards instead of rebuilding it: it
keeps every committed row (including the hours fetched live after the corpus
ends) and prepends corpus hours older than the bundle's first hour, up to
BUNDLE_DAYS. It refuses unless the corpus and the bundle agree bit for bit on
their overlap -- the corpus's own last hour excepted, which is partial there and
was completed by the live feed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.history import (BUNDLE_DAYS, HOURLY_COLS, check_continuity,  # noqa: E402
                           default_bundle_path, save_bundle)


def backfilled(corpus: pd.DataFrame, bundle: pd.DataFrame) -> pd.DataFrame:
    """Corpus hours older than the bundle, then every committed bundle row."""
    import numpy as np
    bundle = bundle[HOURLY_COLS].sort_values("hour_ts", ignore_index=True)
    m = bundle.merge(corpus, on="hour_ts", suffixes=("_b", "_c"))
    m = m[m.hour_ts < corpus.hour_ts.iloc[-1]]        # the corpus's partial last hour
    bad = [c for c in HOURLY_COLS[1:]
           if not np.array_equal(m[c + "_b"].to_numpy(), m[c + "_c"].to_numpy(), equal_nan=True)]
    if len(m) == 0 or bad:
        raise SystemExit(f"REFUSING: corpus and bundle disagree on {len(m)} overlapping "
                         f"hours in {bad or 'no overlap at all'}; not the same construction")
    older = corpus[corpus.hour_ts < bundle.hour_ts.iloc[0]]
    out = pd.concat([older, bundle], ignore_index=True)
    if not check_continuity(out.tail(len(bundle) + 24 * 60))["contiguous"]:
        raise SystemExit("REFUSING: the join between corpus and bundle is not contiguous")
    print(f"[make_bundle] backfill: {len(m)} overlapping hours identical; "
          f"prepending up to {len(older)} corpus hours")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the served history bundle")
    p.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    p.add_argument("--days", type=int, default=BUNDLE_DAYS)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--backfill", action="store_true",
                   help="extend the committed bundle backwards from the corpus")
    a = p.parse_args(argv)

    hourly = a.artifacts / "btcusd_1h.parquet"
    if not hourly.exists():
        raise SystemExit(
            f"{hourly} not found -- run `python -m noctua.episodes` first"
        )

    hours = pd.read_parquet(hourly)[HOURLY_COLS].sort_values("hour_ts", ignore_index=True)
    if a.backfill:
        hours = backfilled(hours, pd.read_parquet(a.out or default_bundle_path()))
    path = save_bundle(hours, a.out, days=a.days)

    written = pd.read_parquet(path)
    info = {
        "path": str(path),
        "size_kb": round(path.stat().st_size / 1024, 1),
        "rows": int(len(written)),
        "start_utc": str(pd.to_datetime(written.hour_ts.iloc[0], unit="s", utc=True)),
        "end_utc": str(pd.to_datetime(written.hour_ts.iloc[-1], unit="s", utc=True)),
        **check_continuity(written),
    }
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

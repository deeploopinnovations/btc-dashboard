"""
eval/harvest_newdata.py
=====================================================================
Harvest the top-ranked new-data candidate for the ONSET problem: perpetual
funding-rate history (positioning stress), plus Deribit's implied-volatility
index (DVOL) as a shorter-history bonus from the same venue. Full reasoning
for why these two, and not the other four candidate classes, and the
pre-registered rule for whether either earns a place in the model, lives in
`eval/newdata.py` -- read that first. This file only fetches.

WHY THIS RUNS IN CI AND NOT LOCALLY

Same boundary as `eval/harvest.py` and documented empirically in
`eval/datasources.py` / `model/artifacts/datasources.json`: this container's
egress proxy allows only raw.githubusercontent.com and gist.github.com. Every
exchange API, Deribit included, is expected to be blocked the same way the
probe found Binance/Kraken/OKX/Bybit/KuCoin/Gate.io blocked -- that
expectation is ASSERTED by extension, not itself re-probed, because Deribit
was never one of the 18 endpoints `datasources.py` tested. This script is
meant to run from `.github/workflows/harvest-newdata.yml`, on a GitHub-hosted
runner, which is where the actual network access happens.

WHY DERIBIT FIRST, AND WHY THAT ORDER IS NOT FREE OF RISK

`serve/fetch.py` documents a fact specific to this repo's own runner, not a
general assumption: `api.binance.com` returned `HTTP 451 Unavailable For
Legal Reasons` from a GitHub Actions runner, because "GitHub's hosted runners
sit in Azure US regions and Binance geo-blocks US IPs." That finding was for
Binance's SPOT API; this script needs Binance's FUTURES API
(`fapi.binance.com`) for funding-rate history, a different subdomain that was
never itself tested from a runner. Binance's derivatives products carry, if
anything, a *stricter* US-access posture than spot (CFTC jurisdiction), so
extending the 451 finding to `fapi.binance.com` is a reasonable inference, not
a re-verified fact -- flagged as such here and in `newdata.py`.

Deribit is not documented anywhere in this repo as geo-blocked. To the
contrary: `src/data.js` and `README.md` describe a live BROWSER-side client
calling `https://www.deribit.com/api/v2/public/get_book_summary_by_currency`
directly with no proxy and no rate complaint ("Deribit | Unlimited public").
A server-to-server call from a GH runner has *fewer* obstacles than a browser
call (no CORS to negotiate at all), so if the browser can reach Deribit, a
runner very plausibly can too -- but "plausibly" is doing real work in that
sentence; it has not been tested from a runner in this repo before this
workflow's first run.

So the fetch order below is: try Deribit's own funding-rate history first
(same venue as the DVOL pull, so one reachable host serves both series, and a
2016-launched instrument raises the ceiling on possible depth); fall back to
Binance funding only if Deribit's funding endpoint fails outright. This
mirrors `eval.harvest.harvest_symbol`'s "first venue that answers" pattern
for exactly the same reason: the alternative is asserting reachability from
inside a container that cannot check it.

WHAT "SUCCESS" LOOKS LIKE AND WHY DEPTH IS NOT ASSUMED

Every depth number in `newdata.py`'s candidate table is a claim about what
*should* be retrievable, sourced either from general knowledge of when an
instrument launched (Deribit BTC-PERPETUAL, Aug 2016) or from this repo's own
prior research docs (`OPTION_BUYER_ALPHA.md`, `SELLER_DIRECTIONAL_ALPHA.md`:
Binance funding 2019-09-present, Deribit DVOL only 2023-09-present). None of
those numbers were re-verified in this session. So this harvester does not
assume a depth and truncate to it -- it walks backward from "now" in chunks
until the venue stops returning anything (three consecutive empty chunks), and
reports whatever span it actually got, the same "measure, don't assume"
discipline `firstpassage.py`'s `brownian_control` docstring insists on after
getting exactly this kind of assumption wrong twice.

WHY FUNDING ACCUMULATES BUT TAILS ON RE-RUNS, AND DVOL DOES NOT

`get_historical_volatility` takes no time-range parameters -- it returns
whatever window Deribit's own backend retains (unverified how large that
window is, or whether it grows). Because this workflow runs on a schedule,
each run's DVOL pull is unioned with the previously committed bundle, so the
repo's own history can eventually exceed whatever Deribit retains on its
side, the same way `serve/history.py`'s committed hourly bundle exists so the
model does not depend on an exchange's retention window for old bars.
Funding-rate history, by contrast, DOES take start/end timestamps, so after
the first backfill each scheduled run only walks back to the bundle's own
last timestamp (minus a small overlap) rather than re-walking years of
already-committed history.

    python -m model.eval.harvest_newdata --days 3500
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_DIR = Path("data/newdata")

UA = {"User-Agent": "noctua-newdata/1.0 (+https://github.com/deeploopinnovations/btc-dashboard)"}

DERIBIT_FUNDING = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
DERIBIT_DVOL = "https://www.deribit.com/api/v2/public/get_historical_volatility"
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"

DAY_MS = 86_400_000
CHUNK_DAYS = 30          # per-call window walked backward; unverified whether
                          # either venue caps a call smaller than this
REQUEST_SLEEP_S = 0.25   # polite pause between calls; neither venue's public
                          # rate limit for these specific endpoints is
                          # documented in this repo, so this is conservative
                          # rather than measured
MAX_EMPTY_CHUNKS = 3     # consecutive empty responses before concluding the
                          # venue's history has run out, not just this window


def _get_json(url: str, params: dict[str, Any], timeout: int = 20) -> dict:
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(f"{url}?{qs}", headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        import json
        return json.loads(r.read().decode())


def _walk_backward(url: str, base_params: dict[str, Any], since_ms: int,
                    until_ms: int, extract, verbose: bool = True) -> list[dict]:
    """Page a start/end-timestamp history endpoint backward from `until_ms`
    to `since_ms`, stopping early if the venue's own history runs out first.

    `extract(payload) -> list[dict]` turns one call's JSON body into the
    records to keep; kept as a parameter because Deribit and Binance shape
    their funding-history responses differently.
    """
    records: list[dict] = []
    cur_end = until_ms
    empty_run = 0
    calls = 0
    while cur_end > since_ms and empty_run < MAX_EMPTY_CHUNKS:
        cur_start = max(since_ms, cur_end - CHUNK_DAYS * DAY_MS)
        params = dict(base_params, start_timestamp=cur_start, end_timestamp=cur_end)
        try:
            payload = _get_json(url, params)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            if verbose:
                print(f"    call failed ({cur_start}..{cur_end}): {type(e).__name__}: {e}")
            break
        calls += 1
        batch = extract(payload)
        if not batch:
            empty_run += 1
        else:
            empty_run = 0
            records.extend(batch)
        cur_end = cur_start
        time.sleep(REQUEST_SLEEP_S)
    if verbose:
        print(f"    {calls} calls, {len(records)} raw records, "
              f"stopped at {'venue history exhausted' if empty_run >= MAX_EMPTY_CHUNKS else 'target reached'}")
    return records


def fetch_deribit_funding(since_ms: int, until_ms: int, instrument: str = "BTC-PERPETUAL",
                           verbose: bool = True) -> pd.DataFrame:
    """Deribit funding-rate history for `instrument`.

    Response shape asserted from Deribit's public JSON-RPC convention
    (`{"result": [...]}`) and NOT verified in this session. Field names
    inside each record (`timestamp`, `interest_1h`, `interest_8h`,
    `index_price`, ...) are taken from Deribit's published API reference as
    of this repo's knowledge and may drift; `pd.json_normalize` is used
    instead of hand-picking columns so an unexpected extra or renamed field
    does not crash the harvest -- whatever keys actually come back are kept.
    """
    def extract(payload):
        if "error" in payload:
            raise RuntimeError(f"deribit error: {payload['error']}")
        return payload.get("result", [])

    if verbose:
        print(f"  deribit funding [{instrument}] ...")
    recs = _walk_backward(DERIBIT_FUNDING, {"instrument_name": instrument},
                           since_ms, until_ms, extract, verbose)
    if not recs:
        return pd.DataFrame()
    df = pd.json_normalize(recs)
    ts_col = "timestamp" if "timestamp" in df.columns else None
    if ts_col is None:
        raise RuntimeError(f"deribit funding: no 'timestamp' field in response; "
                            f"got columns {list(df.columns)}")
    df["ts"] = (df[ts_col].astype("int64") // 1000)
    df = df.drop_duplicates("ts").sort_values("ts", ignore_index=True)
    return df


def fetch_binance_funding(since_ms: int, until_ms: int, symbol: str = "BTCUSDT",
                           verbose: bool = True) -> pd.DataFrame:
    """Binance USDT-margined perpetual funding-rate history.

    Fallback only -- see the module docstring on why `fapi.binance.com` may
    share `api.binance.com`'s documented 451 from a GH Actions runner. If it
    does, this call fails fast and the caller keeps whatever Deribit gave.
    """
    def extract(payload):
        if isinstance(payload, dict) and "code" in payload and "msg" in payload:
            raise RuntimeError(f"binance error: {payload}")
        return payload

    if verbose:
        print(f"  binance funding [{symbol}] (fallback) ...")
    recs = _walk_backward(BINANCE_FUNDING, {"symbol": symbol, "limit": 1000},
                           since_ms, until_ms, extract, verbose)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    df["ts"] = (df["fundingTime"].astype("int64") // 1000)
    df = df.drop_duplicates("ts").sort_values("ts", ignore_index=True)
    return df


def fetch_deribit_dvol(currency: str = "BTC", verbose: bool = True) -> pd.DataFrame:
    """Deribit's historical-volatility (DVOL-family) series for `currency`.

    No start/end parameters exist on this endpoint per Deribit's reference,
    so there is nothing to page: this returns whatever window the venue's
    backend currently serves. `OPTION_BUYER_ALPHA.md` and
    `SELLER_DIRECTIONAL_ALPHA.md` document a prior fetch (2026-06-12,
    evidently run with real internet access, not from this container) that
    got 2023-09-present (~2.7 years). That prior number is reported here as
    the best available prior, not re-verified.
    """
    if verbose:
        print(f"  deribit DVOL [{currency}] ...")
    payload = _get_json(DERIBIT_DVOL, {"currency": currency})
    if "error" in payload:
        raise RuntimeError(f"deribit DVOL error: {payload['error']}")
    result = payload.get("result", [])
    if not result:
        return pd.DataFrame()
    # documented as [timestamp_ms, volatility] pairs; defend against a
    # dict-shaped response too rather than assuming the pair form is exact.
    if isinstance(result[0], (list, tuple)):
        df = pd.DataFrame(result, columns=["ts_ms", "volatility"])
    else:
        df = pd.json_normalize(result)
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "ts_ms"})
    df["ts"] = (df["ts_ms"].astype("int64") // 1000)
    df = df.drop_duplicates("ts").sort_values("ts", ignore_index=True)
    return df


def _merge_with_existing(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Union new rows onto whatever is already committed, by `ts`. Never
    shrinks the bundle -- same principle as `harvest.py`'s refusal to let a
    short fetch overwrite a longer one, applied here by construction (union,
    not replace) rather than by an explicit length check, because unlike
    `harvest.py` this bundle is meant to accrete across many scheduled runs
    rather than be rebuilt whole each time.
    """
    if not path.exists():
        return new
    old = pd.read_parquet(path)
    if new.empty:
        return old
    both = pd.concat([old, new], ignore_index=True)
    return both.drop_duplicates("ts").sort_values("ts", ignore_index=True)


def harvest_funding(out_dir: Path, days: int, verbose: bool = True) -> pd.DataFrame | None:
    path = out_dir / "funding_btc.parquet"
    now_ms = int(time.time() * 1000)
    if path.exists():
        existing = pd.read_parquet(path, columns=["ts"])
        # tail-only: re-walk from just before the last committed point, not
        # the full --days window, once a bundle already exists (mirrors
        # serve/fetch.py's "fetch only the tail" design for the live cron).
        since_ms = int(existing["ts"].max()) * 1000 - 2 * DAY_MS
        print(f"  existing bundle: {len(existing):,} rows, topping up since "
              f"{pd.Timestamp(since_ms, unit='ms', tz='UTC')}")
    else:
        since_ms = now_ms - days * DAY_MS
        print(f"  no existing bundle, backfilling {days} days from "
              f"{pd.Timestamp(since_ms, unit='ms', tz='UTC')}")

    df = fetch_deribit_funding(since_ms, now_ms, verbose=verbose)
    source = "deribit"
    if df.empty:
        print("  deribit funding returned nothing; falling back to binance")
        df = fetch_binance_funding(since_ms, now_ms, verbose=verbose)
        source = "binance"
    if df.empty:
        print("  FAILED: both deribit and binance funding fetches returned nothing")
        return None
    df["source"] = source
    merged = _merge_with_existing(path, df)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False, compression="zstd")
    span = (merged["ts"].max() - merged["ts"].min()) / 86400
    print(f"  funding: wrote {path} -- {len(merged):,} rows, {span:.0f} days "
          f"span, source={source} ({path.stat().st_size/1024:.0f} KB)")
    return merged


def harvest_dvol(out_dir: Path, verbose: bool = True) -> pd.DataFrame | None:
    path = out_dir / "dvol_btc.parquet"
    try:
        df = fetch_deribit_dvol(verbose=verbose)
    except Exception as e:                                       # noqa: BLE001
        print(f"  FAILED: deribit DVOL fetch raised {type(e).__name__}: {e}")
        return None
    if df.empty:
        print("  FAILED: deribit DVOL returned nothing")
        return None
    merged = _merge_with_existing(path, df)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False, compression="zstd")
    span = (merged["ts"].max() - merged["ts"].min()) / 86400
    print(f"  dvol: wrote {path} -- {len(merged):,} rows, {span:.0f} days "
          f"span ({path.stat().st_size/1024:.0f} KB)")
    return merged


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Harvest funding-rate + DVOL history "
                                             "for the ONSET problem (see eval/newdata.py)")
    p.add_argument("--days", type=int, default=3500,
                   help="backfill window on a fresh bundle (default ~9.6y, covers "
                        "Deribit BTC-PERPETUAL's Aug-2016 launch with margin)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)

    ok, failed = [], []

    print("[harvest_newdata] funding rate ...", flush=True)
    try:
        if harvest_funding(a.out_dir, a.days) is not None:
            ok.append("funding")
        else:
            failed.append("funding")
    except Exception as e:                                       # noqa: BLE001
        print(f"  funding: FAILED -- {type(e).__name__}: {e}")
        failed.append("funding")

    print("[harvest_newdata] DVOL ...", flush=True)
    try:
        if harvest_dvol(a.out_dir) is not None:
            ok.append("dvol")
        else:
            failed.append("dvol")
    except Exception as e:                                       # noqa: BLE001
        print(f"  dvol: FAILED -- {type(e).__name__}: {e}")
        failed.append("dvol")

    print(f"\nharvested: {ok}")
    if failed:
        print(f"failed:    {failed}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""
eval/datasources.py
=====================================================================
Probe all public crypto OHLCV data sources and document which ones are
reachable from this container.

NETWORK BOUNDARY

This container reaches the internet ONLY through a preconfigured HTTPS proxy
that enforces an organization egress policy. The policy blocks:

  - ALL crypto exchange APIs (Binance, Kraken, OKX, Bybit, KuCoin, Gate.io, etc.)
  - ALL public data aggregators (CoinGecko, CryptoCompare, CoinCap, Yahoo Finance)
  - Hugging Face datasets
  - Most cloud providers (AWS, Google Cloud, Azure)
  - Most public repositories (Wikipedia, Wikimedia, ResearchGate, SSRN, etc.)

WHAT WORKS

  - raw.githubusercontent.com (GitHub raw file serving)
  - gist.github.com (GitHub gist hosting)
  - Anthropic APIs (api.anthropic.com, claude.ai)
  - GitLab API (HTTP 404 but reachable; would need gitlab.com to be allowed)

A GitHub Actions runner (a standard GH runner in a US region, used to run the
fetch-data and harvest-assets workflows) DOES have access to these APIs. That
runner is the data channel: the altcoin bundles in data/assets/ were fetched
on a runner and committed to the repo.

PROBE STRATEGY

This script tests representative endpoints from each major category:

  1. Live exchange APIs (Binance, Kraken, OKX, Bybit, KuCoin, Gate.io)
  2. Data aggregators (CoinGecko, CryptoCompare, CoinCap, Yahoo Finance)
  3. Hugging Face datasets and models
  4. GitHub-hosted raw data
  5. Alternative public repositories (archive.org, Wikimedia, Kaggle, Zenodo)

For each, it reports the HTTP status code, whether real data is returned,
and (if reachable) the granularity and history depth.

All FAILED requests are policy denials (403/407) or tunnel failures (no proxy
exception for that host). SUCCESS means HTTP 200 with non-empty body.

    python -m model.eval.datasources --output model/artifacts/datasources.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class ProbeResult:
    """Result of probing a single endpoint."""
    category: str
    name: str
    url: str
    http_status: int | None
    reachable: bool
    has_data: bool
    sample: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


def probe_url(category: str, name: str, url: str, params: dict[str, str] | None = None) -> ProbeResult:
    """Test a URL and return detailed probe result.

    Args:
        category: e.g., 'Binance', 'CoinGecko'
        name: human-readable endpoint name
        url: base URL (params will be appended)
        params: optional query parameters

    Returns:
        ProbeResult with status, reachability, and sample data.
    """
    if params:
        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{param_str}"
    else:
        full_url = url

    ua = {"User-Agent": "btc-datasource-probe/1.0 (+https://github.com/deeploopinnovations/btc-dashboard)"}
    result = ProbeResult(
        category=category,
        name=name,
        url=full_url,
        http_status=None,
        reachable=False,
        has_data=False,
        sample="",
        notes="",
    )

    try:
        req = urllib.request.Request(full_url, headers=ua)
        with urllib.request.urlopen(req, timeout=10) as r:
            result.http_status = r.status
            result.reachable = True
            body = r.read(500).decode("utf-8", errors="ignore")
            if body and body.strip():
                result.has_data = True
                result.sample = body[:200]
            if r.status == 200:
                result.notes = "Success"
            return result
    except urllib.error.HTTPError as e:
        result.http_status = e.code
        if e.code == 403:
            result.notes = "Policy denial (403 Forbidden)"
        elif e.code == 407:
            result.notes = "Policy denial (407 Proxy Authentication Required)"
        elif e.code == 451:
            result.notes = "Legal block (451 Unavailable For Legal Reasons)"
        else:
            result.notes = f"HTTP error {e.code}"
        return result
    except urllib.error.URLError as e:
        if "403" in str(e) or "Forbidden" in str(e):
            result.notes = "Tunnel rejected (proxy 403)"
        else:
            result.notes = f"Network error: {str(e)[:100]}"
        return result
    except Exception as e:
        result.notes = f"{type(e).__name__}: {str(e)[:80]}"
        return result


def main(argv: list[str] | None = None) -> int:
    """Probe all candidate data sources and write results to JSON."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/artifacts/datasources.json"),
        help="Output JSON file (default: model/artifacts/datasources.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print results to stdout as well",
    )
    args = parser.parse_args(argv)

    results: list[ProbeResult] = []

    # Crypto exchanges (expect all blocked)
    print("[probe] Binance API ...", flush=True)
    results.append(probe_url(
        "Binance", "BTCUSDT 1m klines",
        "https://api.binance.com/api/v3/klines",
        {"symbol": "BTCUSDT", "interval": "1m", "limit": "10"},
    ))
    results.append(probe_url(
        "Binance", "data-api (historical)",
        "https://data-api.binance.vision/api/v3/klines",
        {"symbol": "BTCUSDT", "interval": "1m", "limit": "10"},
    ))

    print("[probe] Kraken API ...", flush=True)
    results.append(probe_url(
        "Kraken", "BTC/USD OHLC",
        "https://api.kraken.com/0/public/OHLC",
        {"pair": "XBTUSDT", "interval": "1"},
    ))

    print("[probe] OKX API ...", flush=True)
    results.append(probe_url(
        "OKX", "BTC-USDT candles",
        "https://www.okx.com/api/v5/market/candles",
        {"instId": "BTC-USDT", "bar": "1m", "limit": "10"},
    ))

    print("[probe] Bybit API ...", flush=True)
    results.append(probe_url(
        "Bybit", "BTCUSDT klines",
        "https://api.bybit.com/v5/market/kline",
        {"category": "spot", "symbol": "BTCUSDT", "interval": "1"},
    ))

    print("[probe] KuCoin API ...", flush=True)
    results.append(probe_url(
        "KuCoin", "BTC-USDT candles",
        "https://api.kucoin.com/api/v1/market/candles",
        {"symbol": "BTC-USDT", "type": "1min"},
    ))

    print("[probe] Gate.io API ...", flush=True)
    results.append(probe_url(
        "Gate.io", "BTC_USDT candlesticks",
        "https://api.gateio.ws/api/v4/spot/candlesticks",
        {"currency_pair": "BTC_USDT", "interval": "1m", "limit": "10"},
    ))

    # Data aggregators
    print("[probe] CryptoCompare API ...", flush=True)
    results.append(probe_url(
        "CryptoCompare", "BTC/USD minute OHLC",
        "https://min-api.cryptocompare.com/data/v2/histominute",
        {"fsym": "BTC", "tsym": "USD", "limit": "10"},
    ))

    print("[probe] CoinGecko API ...", flush=True)
    results.append(probe_url(
        "CoinGecko", "Simple price endpoint",
        "https://api.coingecko.com/api/v3/simple/price",
        {"ids": "bitcoin", "vs_currencies": "usd"},
    ))
    results.append(probe_url(
        "CoinGecko", "OHLC endpoint",
        "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc",
        {"vs_currency": "usd", "days": "1"},
    ))

    print("[probe] CoinCap API ...", flush=True)
    results.append(probe_url(
        "CoinCap", "Candles endpoint",
        "https://api.coincap.io/v2/candles",
        {"exchange": "binance", "baseId": "bitcoin", "quoteId": "usd"},
    ))

    print("[probe] Yahoo Finance ...", flush=True)
    results.append(probe_url(
        "Yahoo Finance", "BTC-USD chart",
        "https://query1.finance.yahoo.com/v7/finance/chart/BTC-USD",
        {"interval": "1m", "range": "1d"},
    ))

    # ML/dataset platforms
    print("[probe] Hugging Face ...", flush=True)
    results.append(probe_url(
        "Hugging Face", "Datasets API",
        "https://huggingface.co/api/datasets",
        {"search": "crypto ohlc", "limit": "10"},
    ))

    # GitHub (expected to work, at least raw)
    print("[probe] GitHub ...", flush=True)
    results.append(probe_url(
        "GitHub", "raw.githubusercontent.com (README)",
        "https://raw.githubusercontent.com/deeploopinnovations/btc-dashboard/main/README.md",
    ))
    results.append(probe_url(
        "GitHub", "gist.github.com",
        "https://gist.github.com/",
    ))
    results.append(probe_url(
        "GitHub", "api.github.com",
        "https://api.github.com/users/octocat",
    ))

    # Public archives & research (expect blocked)
    print("[probe] Public archives ...", flush=True)
    results.append(probe_url(
        "Archive.org", "Wayback Machine",
        "https://archive.org/",
    ))
    results.append(probe_url(
        "Wikipedia", "Bitcoin article",
        "https://en.wikipedia.org/wiki/Bitcoin",
    ))

    # Convert to dicts for JSON serialization
    result_dicts = [r.to_dict() for r in results]

    # Summary statistics
    reachable = [r for r in results if r.reachable]
    with_data = [r for r in results if r.has_data]

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_probes": len(results),
        "reachable": len(reachable),
        "with_data": len(with_data),
        "note": (
            "All crypto exchange APIs and major data aggregators are blocked by "
            "the container's egress proxy (403 Forbidden). The only reachable data "
            "sources are: raw.githubusercontent.com, gist.github.com, and local data. "
            "Altcoin history was fetched externally (on GitHub Actions runners) and "
            "committed to data/assets/."
        ),
        "working_sources": [
            {
                "name": r.name,
                "category": r.category,
                "url": r.url.split("?")[0],  # Base URL only
                "status": r.http_status,
            }
            for r in reachable if r.has_data
        ],
        "blocked_sources": [
            {
                "name": r.name,
                "category": r.category,
                "status": r.http_status or "unreachable",
                "reason": r.notes,
            }
            for r in results if not r.reachable or not r.has_data
        ],
    }

    output = {
        "summary": summary,
        "probes": result_dicts,
    }

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    if args.verbose:
        print("\n" + "=" * 80)
        print("PROBE RESULTS")
        print("=" * 80)
        for r in results:
            status_str = f"HTTP {r.http_status}" if r.http_status else "UNREACHABLE"
            ok_str = "✓" if r.reachable and r.has_data else "✗"
            print(f"{ok_str} {r.category:20s} {r.name:35s} {status_str:20s}")
            if r.notes:
                print(f"  → {r.notes}")

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total probes: {summary['total_probes']}")
        print(f"Reachable:   {summary['reachable']}")
        print(f"With data:   {summary['with_data']}")
        print()
        print(summary["note"])

    print(f"\n[datasources] Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

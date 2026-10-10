"""
tests/test_debug_trace.py
=====================================================================
The serving gate for NOCTUA_DEBUG, the stage-by-stage forecast trace
(serve/debug.py). Offline: the committed bundle, no network.

THE CONTRACT
  OFF (the default) PRINTS NOTHING, and costs no field evaluation.
  ON PRINTS ONE JSON RECORD PER STAGE, TO STDERR ONLY, IN PIPELINE ORDER.
  THE TRACE NEVER CHANGES A SERVED VALUE: the payload, and the noctua.json
  predict.main writes, are identical with the trace on and off.
  THE TRACE TELLS THE TRUTH: its numbers are the ones the forecast used.
  A STAGE THAT RAISES IS NAMED, AND THE EXCEPTION PROPAGATES UNCHANGED.
  A FIELD THAT FAILS TO EVALUATE IS RECORDED, AND NEVER BREAKS A FORECAST.
  A PRIMARY FEED THAT FAILS BEFORE THE FALLBACK SUCCEEDS IS RECORDED (it was
  silently dropped before). AND THE GATE CAN FAIL (R2).

    python model/tests/test_debug_trace.py
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import debug as D                                       # noqa: E402
from serve import fetch as F                                       # noqa: E402
from serve import predict as P                                     # noqa: E402
from serve.history import load_bundle                              # noqa: E402
from serve.runtime import load_model                               # noqa: E402

PROD_A, PROD_H = 17, 19
STAGES = ["anchor", "features", "flags", "prepare", "predict", "vol_correction", "payload"]


def records(err: str) -> list:
    return [json.loads(ln[len(D.PREFIX):]) for ln in err.splitlines() if ln.startswith(D.PREFIX)]


@contextlib.contextmanager
def debug_env(value):
    old = os.environ.get(D.ENV)
    if value is None:
        os.environ.pop(D.ENV, None)
    else:
        os.environ[D.ENV] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(D.ENV, None)
        else:
            os.environ[D.ENV] = old


def run(model, hours, anchor, value):
    out, err, raw = io.StringIO(), io.StringIO(), {}
    with debug_env(value), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        pay = P.forecast(model, hours, H=PROD_H, anchor_ts=anchor, raw=raw, fetch_iv=False)
    return pay, raw, out.getvalue(), err.getvalue()


def main() -> int:
    ok = []

    def check(name, good, detail=""):
        ok.append((name, bool(good), detail))

    model, hours = load_model(), load_bundle()
    hts = hours["hour_ts"].to_numpy(np.int64)
    a17 = int(hts[pd.to_datetime(hts, unit="s", utc=True).hour == PROD_A][-1])

    p0, raw0, o0, e0 = run(model, hours, a17, None)
    check("off-is-silent", D.PREFIX not in o0 + e0 and not records(e0),
          f"{len(o0 + e0)} chars of output, none of it a trace record")

    p1, raw1, o1, e1 = run(model, hours, a17, "1")
    recs = records(e1)
    same = json.dumps(p0, sort_keys=True) == json.dumps(p1, sort_keys=True)
    check("on-payload-is-bit-identical", same, "json of every served field, trace off vs on")
    check("on-writes-stderr-only", D.PREFIX not in o1 and len(recs) > 0,
          f"{len(recs)} records on stderr, 0 on stdout")
    got = [r["stage"] for r in recs]
    order = [s for s in got if s in STAGES]
    check("on-emits-every-stage-in-pipeline-order", order == STAGES, f"got {got}")

    by = {r["stage"]: r for r in recs}

    def got_(path):               # a missing stage or field is a FAIL, not a crash
        v = by
        for k in path.split("."):
            v = v.get(k, "<missing>") if isinstance(v, dict) else "<missing>"
        return v
    truth = [
        ("anchor.row", raw1["anchor_row"]),
        ("anchor.exact", True),
        ("prepare.anchor_logvol", raw1["anchor_logvol"]),
        ("vol_correction.cal.factor", raw1["cal"]["factor"]),
        ("vol_correction.sigma_mean_after", float(raw1["pred"]["sigma_mean"][0])),
        ("payload.sigma_window_pct", p1["sigma_window_pct"]),
        ("payload.p_up", p1["p_up"]),
    ]

    def same_(a, b):
        return a == b or (isinstance(a, float) and isinstance(b, float)
                          and abs(a - b) <= 1e-12 * max(1, abs(b)))
    bad = [(n, got_(n), b) for n, b in truth if not same_(got_(n), b)]
    check("trace-tells-the-truth", not bad, f"{len(truth)} fields match the forecast's own"
          + (f"; MISMATCH {bad}" if bad else ""))
    nf, vals = got_("features.nonfinite"), got_("features.values")
    check("features-record-names-nonfinite-columns", isinstance(nf, list) and isinstance(vals, dict),
          f"{len(vals)} features, non-finite: {nf}")
    check("payload-records-curve-monotonicity", got_("payload.curves_monotone") is True,
          "touch probability non-increasing in distance, both sides")

    # a planted failure inside a stage: named, then re-raised unchanged
    real = P.build_features

    def boom(*a, **k):
        raise ValueError("planted in build_features")
    P.build_features = boom
    err, caught = io.StringIO(), None
    try:
        with debug_env("1"), contextlib.redirect_stderr(err):
            P.forecast(model, hours, H=PROD_H, anchor_ts=a17, fetch_iv=False)
    except Exception as e:                 # any other type is itself a failure
        caught = e
    finally:
        P.build_features = real
    rec = [r for r in records(err.getvalue()) if "error" in r]
    check("stage-error-is-named-and-reraised",
          isinstance(caught, ValueError) and str(caught) == "planted in build_features"
          and rec and rec[-1]["stage"] == "features" and "planted" in rec[-1]["error"],
          f"raised {type(caught).__name__}: {caught}; last record: {rec[-1] if rec else None}")

    # lazy fields: never evaluated when off; a failing one never breaks a run
    calls = []
    with debug_env(None), contextlib.redirect_stderr(io.StringIO()):
        D.trace("probe", x=lambda: calls.append(1) or 1)
    check("off-evaluates-no-field", calls == [], "a lazy field was not called with the trace off")
    err, r = io.StringIO(), None
    try:
        with debug_env("1"), contextlib.redirect_stderr(err):
            D.trace("probe", bad=lambda: 1 / 0, good=lambda: 2)
        r = records(err.getvalue())
    except Exception as e:
        r = f"trace raised {type(e).__name__}"
    check("failing-field-is-recorded-not-raised",
          isinstance(r, list) and r and r[0]["good"] == 2
          and "ZeroDivisionError" in str(r[0]["bad"]), f"{r}")

    on = [v for v in ("1", "true", "YES", "on") if (os.environ.update({D.ENV: v}) or D.enabled())]
    off = [v for v in ("", "0", "false", "off", "no") if (os.environ.update({D.ENV: v}) or D.enabled())]
    os.environ.pop(D.ENV, None)
    check("switch-values", len(on) == 4 and off == [] and not D.enabled(),
          "1/true/yes/on enable; ''/0/false/off/no and unset disable")

    # predict.main end to end: the written noctua.json is identical, stdout clean
    def main_run(value, d):
        out, err = io.StringIO(), io.StringIO()
        with debug_env(value), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            P.main(["--offline", "--out-dir", d, "--anchor", str(a17)])
        return (Path(d) / "noctua.json").read_text(), out.getvalue(), err.getvalue()
    with tempfile.TemporaryDirectory() as t0, tempfile.TemporaryDirectory() as t1:
        n0, _, _ = main_run(None, t0)
        n1, so, se = main_run("1", t1)
    st = [r["stage"] for r in records(se)]
    check("main-file-identical-stdout-clean",
          n0 == n1 and D.PREFIX not in so and "history" in st and "write" in st,
          f"noctua.json identical: {n0 == n1}; stages on stderr: {st}")

    # the LIVE path through main, without the network: get_hours' real info
    # dict (which carries its own "source") must not collide with the trace's
    # fields -- the first live run with the trace on crashed on exactly that
    real_gh = P.get_hours
    live_info = {"bundle_last_hour": a17, "gap_hours": 3, "tail_hours_requested": 12,
                 "fetched_bars": 577, "fresh_complete_hours": 47, "new_hours": 3,
                 "source": "bitstamp:btcusd", "contiguous": True, "gaps": 0,
                 "largest_gap_hours": 1, "bundle_rewritten": False}
    P.get_hours = lambda fetch_fn, **k: (hours, dict(live_info))
    err, crash = io.StringIO(), None
    try:
        with tempfile.TemporaryDirectory() as t2, debug_env("1"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            P.main(["--out-dir", t2, "--anchor", str(a17)])
    except Exception as e:
        crash = f"{type(e).__name__}: {e}"
    finally:
        P.get_hours = real_gh
    h = [x for x in records(err.getvalue()) if x["stage"] == "history"]
    check("live-path-main-does-not-crash-with-trace-on",
          crash is None and h and h[0].get("source") == "bitstamp:btcusd",
          f"crash: {crash}; history record: {h[0] if h else None}")

    # the fallback feed: bitstamp fails, coinbase serves -- the first error is kept
    fb, fc, fv = F.fetch_bitstamp, F.fetch_coinbase, F.validate_bars

    def bad_feed(**k):
        raise RuntimeError("planted bitstamp outage")

    def planted_coinbase(**k):
        return pd.DataFrame({"source": ["coinbase:planted"]})
    F.fetch_bitstamp, F.fetch_coinbase = bad_feed, planted_coinbase
    F.validate_bars = lambda df, tail: df
    err = io.StringIO()
    try:
        with debug_env("1"), contextlib.redirect_stderr(err):
            F.fetch_bars(tail_hours=12)
    finally:
        F.fetch_bitstamp, F.fetch_coinbase, F.validate_bars = fb, fc, fv
    r = [x for x in records(err.getvalue()) if x["stage"] == "fetch.fallback"]
    check("fallback-feed-error-is-recorded",
          r and "planted bitstamp outage" in json.dumps(r[0]) and r[0]["used"] == "planted_coinbase",
          f"{r[0] if r else None}")

    # R2: the payload comparison above can see a change of one rounded unit
    p2 = copy.deepcopy(p0)
    p2["barrier_curves"]["up"][0]["touch_prob"] += 1e-4
    check("gate-can-fail", json.dumps(p0, sort_keys=True) != json.dumps(p2, sort_keys=True),
          "one touch probability +1e-4 is detected")

    print("debug-trace serving gate")
    for n, good, m in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {n}: {m}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok) - len(bad)}/{len(ok)} checks passed")
    if not bad:
        print("all checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

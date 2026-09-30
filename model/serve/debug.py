"""
serve/debug.py
=====================================================================
NOCTUA_DEBUG -- a stage-by-stage trace of one forecast. OFF by default.

WHY THIS EXISTS

A wrong forecast does not crash. Every stage hands the next a finite number,
so a bad anchor, a stale feature or a factor on its clip rail comes out the
far end as a plausible sigma, and the only way to find which stage went wrong
has been to re-run the pipeline by hand and print things. The trace prints,
for each stage, what it was given and what it produced, so a bad number can be
walked back to the FIRST stage whose output is wrong instead of guessed at.

    NOCTUA_DEBUG=1 python model/serve/predict.py --offline --out-dir /tmp/x 2> trace.log
    grep '^\\[noctua-debug\\]' trace.log | cut -c16- | jq .

One record per stage, one line each, on STDERR: `[noctua-debug] {json}`.
stdout and every written file are unchanged.

BACK TO NORMAL: unset NOCTUA_DEBUG (or set it to 0/false/off/no). Nothing else
to undo -- the switch is read at call time, so no code or config changes, and
OFF costs one environment lookup per stage: field values may be passed as
callables, which are never evaluated when the trace is off.

THE CONTRACT (tests/test_debug_trace.py)
  - OFF prints nothing; ON prints to stderr only.
  - The trace never changes a served value (payload bit-identical on vs off).
  - A field that fails to evaluate is recorded as an error string; the trace
    never breaks a forecast.
  - A stage that raises is recorded with its name, and the exception then
    propagates unchanged.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time

import numpy as np

ENV = "NOCTUA_DEBUG"
PREFIX = "[noctua-debug] "
_OFF = ("", "0", "false", "off", "no")
_MAX_LIST = 16          # longer arrays are summarised, not dumped


def enabled() -> bool:
    return os.environ.get(ENV, "").strip().lower() not in _OFF


def _plain(v):
    """JSON-safe version of a traced value; large arrays become a summary."""
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return str(v)
    if isinstance(v, np.ndarray):
        if v.size <= _MAX_LIST:
            return [_plain(x) for x in v.ravel().tolist()]
        f = v[np.isfinite(v)] if v.dtype.kind == "f" else v
        return {"shape": list(v.shape), "n_nonfinite": int(v.size - f.size),
                "min": _plain(f.min()) if f.size else None,
                "max": _plain(f.max()) if f.size else None}
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def trace(stage: str, **fields) -> None:
    """Print one record for `stage` when the trace is on. A callable field is
    evaluated only then, and a failure to evaluate is recorded, not raised."""
    if not enabled():
        return
    rec = {"stage": stage}
    for k, v in fields.items():
        if callable(v):
            try:
                v = v()
            except Exception as e:                         # noqa: BLE001
                v = f"<field failed: {type(e).__name__}: {e}>"
        rec[k] = _plain(v)
    print(PREFIX + json.dumps(rec, default=str), file=sys.stderr, flush=True)


@contextlib.contextmanager
def stage(name: str):
    """Name the stage an exception came from, then let it propagate unchanged."""
    if not enabled():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    except BaseException as e:
        trace(name, error=f"{type(e).__name__}: {e}",
              ms=round(1000 * (time.perf_counter() - t0), 1))
        raise

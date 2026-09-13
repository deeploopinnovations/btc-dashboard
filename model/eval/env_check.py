"""
eval/env_check.py
=====================================================================
Compare the INSTALLED package versions against `requirements-research.txt`,
and stamp the answer into every result that depends on them.

WHY THIS EXISTS

`noctua_v1` failed to reproduce its published QLIKE at all three horizons while
every deterministic teacher returned to within 1e-5. The cause was not seeds --
two full rebuilds were bit-identical, 528 of 528 arrays -- and not code, and not
the corpus. It was one line: torch is pinned at 2.13.0+cu130 and the container
carried 2.14.0+cu130, while numpy, scipy, pandas and arch all matched exactly.

The packages that drifted and the arms that moved are the same set, which is what
makes it the explanation rather than a suspicion: OLS through numpy and GARCH MLE
through arch reach the same optimum whatever the summation order, while SGD over
thousands of steps turns a last-bit kernel difference into a pooled-loss shift of
0.002-0.004 -- enough to flip the H=24 verdict from a tie to a win.

Nothing checked the pin. A pin nobody verifies is a comment. (R63)

WHAT THIS DOES AND DELIBERATELY DOES NOT DO

It REPORTS drift and returns it as data; it does not refuse. Refusing would have
blocked the corpus restoration and the first reproduction of the load-bearing
Phase 2 number, both of which were worth doing on a drifted torch as long as the
drift is recorded beside the numbers. What must never happen again is the drift
being INVISIBLE, so the remedy is provenance plus a loud line, not a gate.

`drift_is_fatal_for()` names which arms a given drift actually invalidates, so a
numpy drift is not reported as casting doubt on a GARCH result and a torch drift
is not reported as casting doubt on an OLS one.

    python -m model.eval.env_check
    python -m model.eval.env_check --selftest
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQ = Path("model/requirements-research.txt")

# Which arms each package's arithmetic can move. A closed-form or MLE estimator
# converges to the same optimum regardless of summation order; an SGD-trained
# network does not.
OWNS = {
    "torch":  ("neural", "noctua_* arms -- SGD amplifies last-bit differences"),
    "numpy":  ("numeric", "every arm, but closed-form fits converge anyway"),
    "scipy":  ("optimiser", "garch_* arms via the MLE optimiser"),
    "arch":   ("optimiser", "garch_* arms"),
    "statsmodels": ("optimiser", "garch_* arms -- arch depends on it; the "
                    "2026-09-12 rebuild CLEARED it empirically, garch_t "
                    "returning to 1.3e-4 across a 0.14.6 -> 0.15.0 drift"),
    "pandas": ("dataframe", "feature construction and alignment"),
    "pyarrow": ("io", "parquet round-trip only; float64 is exact through it, "
                "so a drift here moves no number"),
}


def parse_pins(path: Path = REQ) -> dict:
    """Pins only. A requirement without `==` is not a pin and is not reported."""
    pins = {}
    if not path.exists():
        return pins
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*(\S+)$", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def installed(names) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = __import__(n).__version__
        except Exception:                                         # noqa: BLE001
            out[n] = None
    return out


def compare(pins: dict, have: dict) -> list:
    """[(name, pinned, found, state)] with state in ok/DRIFT/ABSENT."""
    rows = []
    for n in sorted(pins):
        want, got = pins[n], have.get(n)
        state = "ABSENT" if got is None else ("ok" if got == want else "DRIFT")
        rows.append((n, want, got, state))
    return rows


def drift_is_fatal_for(rows) -> list:
    """Which arms a drift actually puts in question -- not 'everything'."""
    out = []
    for n, want, got, state in rows:
        if state == "DRIFT" and n in OWNS:
            out.append((n, want, got, OWNS[n][1]))
    return out


def stamp(path: Path = REQ) -> dict:
    """The environment block to store beside any result. Cheap, so always on."""
    pins = parse_pins(path)
    have = installed(pins)
    rows = compare(pins, have)
    return {
        "pins_file": str(path),
        "packages": {n: {"pinned": w, "installed": g, "state": s}
                     for n, w, g, s in rows},
        "drift": [n for n, _, _, s in rows if s == "DRIFT"],
        "python": sys.version.split()[0],
    }


def report(path: Path = REQ) -> int:
    pins = parse_pins(path)
    if not pins:
        print(f"no pins found in {path}", file=sys.stderr)
        return 2
    rows = compare(pins, installed(pins))
    w = max(len(n) for n, *_ in rows)
    print(f"environment against {path}")
    for n, want, got, state in rows:
        mark = {"ok": "ok  ", "DRIFT": "DRIFT", "ABSENT": "ABSENT"}[state]
        print(f"  [{mark:6}] {n:<{w}}  pinned {want:<16} installed {got}")
    drifted = [(n, want, got) for n, want, got, st in rows if st == "DRIFT"]
    fatal = drift_is_fatal_for(rows)
    if not drifted:
        print("\nno drift: neural and deterministic results are both "
              "reproducible against this environment.")
        return 0
    # The count is of DRIFTS, not of attributions. An earlier version of this
    # function printed len(fatal) and called it the number of drifted packages,
    # so two real drifts outside the attribution map disappeared from a report
    # whose whole purpose is that drift stops being invisible.
    print(f"\n{len(drifted)} package(s) drifted from the pin "
          f"({', '.join(n for n, _, _ in drifted)}).")
    print("What each one puts in question, and ONLY that:")
    for n, want, got, who in fatal:
        print(f"  {n} {want} -> {got}: {who}")
    unattributed = [n for n, _, _ in drifted if n not in OWNS]
    if unattributed:
        print(f"  NOT ATTRIBUTED to any arm, which is a gap in this file's map "
              f"rather than a clean bill of health: {', '.join(unattributed)}")
    print("\nThis is reported, not refused -- but any number produced here must "
          "carry this stamp (R63).")
    return 0


def selftest() -> int:
    import tempfile
    checks = []
    p = Path(tempfile.mkdtemp()) / "req.txt"
    p.write_text("# comment\nnumpy==1.2.3\ntorch==9.9.9+cu130\n"
                 "pandas>=2.0\n\nscipy==1.17.1  # inline comment\n")
    pins = parse_pins(p)
    checks.append(("pins-only-are-parsed",
                   pins == {"numpy": "1.2.3", "torch": "9.9.9+cu130",
                            "scipy": "1.17.1"},
                   f"{pins} -- `pandas>=2.0` is not a pin and is excluded"))

    rows = compare(pins, {"numpy": "1.2.3", "torch": "2.14.0+cu130",
                          "scipy": None})
    st = {n: s for n, _, _, s in rows}
    checks.append(("states-are-classified",
                   st == {"numpy": "ok", "torch": "DRIFT", "scipy": "ABSENT"},
                   f"{st}"))

    fatal = drift_is_fatal_for(rows)
    checks.append(("drift-is-attributed-narrowly",
                   len(fatal) == 1 and fatal[0][0] == "torch",
                   f"a torch drift implicates {fatal[0][3]!r} and an ABSENT "
                   f"scipy is not reported as a drift"))

    clean = compare({"numpy": "1.2.3"}, {"numpy": "1.2.3"})
    checks.append(("no-false-drift-when-matched",
                   not drift_is_fatal_for(clean), "matched pins raise nothing"))

    # the bug this file shipped with: a drift outside OWNS must still be counted
    rows2 = compare({"torch": "1.0", "somelib": "1.0"},
                    {"torch": "2.0", "somelib": "2.0"})
    n_drift = len([1 for *_, st in rows2 if st == "DRIFT"])
    checks.append(("unattributed-drift-is-still-counted",
                   n_drift == 2 and len(drift_is_fatal_for(rows2)) == 1,
                   f"{n_drift} drifts, 1 attributable -- the report must say "
                   f"2, not 1, or an unmapped drift goes invisible"))

    # the real environment: this is the check that was missing, so it must see
    # the drift that actually happened
    real = stamp()
    checks.append(("real-environment-is-inspected",
                   bool(real["packages"]),
                   f"{len(real['packages'])} pinned package(s); drift: "
                   f"{real['drift'] or 'none'}"))

    print("env_check selftest")
    bad = 0
    for name, ok, detail in checks:
        if not ok:
            bad += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n{len(checks) - bad}/{len(checks)} checks passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="verify the environment pins")
    ap.add_argument("--req", type=Path, default=REQ)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    return selftest() if a.selftest else report(a.req)


if __name__ == "__main__":
    sys.exit(main())

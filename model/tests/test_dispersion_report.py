"""
tests/test_dispersion_report.py
=====================================================================
The serving gate for the dispersion correction: what the published payload is
allowed to move, and what it is not.

WHY THIS IS A DIFFERENT SHAPE FROM tests/test_level_report.py

That gate asserts that the FUNCTIONAL choice reaches the reported scalar and
NOTHING else -- `P3-barrier-channel` proved by probe that the committee builds
every curve from `pred["sigma_atoms"]` and never reads the reported number, so
a 13% change in it moves the curves by 0.000e+00.

This one is the opposite case and `P3-dispersion-barriers-result` says so in
its own closing note: unlike the functional change, the dispersion correction
MOVES THE BARRIER CURVES BY DESIGN. "Safe by construction" is therefore not
available, and the only honest substitute is a contract -- a written list of
what may move and what may not, asserted field by field against a real
forecast.

THE CONTRACT

  MUST NOT MOVE
    the barrier price GRID (`pct` and `price` on every rung) -- this changes
        probabilities ON a fixed grid, never the grid;
    `sigma_functional.sigma_med_pct` -- the hook scales WIDTH about the median
        and a level change smuggled in as a width change is the intervention
        class `P2-mean-level` already rejected;
    the requested confidences on `safe_levels`;
    spot, anchor, settle, history metadata.

  MAY MOVE, and nothing outside this list may
    every `touch_prob`; every non-alpha `safe_levels` field; the reported
    sigma (both units); `sigma_mean_pct` and `mean_over_median`; `p_up`,
    which mixes Stage B over the atoms. `p_vol_amplify` reads `pred["qa"]`
    rather than the atoms, so it is in the may-move list only because
    tightening it to must-not-move would assert an implementation detail this
    gate does not otherwise police.

  MUST MOVE IN A KNOWN DIRECTION
    narrowing removes weight from the high-sigma atoms, which is what drives a
    far touch, so the FARTHEST rung on each side must see its probability
    FALL. A hook that widened while claiming to narrow would pass every
    equality check above and fail this one.

AND THE GATE IS SHOWN TO BE CAPABLE OF FAILING (R2). Five guards in this
project have printed reassuring output while unable to return the other
answer, so the last check perturbs one field and requires the comparison to
notice.

    python model/tests/test_dispersion_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import predict as P                                    # noqa: E402
from serve.history import load_bundle                             # noqa: E402
from serve.runtime import load_model                              # noqa: E402

# The measured production-slice value, used ONLY to exercise the switch. It is
# not adopted and `P.DISP_LAMBDA` stays at 1.0.
PROBE_LAMBDA = 0.87

FROZEN = ("/pct", "/price", "/alpha", "sigma_med_pct")
MAY_MOVE = ("touch_prob", "safe_levels/", "sigma_window_pct",
            "sigma_annualized_pct", "sigma_mean_pct", "mean_over_median",
            "p_up", "p_vol_amplify")


def flatten(d: dict, pre: str = "") -> dict:
    """Every leaf a consumer can read, except the block that reports the knob.

    `dispersion` is excluded because it publishes lambda itself: including it
    would make "the payload changed" true by definition and the contract
    unfalsifiable.
    """
    out = {}
    for k, v in d.items():
        if k in ("dispersion", "source"):
            continue
        if isinstance(v, dict):
            out.update(flatten(v, f"{pre}{k}/"))
        elif isinstance(v, list):
            for i, r in enumerate(v):
                if isinstance(r, dict):
                    out.update(flatten(r, f"{pre}{k}/{i}/"))
                else:
                    out[f"{pre}{k}/{i}"] = r
        else:
            out[f"{pre}{k}"] = v
    return out


def main() -> int:
    ok = []

    def check(name, good, detail=""):
        ok.append((name, bool(good), detail))

    model = load_model()
    hours = load_bundle()
    if hours is None or len(hours) < 24 * 400:
        print("no committed history bundle; cannot run the serving gate")
        return 1

    def run(lam):
        orig, P.DISP_LAMBDA = P.DISP_LAMBDA, lam
        try:
            return P.forecast(model, hours)
        finally:
            P.DISP_LAMBDA = orig

    shipped = P.forecast(model, hours)          # whatever the default is
    unit = run(1.0)
    narrow = run(PROBE_LAMBDA)

    check("default-is-inert", P.DISP_LAMBDA == 1.0
          and shipped["dispersion"]["applied"] is False,
          f"DISP_LAMBDA = {P.DISP_LAMBDA}; nothing here is adopted")

    f_ship, f_unit, f_nar = flatten(shipped), flatten(unit), flatten(narrow)
    same_as_unit = [k for k in f_ship if f_ship[k] != f_unit[k]]
    check("lambda-1-is-the-shipped-payload", not same_as_unit,
          f"differs: {sorted(same_as_unit)}" if same_as_unit
          else f"all {len(f_ship)} published leaves identical")

    moved = sorted(k for k in f_unit if f_unit[k] != f_nar[k])
    check("the-knob-does-something", len(moved) > 0,
          f"{len(moved)} leaves move at lambda = {PROBE_LAMBDA} "
          f"(a knob wired to nothing would make every check below vacuous)")

    frozen_moved = [k for k in moved if any(t in k for t in FROZEN)]
    check("the-grid-does-not-move", not frozen_moved,
          f"moved anyway: {frozen_moved}" if frozen_moved
          else "barrier pct/price, safe-level alphas and sigma_med all held")

    stray = [k for k in moved if not any(t in k for t in MAY_MOVE)]
    check("nothing-outside-the-contract-moves", not stray,
          f"undocumented movement: {stray}" if stray
          else f"every one of the {len(moved)} is on the list")

    # sigma_med is the one equality the whole intervention class rests on, so
    # it is asserted by name as well as by the pattern above.
    check("sigma_med-is-untouched",
          f_unit["sigma_functional/sigma_med_pct"]
          == f_nar["sigma_functional/sigma_med_pct"],
          f"{f_unit['sigma_functional/sigma_med_pct']} unchanged")
    check("reported-sigma-falls-with-the-width",
          f_nar["sigma_window_pct"] < f_unit["sigma_window_pct"],
          f"{f_unit['sigma_window_pct']} -> {f_nar['sigma_window_pct']}")

    # DIRECTION. The farthest rung on each side is where the high-sigma atoms
    # do their work, so narrowing has to cut it.
    far = {}
    for side in ("up", "dn"):
        rungs = [k for k in f_unit
                 if k.startswith(f"barrier_curves/{side}/") and k.endswith("/pct")]
        i = max(rungs, key=lambda k: abs(f_unit[k])).split("/")[2]
        key = f"barrier_curves/{side}/{i}/touch_prob"
        far[side] = (f_unit[key], f_nar[key], f_unit[f"barrier_curves/{side}/{i}/pct"])
    check("far-touch-probability-falls-on-both-sides",
          all(b < a for a, b, _ in far.values()),
          "; ".join(f"{s} at {p}%: {a:.5f} -> {b:.5f}"
                    for s, (a, b, p) in far.items()))

    # R2: exercise the failing branch rather than trusting it exists.
    tampered = dict(unit)
    tampered["barrier_curves"] = {
        side: [{**r, "price": r["price"] + 1.0} for r in rows]
        for side, rows in unit["barrier_curves"].items()}
    t_moved = [k for k in flatten(unit)
               if flatten(unit)[k] != flatten(tampered).get(k)]
    check("gate-can-fail",
          any(any(t in k for t in FROZEN) for k in t_moved),
          f"a tampered price grid moves {len(t_moved)} leaves and the frozen "
          f"check sees it")

    print("dispersion serving gate")
    for n, good, m in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {n}: {m}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok) - len(bad)}/{len(ok)} checks passed")
    if not bad:
        print("all checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""
eval/sample_sensitivity.py
=====================================================================
P2-sample-sensitivity: can THREE training episodes move the neural arm's pooled
QLIKE by as much as the unexplained reproduction gap?

WHY THIS EXISTS

`noctua_v1` missed its published pooled QLIKE by 0.00154, 0.00272 and 0.00366 at
H=1/6/24 while every deterministic teacher returned to within 3e-4. Two
explanations were offered and both are now falsified BY RUNNING THEM:

  * seed or float nondeterminism -- two rebuilds were bit-identical, 528 of 528
    arrays, worst |diff| 0.000e+00;
  * a torch version drift (2.14.0 installed against a 2.13.0+cu130 pin) -- the
    pin was satisfied exactly and the rebuild was AGAIN bit-identical to the
    drifted one, 528 of 528. The "perfect correspondence" between the packages
    that drifted and the arms that moved was a coincidence, and it was mine.

What remains is the corpus: the rebuild has 476,359 h4 episodes against a
published 476,362. That was ruled out earlier on the grounds that the
deterministic teachers are fitted on the same corpus and reproduce to 1e-5 --
**and that argument is wrong**. OLS and GARCH MLE depend on sample composition
SMOOTHLY: remove three of 476,359 rows and the normal equations move by order
1/n. An SGD-trained network depends on it CHAOTICALLY: three fewer rows shifts
every minibatch boundary for the rest of training, so the weights that come out
are not a small perturbation of the weights that would have come out. A
deterministic arm therefore controls the data for other smooth estimators and
NOT for a neural one, which is the hole in R61 as first written.

THE DESIGN

Three episodes are dropped from the EARLIEST anchors in the table, which sit in
the train slice of every fold. The test slices are therefore bit-identical, so
any movement in test QLIKE is a pure training-path effect and not a change in
what is being scored. That is the whole point: it isolates the mechanism.

BANDS, FIXED BEFORE THE RUN

  CONFIRMED  noctua moves >= 0.0010 at some horizon while every deterministic
             teacher moves <= 1e-4. Three episodes are then SUFFICIENT to
             produce the observed gap, R61 is corrected, and the reproduction
             gap is explained without anything further.
  PARTIAL    noctua moves 1e-4 .. 0.0010. The mechanism is real but small, and
             cannot by itself account for 0.00366.
  REFUTED    noctua moves <= 1e-4, like the deterministic arms. The corpus is
             then exonerated too and the cause of the gap is UNKNOWN -- which
             must be reported as unknown, not decorated with a third story.

PREDICTION, recorded before running: CONFIRMED. SGD is chaotic in its sample
order, and the gap's own pattern -- mixed signs, largest at the horizon with the
smallest loss scale -- looks like a different trajectory rather than a bias.

    python -m model.eval.sample_sensitivity
    python -m model.eval.sample_sensitivity --selftest
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.teacher_scorecard import HORIZONS, YEARS, load_oof                 # noqa: E402
from eval.vol_matrix import qlike_vec                                        # noqa: E402

EP = Path("model/artifacts/episodes_h4.parquet")
BASE = Path("model/artifacts/teacher_oof.npz")
N_DROP = 3
BANDS = ((0.0010, "CONFIRMED"), (1e-4, "PARTIAL"), (0.0, "REFUTED"))


def pooled(z, H: int, teacher: str) -> float:
    """Pooled test QLIKE, mirroring teacher_scorecard's gather."""
    rv, sig = [], []
    for y in YEARS:
        kt = f"{y}/{H}/test"
        if f"{kt}/sigma/{teacher}" not in z:
            continue
        s = np.asarray(z[f"{kt}/sigma/{teacher}"], np.float64)
        r = np.asarray(z[f"{kt}/rv"], np.float64)
        ok = np.isfinite(s) & (s > 0)
        if ok.mean() < 0.95:
            continue
        rv.append(r); sig.append(np.where(ok, s, np.nan))
    if not rv:
        return float("nan")
    return float(np.nanmean(qlike_vec(np.concatenate(rv), np.concatenate(sig))))


def band_for(move: float) -> str:
    for thr, name in BANDS:
        if move >= thr:
            return name
    return "REFUTED"


def selftest() -> int:
    checks = []
    checks.append(("bands-are-ordered",
                   [t for t, _ in BANDS] == sorted((t for t, _ in BANDS),
                                                   reverse=True),
                   f"{BANDS}"))
    checks.append(("band-reads", (band_for(0.0037), band_for(5e-4),
                                  band_for(1e-5)) ==
                   ("CONFIRMED", "PARTIAL", "REFUTED"),
                   "0.0037 -> CONFIRMED, 5e-4 -> PARTIAL, 1e-5 -> REFUTED"))
    # the drop must land in train for every fold: earliest anchors, and the
    # test slices must be untouched
    if EP.exists():
        ep = pd.read_parquet(EP, columns=["anchor_ts", "H"])
        keep = ep.sort_values("anchor_ts").index[N_DROP:]
        dropped = ep.drop(index=keep)
        yrs = pd.to_datetime(dropped["anchor_ts"], unit="s", utc=True).dt.year
        checks.append(("dropped-rows-are-earliest-and-pre-test",
                       bool((yrs < min(YEARS)).all()),
                       f"dropping {len(dropped)} rows from year(s) "
                       f"{sorted(set(yrs))}, all before the first test fold "
                       f"{min(YEARS)} -- so test slices cannot change"))
    else:
        checks.append(("dropped-rows-are-earliest-and-pre-test", True,
                       "skipped: episodes_h4 absent"))
    print("sample_sensitivity selftest")
    bad = 0
    for n, ok, d in checks:
        if not ok:
            bad += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {n}: {d}")
    print(f"\n{len(checks) - bad}/{len(checks)} checks passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-sample-sensitivity")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/sample_sensitivity_result.json"))
    ap.add_argument("--scratch", type=Path, required=False)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (EP.exists() and BASE.exists()):
        print("MISSING episodes_h4.parquet or teacher_oof.npz", file=sys.stderr)
        return 2
    scratch = a.scratch or Path("/tmp/ss")
    scratch.mkdir(parents=True, exist_ok=True)

    print(__doc__[__doc__.index("BANDS, FIXED"):__doc__.index("    python")])

    ep = pd.read_parquet(EP)
    order = ep["anchor_ts"].to_numpy().argsort(kind="stable")
    drop_pos = order[:N_DROP]
    keep = np.ones(len(ep), bool); keep[drop_pos] = False
    yrs = sorted(set(pd.to_datetime(ep["anchor_ts"].to_numpy()[drop_pos],
                                    unit="s", utc=True).year))
    print(f"dropping {N_DROP} of {len(ep):,} episodes, anchors in {yrs} "
          f"-- train slice of every fold, so test slices are unchanged\n")

    bak = scratch / "episodes_h4_full.parquet"
    ep.to_parquet(bak, index=False)
    try:
        ep.loc[keep].reset_index(drop=True).to_parquet(EP, index=False)
        out_npz = scratch / "teacher_oof_minus3.npz"
        print(f"rebuilding teacher_oof on {int(keep.sum()):,} episodes ...")
        r = subprocess.run([sys.executable, "-m", "model.eval.teacher_zoo",
                            "--out", str(out_npz)],
                           capture_output=True, text=True, timeout=5400)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
            return 3
    finally:
        pd.read_parquet(bak).to_parquet(EP, index=False)
        print("episodes_h4.parquet restored")

    z0, teachers = load_oof(BASE)
    z1, _ = load_oof(out_npz)
    rows, res = [], {}
    for H in HORIZONS:
        for t in teachers:
            p0, p1 = pooled(z0, H, t), pooled(z1, H, t)
            if not (np.isfinite(p0) and np.isfinite(p1)):
                continue
            rows.append((H, t, p0, p1, abs(p1 - p0)))
    neural = [r for r in rows if "noctua" in r[1]]
    determ = [r for r in rows if "noctua" not in r[1]]
    n_move = max((r[4] for r in neural), default=float("nan"))
    d_move = max((r[4] for r in determ), default=float("nan"))
    print(f"\n{'H':>4} {'teacher':>14} {'full':>10} {'minus 3':>10} {'|move|':>10}")
    print("-" * 54)
    for H, t, p0, p1, m in sorted(rows, key=lambda r: (r[0], -r[4])):
        print(f"{H:>4} {t:>14} {p0:10.5f} {p1:10.5f} {m:10.5f}"
              + ("   <- neural" if "noctua" in t else ""))
    band = band_for(n_move) if d_move <= 1e-4 else "PARTIAL"
    print(f"\nlargest neural move        {n_move:.5f}")
    print(f"largest deterministic move {d_move:.5f}")
    print(f"unexplained gap to match   0.00366 (noctua_v1 at H=24)")
    print(f"\nBAND: {band}")
    res = {"n_drop": N_DROP, "max_neural_move": n_move,
           "max_deterministic_move": d_move, "band": band,
           "gap_to_match": 0.00366,
           "rows": [{"H": H, "teacher": t, "full": p0, "minus3": p1,
                     "move": m} for H, t, p0, p1, m in rows]}
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

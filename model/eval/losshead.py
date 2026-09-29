"""
eval/losshead.py
=====================================================================
The model spends most of its loss budget on the one thing it cannot predict.

THE COLLISION OF TWO MEASUREMENTS

  * `eval/direction.py`: the sign of the BTC return is not predictable at this
    horizon. NOCTUA's own `prob_up` -- which is read off the terminal-return
    head `q_r` -- loses 0.071 nats to a constant, is miscalibrated by a factor
    of 58 relative to its own discrimination, and is deliberately not
    published.

  * Instrumenting the training loop: `pinball_loss(q_r, r)` is the LARGEST
    single term in the objective, roughly 2.4x the stage-A volatility term
    (measured at the validation minimum: a = 0.0654, r = 0.1560, up = 0.1059,
    dn = 0.0985).

  * `eval/firstpassage.py`: 92% of the barrier error lives in the excursion
    SHAPE -- that is, in `q_up` and `q_dn` -- and only 6-8.5% in the
    volatility level.

Put together: the largest share of the gradient budget is spent fitting the
quantity with no measurable signal, while the heads carrying the deployed
task's error get less. That is a hypothesis about capacity allocation, not a
proof, and the honest response is an ablation rather than a rewrite.

WHY q_r CANNOT SIMPLY BE DELETED

`coupling_penalty` enforces the path identities

    m_up >= max(0,  r)        m_dn >= max(0, -r)

which are true on every real path and act as a genuine regulariser on the
excursion tails -- exactly where the seller's risk lives and the data is
thinnest. Deleting `q_r` would delete that constraint too, confounding
"removed a useless head" with "removed a useful regulariser". So the arm is a
WEIGHT on the head's own pinball term, leaving the coupling penalty at full
strength throughout.

A caveat recorded before the run, because it is the obvious way to be fooled:
the instrumented loss breakdown came from a REDUCED configuration (hidden 64,
50k-row subsample) chosen for speed, not the shipped one (hidden 32, full
data). The relative sizes of the loss terms should be robust to that, but the
overfitting signal seen in the same run should NOT be assumed to transfer, and
is not relied on here.

DECISION RULE, fixed before running: adopt a non-default `lam_r` only if it
improves DSC/UNC -- the barrier discrimination metric the product is sold on
-- in at least 5 of 6 folds. Anything less is noise and gets reported as null.

    python -m model.eval.losshead
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import run_fold                                       # noqa: E402
from eval.efficiency import summarise                                     # noqa: E402
from noctua import splits as S                                            # noqa: E402
from noctua.train import load_all                                         # noqa: E402


def causal_sigma_fn(ep, X):
    """The shipped stage-B reference (`serve_consistent`)."""
    H = ep["H"].to_numpy(np.float64)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(H)

    def fn(train_mask: np.ndarray) -> np.ndarray:
        lo, hi = np.quantile(raw[train_mask], [0.005, 0.995])
        return np.maximum(np.clip(raw, lo, hi), 1e-12)

    return fn


def paired(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    return {"mean": float(d.mean()), "wins": int((d > 0).sum()), "n": n,
            "t_like": float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Down-weight the useless head?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--weights", default="1.0,0.25,0.0",
                    help="lam_r values to compare; 1.0 is the shipped default")
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/losshead.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    sig_fn = causal_sigma_fn(ep, X)
    lams = [float(x) for x in a.weights.split(",")]
    print(f"lam_r arms: {lams}  (1.0 = shipped)\n")

    recs = []
    for f in folds:
        line = {"year": f["year"]}
        for lam in lams:
            t0 = time.time()
            out = run_fold(ep, X, f, hidden=a.hidden, seeds=a.seeds,
                           sigma_ref_fn=sig_fn, lam_r=lam)
            if out is None:
                print(f"  {f['year']}  lam_r={lam:<5} SKIPPED")
                continue
            s = summarise(out["rows"])
            s["qlike"] = out["vol"]["noctua"]
            line[f"lam_{lam}"] = s
            print(f"  {f['year']}  lam_r={lam:<5} DSC/UNC {s['DSS']:.5f}  "
                  f"pinball {s['pinball']:.6f}  CRPS {s['crps']:.6f}  "
                  f"QLIKE {s['qlike']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        recs.append(line)

    base = f"lam_{lams[0]}"
    ok = [r for r in recs if base in r]
    print(f"\n{'arm':>10} {'DSC/UNC':>10} {'pinball':>10} {'CRPS':>10} {'QLIKE':>9} "
          f"{'dDSC wins':>10} {'t-like':>7}")
    report = {}
    for lam in lams:
        key = f"lam_{lam}"
        rows = [r for r in ok if key in r]
        if not rows:
            continue
        dss = np.array([r[key]["DSS"] for r in rows])
        pin = np.array([r[key]["pinball"] for r in rows])
        crp = np.array([r[key]["crps"] for r in rows])
        ql = np.array([r[key]["qlike"] for r in rows])
        b = np.array([r[base]["DSS"] for r in rows])
        p = paired(dss - b)
        report[key] = {"DSS": float(dss.mean()), "pinball": float(pin.mean()),
                       "crps": float(crp.mean()), "qlike": float(ql.mean()), **p}
        print(f"{key:>10} {dss.mean():10.5f} {pin.mean():10.6f} {crp.mean():10.6f} "
              f"{ql.mean():9.4f} {p['wins']:>7}/{len(rows):<2} {p['t_like']:+7.2f}")
    print("\n  'dDSC wins' counts folds where this arm beat the shipped lam_r=1.0")
    print("  on barrier discrimination. Decision rule fixed before the run: "
          "adopt only at >= 5/6.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"arms": lams, "folds": recs, "summary": report,
                                 "seeds": a.seeds, "hidden": a.hidden},
                                indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

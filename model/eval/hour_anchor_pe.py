"""
eval/hour_anchor_pe.py
=====================================================================
P4-hour-anchor-pe: P4-hour-anchor-result re-scored PER EPISODE.

The fold bootstrap cannot fail when all six folds share a sign (R88), and the
fold t-interval has little power at n = 6. Brier, log score, pinball and CRPS
are means of per-episode losses, so a moving-block bootstrap over the ~2,000
production nights is a proper interval with real power. This reruns the same
arms (M0s, As, and the mirror Ams -- constructed exactly as eval/hour_anchor.py
builds them) with run_fold's saved curves, scores them with eval/product_score,
and refuses unless the offline battery reproduces run_fold's to 1e-10.

SAME DATA. This is a better ESTIMATOR on the evidence already seen, not new
evidence: it cannot answer the forking-path caveat in P4-hour-anchor-result,
which only the forward holdout frozen on 2026-09-27 can.

    python -m model.eval.hour_anchor_pe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import run_fold                                    # noqa: E402
from eval.direction import mean_ci                                     # noqa: E402
from eval.hour_anchor import (FAC_STRIDE_H, PROD_A, PROD_H,            # noqa: E402
                              fold_anchors, served_log_factor)
from eval.product_score import battery, per_episode                    # noqa: E402
from eval.scale_adopt import barrier_cols                              # noqa: E402
from eval.vol_matrix import block_len_for                              # noqa: E402
from noctua import infer as I                                          # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

EP_METRICS = ("brier", "logs", "pinball", "crps")
CONTRASTS = (("As", "M0s"), ("Ams", "M0s"))
TOL = 1e-10


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="hour anchor, per-episode")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/hour_anchor_pe.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    ts_all = ep["anchor_ts"].to_numpy(np.int64)
    ah = ep["anchor_hour"].to_numpy()
    at19 = (ep.H == PROD_H).to_numpy()
    fac_hours = {(PROD_A - PROD_H + FAC_STRIDE_H * k) % 24 for k in range(4)}
    fac_mask = at19 & np.isin(ah, sorted(fac_hours))
    w = I.BLEND_W
    n_family = len(CONTRASTS) * len(EP_METRICS)
    alpha = 0.05 / n_family
    print(f"P4-hour-anchor-pe   per-episode, family {n_family} -> {100*(1-alpha):.3f}%\n")

    pe = {k: {m: [] for m in EP_METRICS} for k in ("M0s", "As", "Ams")}
    years = []
    for f in folds:
        anc = fold_anchors(ep, X, f)
        sh = {"M0s": np.zeros(len(ep)),
              "As": np.nan_to_num((1 - w) * (anc["A"] - anc["lhc"]))}
        sh["Ams"] = -sh["As"]
        rF = run_fold(ep, X, f, a.hidden, a.seeds, prod_override=fac_mask)
        r0 = run_fold(ep, X, f, a.hidden, a.seeds)
        if rF is None or r0 is None:
            continue
        peF, pe0 = rF["per_episode"], r0["per_episode"]
        ci, ti = pe0["cal_idx"], pe0["test_idx"]
        h_idx = np.concatenate([ci, peF["test_idx"]])
        h_rv = np.concatenate([pe0["rv_cal"], peF["rv"]])
        h_sig = np.concatenate([pe0["sigma_cal"], peF["sigma_med"]])
        okf = True
        for arm in ("M0s", "As", "Ams"):
            lf = served_log_factor(ts_all[ti], ts_all[h_idx], h_rv,
                                   h_sig * np.exp(sh[arm][h_idx]))
            full = sh[arm].copy(); full[ti] += lf
            r1 = run_fold(ep, X, f, a.hidden, a.seeds,
                          post_shift_fn=lambda mask, _mt, _s=full: _s[mask])
            if r1 is None:
                okf = False; break
            p1 = r1["per_episode"]
            if not np.array_equal(p1["test_idx"], ti):
                raise SystemExit("REFUSING: arms scored different episodes")
            cur, M = p1["curves"]["noctua_v2"], p1["M_abs"]
            b, b_run = battery(cur, M), barrier_cols(r1["rows"])
            err = max(abs(b[k] - b_run[k]) for k in b_run)
            if not err < TOL:
                raise SystemExit(f"REFUSING: offline battery differs by {err:.2e}")
            p = per_episode(cur, M)
            for m in EP_METRICS:
                pe[arm][m].append(p[m])
        if okf:
            years.append(f["year"])
            print(f"  fold {f['year']}: guards ok", flush=True)

    n_ep = sum(len(x) for x in pe["M0s"]["brier"])
    L = block_len_for(PROD_H, n_ep)
    out, res = {}, {c: {"better": [], "worse": []} for c in CONTRASTS}
    print(f"\n{n_ep} production nights, block length {L}")
    print(f"{'metric':>8} {'contrast':>9} {'delta (+ = first better)':>26} {'interval':>30}")
    for c, o in CONTRASTS:
        for m in EP_METRICS:
            d = np.concatenate(pe[o][m]) - np.concatenate(pe[c][m])
            lo, hi = mean_ci(d, alpha=alpha, block_len=L)["ci95"]
            out[f"{m}_{c}_vs_{o}"] = {"delta": float(d.mean()), "ci95": [lo, hi],
                                      "rel_pct": float(100 * d.mean() /
                                                       np.concatenate(pe[o][m]).mean())}
            if lo > 0: res[(c, o)]["better"].append(m)
            if hi < 0: res[(c, o)]["worse"].append(m)
            print(f"{m:>8} {c+'-'+o:>9} {d.mean():+26.6f} [{lo:+.6f}, {hi:+.6f}]  "
                  f"({out[f'{m}_{c}_vs_{o}']['rel_pct']:+.2f}%)")
    ra, rm = res[("As", "M0s")], res[("Ams", "M0s")]
    met = len(ra["better"]) >= 2 and not ra["worse"] and not rm["better"]
    print(f"\n--- registered reading: As better on >= 2 of 4 per-episode metrics, worse on none,")
    print(f"    and the mirror better on none")
    print(f"   As  vs M0s: better {ra['better'] or 'nothing'}, worse {ra['worse'] or 'nothing'}")
    print(f"   Ams vs M0s: better {rm['better'] or 'nothing'}, worse {rm['worse'] or 'nothing'}")
    print(f"   -> {'CONFIRMED on the per-episode estimator' if met else 'NOT CONFIRMED'}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"years": years, "alpha": alpha, "n_nights": n_ep,
                                 "ci": out, "results": {f"{c}_vs_{o}": v for (c, o), v in res.items()},
                                 "confirmed": bool(met)}, indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

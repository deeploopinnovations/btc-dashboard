"""
eval/event_window.py
=====================================================================
P2-event-window: does a small event-hour indicator defined in US EASTERN local
time recover the intraday signal that 23 UTC dummies could not pay for?

WHAT IS ALREADY ESTABLISHED

  * `P2-dst-shift-result` -- THE CLOCK. The H=1 footprint shifts by exactly one
    hour with US daylight saving: argmax lag +1, correlation 3.6476 against
    2.5317 at lag 0, permutation p = 0.0005, with an even/odd-ISO-week placebo
    at argmax 0 and the H=24 control at argmax 17.
  * `P2-intraday-basis-result` -- THE COST. 23 UTC hour dummies cost 1.86 % at
    H=1 and 4.05 % at H=6 in pure capacity, and returned 0.12 % with true
    labels. The clock information is real everywhere and worth almost exactly
    the capacity it consumes.

Neither says an aligned feature HELPS. That is this file.

THE HOURS COME FROM THE SCHEDULE, NOT FROM THE ERROR TABLES

The lift tables are computed on TEST-slice errors, so naming a feature after
the hours they pick out would be selection on test. These hours are published
release times in US Eastern -- 8:30 CPI/NFP/PPI, 9:30 equity cash open, 10:00
release slot, 14:00 FOMC statement -- so the anchor hours whose forward window
contains them are ET {8, 9, 10, 14}. Structural facts about the schedule, and
they need NO calendar of dates, which matters because federalreserve.gov,
api.stlouisfed.org and bls.gov all return 403 to CONNECT under this
environment's network policy.

THE SECOND CONTROL IS THE EXPERIMENT

    E1  one column, ET hour in {8, 9, 10, 14}
    E2  four columns, one per ET event hour
    C1  SHUFFLE -- E1's indicator permuted within the fold. Capacity matched.
    C2  WRONG CLOCK -- the same four hours in UTC under the winter mapping,
        {13, 14, 15, 19}, which is precisely the specification error that
        P2-dst-shift diagnoses.

C2 is a rival, not a placebo. If alignment does not matter, C2 does as well as
E1. **The primary claim is E1 > C2**, not E1 > base.

    python -m model.eval.event_window --self-test
    python -m model.eval.event_window
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                           # noqa: E402
from eval.vol_matrix import (UNDEFINED_AT_1W, block_len_for, build_h4_table,  # noqa: E402
                             qlike_vec)
from noctua import baselines as B                                            # noqa: E402
from noctua import infer as I                                                # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.model import BASE_COLS                                           # noqa: E402
from noctua.spec import SHAPE_COLS                                           # noqa: E402
from noctua.train import prepare, train_model                                # noqa: E402

EASTERN = ZoneInfo("America/New_York")
ET_EVENT_HOURS = (8, 9, 10, 14)      # 8:30, 9:30, 10:00, 14:00 Eastern
UTC_WINTER_HOURS = (13, 14, 15, 19)  # the same four under the WINTER mapping
ARMS = ("base", "E1", "E2", "C1", "C2")
PRIMARY_H = (1, 6)
CONTROL_H = (24,)
N_FAMILY = 4                          # 2 arms x 2 primary horizons, fixed now
# stated before the run, from P2-intraday-basis's realised CI half-widths
MDE_PCT = {1: 0.24, 6: 0.47}


def eastern_hour(ts: np.ndarray) -> np.ndarray:
    uniq, inv = np.unique(ts, return_inverse=True)
    h = np.array([datetime.fromtimestamp(int(t), tz=timezone.utc)
                  .astimezone(EASTERN).hour for t in uniq], dtype=int)
    return h[inv]


def add_cols(X: pd.DataFrame, et_h, utc_h, arm, rng):
    """Return (frame, added column names) for `arm`.

    `base` is returned untouched so the reference is the shipped feature set
    exactly rather than a re-derivation of it.
    """
    if arm == "base":
        return X, []
    Z = X.copy()
    if arm == "E2":
        names = [f"ev_et_{h:02d}" for h in ET_EVENT_HOURS]
        for h, n in zip(ET_EVENT_HOURS, names):
            Z[n] = (et_h == h).astype(np.float64)
        return Z, names
    if arm == "C2":
        v = np.isin(utc_h, UTC_WINTER_HOURS).astype(np.float64)
    else:
        v = np.isin(et_h, ET_EVENT_HOURS).astype(np.float64)
        if arm == "C1":
            v = rng.permutation(v)
    Z["ev_block"] = v
    return Z, ["ev_block"]


def self_test() -> int:
    ok = []
    # a winter and a summer timestamp that are the SAME Eastern hour
    win = int(datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc).timestamp())
    sum_ = int(datetime(2024, 7, 15, 13, 0, tzinfo=timezone.utc).timestamp())
    et = eastern_hour(np.array([win, sum_]))
    ok.append(("dst-aware", tuple(et) == (9, 9),
               f"14:00 UTC in January and 13:00 UTC in July are both 09:00 ET "
               f"(got {tuple(et)})"))
    # The fixture MUST span both DST regimes. The first version ran 480
    # consecutive January hours, where the ET and UTC definitions agree by
    # construction, and the wrong-clock check failed for lack of summer rather
    # than for a defect in the code.
    n = 480
    ts = np.concatenate([win + 3600 * np.arange(n // 2),
                         sum_ + 3600 * np.arange(n // 2)])
    et_h, utc_h = eastern_hour(ts), ((ts // 3600) % 24).astype(int)
    X = pd.DataFrame({c: np.zeros(n) for c in sorted(set(SHAPE_COLS) | set(BASE_COLS))})
    rng = np.random.default_rng(0)
    b, nb = add_cols(X, et_h, utc_h, "base", rng)
    ok.append(("base-untouched", b is X and nb == [], "base returns the frame unchanged"))
    e1, n1 = add_cols(X, et_h, utc_h, "E1", rng)
    ok.append(("E1-one-column", len(n1) == 1 and e1.shape[1] == X.shape[1] + 1,
               "E1 adds exactly one column"))
    e2, n2 = add_cols(X, et_h, utc_h, "E2", rng)
    ok.append(("E2-four-columns", len(n2) == 4 and e2.shape[1] == X.shape[1] + 4,
               "E2 adds exactly four"))
    ok.append(("E1-equals-E2-sum",
               np.allclose(e1["ev_block"].to_numpy(),
                           e2[n2].to_numpy().sum(1)),
               "the one-column indicator is the sum of the four"))
    c2, _ = add_cols(X, et_h, utc_h, "C2", rng)
    same = np.array_equal(e1["ev_block"].to_numpy(), c2["ev_block"].to_numpy())
    ok.append(("C2-differs-in-summer", not same,
               "the wrong-clock control disagrees with the ET one somewhere"))
    c1, _ = add_cols(X, et_h, utc_h, "C1", rng)
    ok.append(("C1-same-rate",
               abs(c1["ev_block"].mean() - e1["ev_block"].mean()) < 1e-12
               and not np.array_equal(c1["ev_block"].to_numpy(),
                                      e1["ev_block"].to_numpy()),
               "the shuffle keeps the marginal exactly and breaks the alignment"))
    # the path that broke intraday_basis: prepare() must accept the new columns
    epF = pd.DataFrame({"H": np.full(n, 6.0), "RV": np.full(n, 0.02),
                        "R": rng.normal(size=n) * 0.02,
                        "M_up": np.abs(rng.normal(size=n)) * 0.02,
                        "M_dn": -np.abs(rng.normal(size=n)) * 0.02})
    for arm in ARMS:
        try:
            Z, names = add_cols(X, et_h, utc_h, arm, rng)
            tr, _ = prepare(epF, Z, np.ones(n, bool),
                            shape_cols=list(SHAPE_COLS) + names)
            good = tr["Xs"].shape[1] == len(SHAPE_COLS) + len(names)
            msg = f"stage-B width {tr['Xs'].shape[1]}"
        except Exception as exc:                                   # noqa: BLE001
            good, msg = False, f"{type(exc).__name__}: {exc}"
        ok.append((f"prepare-{arm}", good, msg))
    print("event_window self-test")
    for nm, g, m in ok:
        print(f"  [{'ok ' if g else 'FAIL'}] {nm}: {m}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok)-len(bad)}/{len(ok)} checks passed")
    return 1 if bad else 0


def run_arm(ep, X, fold, H, arm, et_h, utc_h, hidden, seeds, rng, seed0=0):
    at_h = (ep.H == H).to_numpy()
    cols40 = [c for c in X.columns if c not in UNDEFINED_AT_1W]
    Xu, names = add_cols(X[cols40], et_h, utc_h, arm, rng)
    sc = list(SHAPE_COLS) + names
    fin = np.isfinite(Xu.to_numpy(np.float64)).all(1)
    Hall = ep.H.to_numpy(np.float64)
    yall = B.har_target(ep.RV.to_numpy(), Hall)
    m_tr = fold["train"] & fin & at_h
    m_va = fold["calib"] & fin & at_h
    m_te = fold["test"] & fin & at_h
    if m_tr.sum() < 2000 or m_va.sum() < 300 or m_te.sum() < 100:
        return None
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)
    lo, hi = np.quantile(raw[m_tr], [0.005, 0.995])
    sref = np.maximum(np.clip(raw, lo, hi), 1e-12)
    tr, stds = prepare(ep, Xu, m_tr, shape_cols=sc, sigma_ref=sref[m_tr])
    w = S.sample_weights(ep, m_tr)
    va, _ = prepare(ep, Xu, m_va, *stds, shape_cols=sc, sigma_ref=sref[m_va])
    ols = B.OLS(BASE_COLS).fit(pd.DataFrame(tr["Xb"], columns=BASE_COLS),
                               tr["y"].astype(np.float64), w)
    bl = B.fit_vol_baselines(Xu[m_tr], yall[m_tr], w)
    # `seed0` offsets the ensemble so an independent seed SET can be drawn.
    # P2-seed-variance needs several complete ensembles, not several seeds:
    # the quantity in question is the variance of a 3-seed fit, which a single
    # wider ensemble would average away rather than measure.
    models = [train_model(tr, w, va, hidden=hidden, epochs=40, seed=seed0 + k,
                          verbose=False, ols_beta=ols.beta)[0] for k in range(seeds)]
    d, _ = prepare(ep, Xu, m_te, *stds, shape_cols=sc)
    lp = bl["log_har_cal"].predict(Xu[m_te])
    sg = np.mean([I.predict(m, d, har_logvol=lp)["sigma_med"] for m in models], 0)
    return {"rv": ep.RV.to_numpy()[m_te], "sigma": np.asarray(sg, np.float64)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-event-window")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/event_window.json"))
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    ep, X = build_h4_table(a.artifacts)
    ts = ep.anchor_ts.to_numpy(np.int64)
    et_h, utc_h = eastern_hour(ts), ((ts // 3600) % 24).astype(int)
    folds = S.walk_forward_folds(ep)
    alpha = 0.05 / N_FAMILY
    print(f"P2-event-window   ET event hours {ET_EVENT_HOURS}   "
          f"UTC winter mapping {UTC_WINTER_HOURS}")
    print(f"family {N_FAMILY} -> {100*(1-alpha):.2f}% intervals")
    print("MDE stated before the run: "
          + ", ".join(f"H={k} about {v:.2f}%" for k, v in MDE_PCT.items()))
    print("THE PRIMARY CLAIM IS E1 > C2 (right clock vs wrong clock), "
          "not E1 > base.\n")

    out = {"family_size": N_FAMILY, "alpha": alpha, "mde_pct": MDE_PCT,
           "et_hours": list(ET_EVENT_HOURS), "utc_hours": list(UTC_WINTER_HOURS),
           "horizons": {}}
    for H in list(PRIMARY_H) + list(CONTROL_H):
        acc = {k: [] for k in ARMS}
        for f in folds:
            t0 = time.time()
            rng = np.random.default_rng(5000 + f["year"])
            for arm in ARMS:
                r = run_arm(ep, X, f, H, arm, et_h, utc_h, a.hidden, a.seeds, rng)
                if r is not None:
                    acc[arm].append(r)
            print(f"  H={H:>3} fold {f['year']}  ({time.time()-t0:.0f}s)", flush=True)
        if not acc["base"]:
            continue
        rv = np.concatenate([x["rv"] for x in acc["base"]])
        sig = {k: np.concatenate([x["sigma"] for x in acc[k]]) for k in ARMS
               if acc[k]}
        q = {k: qlike_vec(rv, v) for k, v in sig.items()}
        L = block_len_for(H, len(rv))
        tag = "CONTROL HORIZON, nothing may clear" if H in CONTROL_H else "PRIMARY"
        print("\n" + "=" * 92)
        print(f"H = {H}h   {len(rv):,} test episodes   blocks of {L}   [{tag}]")
        print("=" * 92)
        print(f"{'arm':>6} {'QLIKE':>9} {'vs base':>10} {'rel %':>8}   paired CI vs base")
        row = {}
        for arm in ARMS:
            if arm not in q:
                continue
            if arm == "base":
                print(f"{arm:>6} {np.nanmean(q[arm]):9.5f} {'—':>10} {'—':>8}"
                      f"   (reference)")
                row[arm] = {"qlike": float(np.nanmean(q[arm]))}
                continue
            d_ = q["base"] - q[arm]
            g = np.isfinite(d_)
            ci = mean_ci(d_[g], alpha=alpha, block_len=L)
            rel = 100 * float(np.nanmean(d_)) / float(np.nanmean(q["base"]))
            note = {"C1": "  <- SHUFFLE, must not clear",
                    "C2": "  <- WRONG CLOCK, the rival"}.get(arm, "")
            print(f"{arm:>6} {np.nanmean(q[arm]):9.5f} {np.nanmean(d_):+10.5f} "
                  f"{rel:+8.2f}   [{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]"
                  + ("  CLEARS" if ci["ci95"][0] > 0 else "") + note)
            row[arm] = {"qlike": float(np.nanmean(q[arm])),
                        "vs_base": float(np.nanmean(d_)), "rel_pct": rel,
                        "ci": list(ci["ci95"]), "clears": bool(ci["ci95"][0] > 0)}
        if "E1" in q and "C2" in q:
            d_ = q["C2"] - q["E1"]
            g = np.isfinite(d_)
            ci = mean_ci(d_[g], alpha=alpha, block_len=L)
            cl = bool(ci["ci95"][0] > 0)
            print(f"\n  THE PRIMARY CLAIM  E1 vs C2 (right clock vs wrong): "
                  f"{np.nanmean(d_):+.5f}  "
                  f"[{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]  "
                  f"{'CLEARS' if cl else 'does not clear'}")
            row["E1_vs_C2"] = {"gain": float(np.nanmean(d_)),
                               "ci": list(ci["ci95"]), "clears": cl}
        if "C1" in row and row["C1"].get("clears"):
            print("  CONTROL BREACH: the shuffled indicator clears. Capacity, "
                  "not alignment; every arm falls.")
        out["horizons"][str(H)] = row
        print()

    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

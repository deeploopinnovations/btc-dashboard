"""
eval/exogenous.py
=====================================================================
P3-exogenous-dvol: does the options market's forward volatility expectation
carry information that OHLCV and the clock do not?

Design is pre-registered in `model/research/PLAN_EXOGENOUS.md` and in the ledger
entry `P3-exogenous-dvol`, both committed BEFORE this file ran. Read the rule
there; this module only executes it.

THE ONE-LINE VERSION

NOCTUA sees 42 columns, all of them OHLCV plus the clock, so when the market
moves because of something that happened in the world it observes the move and
never the cause. It cannot be told "someone said something". It CAN be told "the
options market just repriced volatility upward relative to what was realized",
which is the part of such an event a volatility model can act on -- and that is
`x_ivrv`, measured at the anchor from hours strictly before it.

WHY DVOL AND NOT NEWS

GDELT is unreachable from this environment (organization network policy). It was
also the weaker instrument: `P3-attention-feature` pre-registered the objection
that articles are written BECAUSE the price moved, making attention a candidate
lagging indicator. DVOL is a forward expectation by construction.

THE CONTROLS ARE THE POINT

Three previous Phase 2 features failed, and two level corrections failed by
producing a uniform gain that looked like information. So:

  * `D1-shuf` permutes the same column within the fold -- identical capacity,
    identical marginal distribution, zero information. A gain the shuffle
    reproduces is not adopted at any size.
  * `D1-lag` builds the feature from a-169 instead of a-1. If week-stale IV-RV
    does as well, the feature is a regime proxy and not news.
  * The spike bucket must gain MORE than the calm bucket, or the hypothesis is
    recorded as failed even if pooled QLIKE clears.

    python -m model.eval.exogenous --self-test
    python -m model.eval.exogenous
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                           # noqa: E402
from eval.vol_matrix import (HORIZONS, UNDEFINED_AT_1W, block_len_for,       # noqa: E402
                             build_h4_table, qlike_vec)
from noctua import baselines as B                                            # noqa: E402
from noctua import infer as I                                                # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.model import BASE_COLS                                           # noqa: E402
from noctua.train import prepare, train_model                                # noqa: E402

DVOL = Path("data/newdata/dvol_btc.parquet")
FUND = Path("data/newdata/funding_btc.parquet")

HOURS_PER_YEAR = 365.0 * 24.0
CHG_LAG = 24                 # x_dvol_chg compares a-1 against a-25
STALE_LAG = 168              # D1-lag reads a-169

EXO_COLS = {
    "base":    [],
    "D1":      ["x_ivrv"],
    "D2":      ["x_ivrv", "x_dvol_chg", "x_fund"],
    "D1-shuf": ["x_ivrv"],
    "D2-shuf": ["x_ivrv", "x_dvol_chg", "x_fund"],
    "D1-lag":  ["x_ivrv"],
}
LIVE_ARMS = ("D1", "D2")
# D2-shuf was ADDED AFTER the first run, and only a control may be added that
# way. D1-shuf is capacity-matched to D1's single column; D2 adds three, so D2's
# pass at H=168 had no capacity-matched control at all. A control can only sink a
# result, never raise one, so adding it cannot inflate the family -- which is
# exactly why the pre-registration puts controls outside it.
CONTROL_ARMS = ("D1-shuf", "D2-shuf", "D1-lag")
N_FAMILY = len(HORIZONS) * len(LIVE_ARMS)       # 8, pre-registered


# ------------------------------------------------------------------ series
def hourly_series(path: Path, value_col: str) -> dict:
    """{hour_index: value} for finite values only. Hour index = ts // 3600."""
    d = pd.read_parquet(path)
    h = (d["ts"].to_numpy().astype(np.int64) // 3600)
    v = d[value_col].to_numpy(np.float64)
    ok = np.isfinite(v)
    return dict(zip(h[ok].tolist(), v[ok].tolist()))


def lookup(series: dict, hours: np.ndarray) -> np.ndarray:
    return np.fromiter((series.get(int(h), np.nan) for h in hours),
                       np.float64, len(hours))


def build_exo(ep: pd.DataFrame, X: pd.DataFrame, dvol: dict,
              fund: dict) -> pd.DataFrame:
    """The exogenous columns, read at hours STRICTLY BEFORE the anchor.

    `a1` is the hour index of (anchor - 1h). Nothing here ever reads the anchor
    hour itself or any later hour, which is the project's no-lookahead contract
    and is proved by `self_test`'s corruption probe rather than asserted.
    """
    a = ep["anchor_ts"].to_numpy().astype(np.int64) // 3600
    a1 = a - 1
    dv = lookup(dvol, a1)                       # annualised vol in percent
    # DVOL as a LOG HOURLY VOL RATE, the same unit as har_target's domain
    log_iv_h = np.log(np.maximum(dv, 1e-9) / 100.0) - 0.5 * np.log(HOURS_PER_YEAR)
    har1d = X["har_1d"].to_numpy(np.float64)    # log hourly realized vol, 24h
    out = pd.DataFrame(index=X.index)
    out["x_ivrv"] = log_iv_h - har1d
    dv_prev = lookup(dvol, a1 - CHG_LAG)
    out["x_dvol_chg"] = (np.log(np.maximum(dv, 1e-9))
                         - np.log(np.maximum(dv_prev, 1e-9)))
    out["x_fund"] = lookup(fund, a1)
    # the week-stale variant, for the D1-lag control
    dv_stale = lookup(dvol, a1 - STALE_LAG)
    out["x_ivrv_stale"] = (np.log(np.maximum(dv_stale, 1e-9) / 100.0)
                           - 0.5 * np.log(HOURS_PER_YEAR)) - har1d
    return out


def common_mask(exo: pd.DataFrame) -> np.ndarray:
    """Episodes every arm can use -- so every arm scores the SAME episodes.

    Includes the columns only the CONTROLS need. If the controls were allowed a
    different sample the control would not be a control (R47, R50).
    """
    need = ["x_ivrv", "x_dvol_chg", "x_fund", "x_ivrv_stale"]
    return np.isfinite(exo[need].to_numpy(np.float64)).all(axis=1)


def arm_frame(Xb: pd.DataFrame, exo: pd.DataFrame, arm: str,
              fold_mask: np.ndarray, rng) -> pd.DataFrame:
    """The 40 base columns plus whatever this arm adds."""
    cols = EXO_COLS[arm]
    if not cols:
        return Xb.copy()
    add = exo[cols].copy()
    if arm == "D1-lag":
        add = add.rename(columns={"x_ivrv": "_tmp"})
        add["_tmp"] = exo["x_ivrv_stale"].to_numpy()
        add = add.rename(columns={"_tmp": "x_ivrv"})
    if arm in ("D1-shuf", "D2-shuf"):
        # permute WITHIN the fold's usable episodes: same capacity, same
        # marginal distribution, no alignment to the episode. Each column gets
        # its OWN permutation, so the joint structure between them is destroyed
        # too -- a shared permutation would preserve their correlation and leave
        # a multi-column arm partly informative.
        idx = np.flatnonzero(fold_mask)
        for c in cols:
            v = add[c].to_numpy(np.float64).copy()
            v[idx] = v[rng.permutation(idx)]
            add[c] = v
    return pd.concat([Xb, add[cols]], axis=1)


# ------------------------------------------------------------------- arms
def run_arm(ep, X, exo, fold, H, arm, keep, hidden, seeds, rng):
    at_h = (ep.H == H).to_numpy()
    cols40 = [c for c in X.columns if c not in UNDEFINED_AT_1W]
    Xb = X[cols40]
    usable = keep & at_h
    Xu = arm_frame(Xb, exo, arm, usable & (fold["train"] | fold["calib"]
                                           | fold["test"]), rng)
    sc = cols40 + EXO_COLS[arm]
    fin = np.isfinite(Xu[sc].to_numpy(np.float64)).all(1)
    Hall = ep.H.to_numpy(np.float64)
    yall = B.har_target(ep.RV.to_numpy(), Hall)
    m_tr = fold["train"] & fin & usable
    m_va = fold["calib"] & fin & usable
    m_te = fold["test"] & fin & usable
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
    models = [train_model(tr, w, va, hidden=hidden, epochs=40, seed=k,
                          verbose=False, ols_beta=ols.beta)[0]
              for k in range(seeds)]
    d, _ = prepare(ep, Xu, m_te, *stds, shape_cols=sc)
    lp = bl["log_har_cal"].predict(Xu[m_te])
    preds = [I.predict(m, d, har_logvol=lp) for m in models]
    sg = np.asarray(np.mean([p["sigma_med"] for p in preds], axis=0), np.float64)
    return {"rv": ep.RV.to_numpy()[m_te], "sigma": sg,
            "anchor": ep.anchor_ts.to_numpy()[m_te], "n_cols": len(sc)}


# --------------------------------------------------------------- selftest
def self_test() -> int:
    ok = []
    rng = np.random.default_rng(0)
    n = 600
    a0 = 1_700_000_000 // 3600
    ep = pd.DataFrame({"anchor_ts": (np.arange(n) + a0) * 3600, "H": 6,
                       "RV": np.exp(rng.normal(-4, .4, n))})
    X = pd.DataFrame({"har_1d": rng.normal(-4, .3, n)})
    dvol = {int(h): 50.0 + (h % 17) for h in range(a0 - 400, a0 + n + 400)}
    fund = {int(h): 1e-4 * ((h % 7) - 3) for h in range(a0 - 400, a0 + n + 400)}

    exo = build_exo(ep, X, dvol, fund)
    ok.append(("features-are-finite-where-covered",
               bool(np.isfinite(exo.to_numpy(np.float64)).all()),
               f"{exo.shape[0]} rows, {list(exo.columns)}"))

    # THE LEAK PROBE, per episode. For episode i, corrupt DVOL at its own anchor
    # hour and EVERY hour after it, then check row i alone. It must be per
    # episode: the first version corrupted every hour from the first anchor
    # onward and compared all rows, but a later episode's anchor is after that
    # point, so its a-1 was inside the corrupted region and the feature moved
    # LEGITIMATELY. The probe failed and the code was right -- which is the only
    # reason a probe that can fail is worth writing.
    ah = ep["anchor_ts"].to_numpy().astype(np.int64) // 3600
    probes = [0, 1, n // 2, n - 1]
    moved_any = []
    for i in probes:
        bad = {h: (999.0 if h >= ah[i] else v) for h, v in dvol.items()}
        e2 = build_exo(ep, X, bad, fund)
        moved_any.append(
            abs(e2["x_ivrv"].to_numpy()[i] - exo["x_ivrv"].to_numpy()[i])
            + abs(e2["x_dvol_chg"].to_numpy()[i]
                  - exo["x_dvol_chg"].to_numpy()[i]))
    ok.append(("no-lookahead-per-episode", max(moved_any) == 0.0,
               f"for episodes {probes}, corrupting DVOL at that episode's own "
               f"anchor hour and every later hour leaves its features "
               f"bit-identical (max move {max(moved_any):.3e})"))

    # and the probe must be able to FAIL: corrupting a-1 MUST move the feature
    worse = {h: (999.0 if h == ah[n // 2] - 1 else v) for h, v in dvol.items()}
    e3 = build_exo(ep, X, worse, fund)
    moved = not np.isclose(e3["x_ivrv"].to_numpy()[n // 2],
                           exo["x_ivrv"].to_numpy()[n // 2])
    ok.append(("leak-probe-can-fail", moved,
               "corrupting the ONE hour the feature is allowed to read (a-1) "
               "does move it, so the probe is not vacuous"))

    # the shuffle must preserve the multiset and break the alignment
    Xb = X.copy()
    m = np.ones(n, bool)
    s1 = arm_frame(Xb, exo, "D1-shuf", m, np.random.default_rng(1))
    base = exo["x_ivrv"].to_numpy()
    ok.append(("shuffle-preserves-distribution",
               np.allclose(np.sort(s1["x_ivrv"].to_numpy()), np.sort(base)),
               "same multiset of values"))
    ok.append(("shuffle-breaks-alignment",
               not np.allclose(s1["x_ivrv"].to_numpy(), base),
               "values no longer sit on their own episodes"))

    # D1-lag must actually use the stale column
    sl = arm_frame(Xb, exo, "D1-lag", m, rng)
    ok.append(("lag-arm-uses-the-stale-column",
               np.allclose(sl["x_ivrv"].to_numpy(),
                           exo["x_ivrv_stale"].to_numpy(), equal_nan=True)
               and not np.allclose(sl["x_ivrv"].to_numpy(), base),
               "D1-lag's x_ivrv is the a-169 version, not the a-1 one"))

    # the common mask must require the CONTROL columns too, or the control
    # would be scored on a different sample than the arm it controls
    gappy = exo.copy()
    gappy.loc[gappy.index[:5], "x_ivrv_stale"] = np.nan
    ok.append(("common-mask-requires-control-columns",
               int(common_mask(gappy).sum()) == n - 5,
               "dropping 5 stale values removes 5 episodes from EVERY arm"))

    ok.append(("family-size-matches-the-preregistration", N_FAMILY == 8,
               f"{len(HORIZONS)} horizons x {len(LIVE_ARMS)} live arms = "
               f"{N_FAMILY}; controls are not family members"))

    print("exogenous self-test")
    for nm, good, msg in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {nm}: {msg}")
    bad_n = [o for o in ok if not o[1]]
    print(f"\n{len(ok)-len(bad_n)}/{len(ok)} checks passed")
    return 1 if bad_n else 0


# ------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3-exogenous-dvol")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--arms", nargs="+",
                    default=["base", *LIVE_ARMS, *CONTROL_ARMS])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/exogenous_result.json"))
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    ep, X = build_h4_table(a.artifacts)
    dvol = hourly_series(DVOL, "volatility")
    fund = hourly_series(FUND, "interest_1h")
    exo = build_exo(ep, X, dvol, fund)
    keep = common_mask(exo)
    alpha = 0.05 / N_FAMILY
    print(f"P3-exogenous-dvol   family {N_FAMILY} -> "
          f"{100*(1-alpha):.3f}% intervals")
    print(f"common sample: {keep.sum():,} of {len(ep):,} episodes "
          f"({100*keep.mean():.1f}%) -- every arm, base included, is fitted and "
          f"scored on exactly these\n")
    folds = S.walk_forward_folds(ep)
    out = {"n_family": N_FAMILY, "alpha": alpha,
           "common_episodes": int(keep.sum()), "horizons": {}}

    for H in a.horizons:
        got = {}
        for arm in a.arms:
            parts = []
            for f in folds:
                r = run_arm(ep, X, exo, f, H, arm, keep, a.hidden, a.seeds,
                            np.random.default_rng(20260913 + f["year"]))
                if r:
                    parts.append(r)
            if not parts:
                continue
            got[arm] = {k: np.concatenate([p[k] for p in parts])
                        for k in ("rv", "sigma", "anchor")}
        if "base" not in got:
            print(f"H={H}: base arm unavailable, skipping")
            continue
        print("=" * 92)
        print(f"H = {H}h   {len(got['base']['rv']):,} test episodes")
        print("=" * 92)
        rv0 = got["base"]["rv"]
        q0 = qlike_vec(rv0, got["base"]["sigma"])
        per_ep = {}
        hi5 = rv0 >= np.quantile(rv0, 0.95)
        print(f"{'arm':>9} {'pooled':>9} {'vs base':>10} {'rel %':>7} "
              f"{'spike':>9} {'calm':>9} {'spike rel%':>10} "
              f"{'paired CI vs base':>28}")
        row = {}
        for arm, d in got.items():
            if not np.array_equal(d["anchor"], got["base"]["anchor"]):
                raise SystemExit(f"REFUSING: arm {arm} scored different "
                                 f"episodes than base -- the pairing is void")
            q = qlike_vec(d["rv"], d["sigma"])
            per_ep[arm] = q                  # kept for the arm-vs-control test
            dd = q0 - q                      # positive = arm better than base
            g = np.isfinite(dd)
            L = block_len_for(H, int(g.sum()))
            ci = mean_ci(dd[g], alpha=alpha, block_len=L)
            sp_rel = 100 * (np.nanmean(q0[hi5]) - np.nanmean(q[hi5])) / \
                np.nanmean(q0[hi5])
            cl_rel = 100 * (np.nanmean(q0[~hi5]) - np.nanmean(q[~hi5])) / \
                np.nanmean(q0[~hi5])
            clears = bool(ci["ci95"][0] > 0)
            row[arm] = {"pooled": float(np.nanmean(q)),
                        "delta": float(np.nanmean(dd)),
                        "rel_pct": float(100 * np.nanmean(dd) /
                                         np.nanmean(q0)),
                        "spike": float(np.nanmean(q[hi5])),
                        "calm": float(np.nanmean(q[~hi5])),
                        "spike_rel_pct": float(sp_rel),
                        "calm_rel_pct": float(cl_rel),
                        "ci": [float(ci["ci95"][0]), float(ci["ci95"][1])],
                        "clears": clears, "block_len": int(L),
                        "n_cols": int(row.get(arm, {}).get("n_cols", 0)) or None}
            mark = "" if arm == "base" else (
                f"[{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]"
                + ("  CLEARS" if clears else ""))
            print(f"{arm:>9} {np.nanmean(q):9.5f} "
                  f"{np.nanmean(dd):+10.5f} "
                  f"{100*np.nanmean(dd)/np.nanmean(q0):+7.2f} "
                  f"{np.nanmean(q[hi5]):9.5f} {np.nanmean(q[~hi5]):9.5f} "
                  f"{sp_rel:+10.2f} {mark:>28}")

        # the pre-registered rule, applied here rather than in prose
        verdicts = {}
        # each live arm is judged against the shuffle of its OWN width
        MATCHED = {"D1": "D1-shuf", "D2": "D2-shuf"}
        for arm in LIVE_ARMS:
            if arm not in row:
                continue
            shuf = row.get(MATCHED[arm], {})
            r = row[arm]
            why = []
            if not r["clears"]:
                why.append("interval does not clear zero favourably")
            if not shuf:
                why.append(f"NO capacity-matched control was run for this arm "
                           f"({MATCHED[arm]} absent) -- the result is "
                           f"unprotected and may not be claimed")
            elif shuf.get("clears"):
                why.append(f"the {MATCHED[arm]} control also clears "
                           f"({shuf['delta']:+.5f}) -- any gain here is "
                           f"capacity or noise, not information")
            elif r["clears"] and r["delta"] <= shuf.get("delta", -1e9):
                why.append(f"the control does not clear but still matches or "
                           f"beats the arm ({shuf['delta']:+.5f} vs "
                           f"{r['delta']:+.5f})")
            if r["clears"] and r["spike_rel_pct"] <= r["calm_rel_pct"]:
                why.append(f"gain is NOT spike-concentrated "
                           f"(spike {r['spike_rel_pct']:+.2f}% vs calm "
                           f"{r['calm_rel_pct']:+.2f}%) -- pre-registered as a "
                           f"failed hypothesis even when QLIKE clears")
            verdicts[arm] = {"passes": not why, "reasons": why}
            print(f"\n  {arm}: {'PASSES the pre-registered rule' if not why else 'FAILS'}")
            for w in why:
                print(f"      - {w}")
        # THE DECIDING CONTRAST, and the one the first version omitted.
        # "the arm clears against base and the control does not" is two separate
        # tests against a third thing; it can be satisfied while the arm and its
        # control are statistically indistinguishable from EACH OTHER. Only a
        # paired interval between them answers the question the control was
        # built to ask.
        for arm in LIVE_ARMS:
            ctl = {"D1": "D1-shuf", "D2": "D2-shuf"}[arm]
            if arm not in per_ep or ctl not in per_ep:
                continue
            dv = per_ep[ctl] - per_ep[arm]   # positive = arm beats its control
            g = np.isfinite(dv)
            L = block_len_for(H, int(g.sum()))
            ci = mean_ci(dv[g], alpha=alpha, block_len=L)
            beats = bool(ci["ci95"][0] > 0)
            print(f"\n  {arm} vs its OWN-WIDTH control {ctl}: "
                  f"{np.nanmean(dv):+.5f}  CI [{ci['ci95'][0]:+.5f}, "
                  f"{ci['ci95'][1]:+.5f}]  "
                  + ("-> the feature beats same-width noise"
                     if beats else
                     "-> NOT DISTINGUISHABLE from same-width noise; any pass "
                     "against base is hollow"))
            row[arm]["vs_control"] = {"delta": float(np.nanmean(dv)),
                                      "ci": [float(ci["ci95"][0]),
                                             float(ci["ci95"][1])],
                                      "beats_control": beats}
            if arm in verdicts and not beats:
                verdicts[arm]["passes"] = False
                verdicts[arm]["reasons"].append(
                    f"not distinguishable from {ctl} in a direct paired test "
                    f"({np.nanmean(dv):+.5f}, CI [{ci['ci95'][0]:+.5f}, "
                    f"{ci['ci95'][1]:+.5f}])")
                print(f"      -> {arm} verdict downgraded to FAILS")

        if "D1-lag" in row and "D1" in row:
            print(f"\n  regime-proxy check: D1 {row['D1']['delta']:+.5f} vs "
                  f"week-stale D1-lag {row['D1-lag']['delta']:+.5f}"
                  + ("  -> stale does as well, so the feature is a REGIME "
                     "PROXY rather than news"
                     if row["D1-lag"]["delta"] >= row["D1"]["delta"] * 0.8
                     else "  -> freshness matters, consistent with news"))
        out["horizons"][str(H)] = {"arms": row, "verdicts": verdicts}
        # written after EVERY horizon: this run takes hours and a crash at the
        # last horizon must not discard the first three
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(f"  (partial result written to {a.out})\n")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

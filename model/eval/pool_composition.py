"""
eval/pool_composition.py
=====================================================================
P2-pool-composition: does a baseline's accuracy at one horizon depend on which
horizons it was POOLED OVER when it was fitted?

WHERE THIS CAME FROM

`P2-armA-adopt`'s anchor validation refused to score and printed the reason:
har_short refitted on the main episodes table (horizons {6,12,19,24}) scored
QLIKE 0.37951 on the 49,124 shared H=6 test episodes against teacher_oof.npz's
0.41710, fitted on episodes_h4 (horizons {1,6,24,168}). 9.01% apart, and the
artifact number is the har_short that Arm A's H=6 "win over its teacher" was
measured against.

That validation existed to stop a run from proceeding under a false name. It
did, and it turned up something larger than the run it blocked.

WHY THE MECHANISM IS PLAUSIBLE BEFORE ANY NUMBER

har_short's five regressors -- har_1h, har_6h, har_1d, har_5d, har_22d -- are
ANCHOR features. They do not vary with H; the columns that do are seas_1d,
seas_5d, seas_22d, cal_H and cal_weekend_frac. So at a single anchor the same x
row appears once per horizon in the panel, each time paired with a different
target y = log(RV_H) - 0.5*log(H). Pooling H=168 with H=1 asks one set of
coefficients to serve a target whose mean reversion is nearly complete and one
whose has barely started.

WHAT THIS FILE HOLDS FIXED

The TABLE. Every panel is fitted on episodes_h4, so the only thing that varies
is which horizons enter the fit. The main table's own {6,12,19,24} number is
printed alongside as the observation that prompted this and is explicitly NOT
one of the panels: it changes the table as well, and would isolate nothing.

    python -m model.eval.pool_composition
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                           # noqa: E402
from eval.vol_matrix import (UNDEFINED_AT_1W, block_len_for, build_h4_table,  # noqa: E402
                             qlike_vec)
from noctua import baselines as B                                            # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.train import load_all                                            # noqa: E402

TEACHER = "har_short"
ARTIFACT_PANEL = (1, 6, 24, 168)
# `{H} alone` IS THE ARTIFACT'S PANEL and is the reference. teacher_zoo's
# _fit_predict_fold fits every OLS teacher per horizon -- "PER-HORIZON fits.
# Phase 1 established that a baseline fitted on a pooled multi-horizon sample
# is a straw man at the extremes (R39)". P2-pool-composition originally
# labelled {1,6,24,168} "the artifact's", which was wrong; the amendment
# corrects the label, not the contrast (R41).
REFERENCE = "{H} alone"
PANELS = {"{H} alone  (artifact)": None,
          "{1,6,24,168}": (1, 6, 24, 168),
          "{1,6,24}": (1, 6, 24),
          "{6,24}": (6, 24)}
SCORE_H = (6, 24)
N_FAMILY = 6                          # 3 alternative panels x 2 horizons


def panel_scores(ep, X, folds, score_h, panel):
    """Pooled H=score_h test predictions from a har_short fitted on `panel`.

    Returns (rv, sigma_raw, sigma_rescaled). The rescaled series carries each
    fold's OWN calib-fitted c = sqrt(mean(RV^2/sigma^2)), fitted on that
    fold's calib slice and applied to that fold's test slice -- per fold
    because a constant fitted on calib pooled across folds had to be withdrawn
    once already (R44).

    THE RESCALED SERIES IS THE ONE THE VERDICT COMES FROM. QLIKE is minimised
    by the conditional MEAN of variance and an OLS in logs exponentiates below
    it, so changing the fitting panel moves the INTERCEPT and a panel sitting
    at a luckier level scores better while knowing nothing more. garch_normal's
    spike wins at all four horizons were exactly that, and rescaling destroyed
    them.
    """
    Hall = ep["H"].to_numpy(np.float64)
    yall = B.har_target(ep["RV"].to_numpy(), Hall)
    # THE COMPLETENESS MASK MUST BE THE ONE teacher_zoo USED, not the obvious
    # one. seas_1d and seas_5d read a window that runs past the anchor once
    # H > 24d, so NO H=168 row has a complete 42-column record. A mask built
    # over all 42 columns therefore deletes the entire long horizon, and the
    # `artifact {1,6,24,168}` panel silently becomes {1,6,24} -- which is one
    # of the treatments it is supposed to be contrasted against. The first run
    # of this file did exactly that and scored 0.42905 for a panel that could
    # not contain the rows the whole experiment is about. R37: point new code
    # at the column policy the trusted code already uses.
    cols40 = [c for c in X.columns if c not in UNDEFINED_AT_1W]
    fin = np.isfinite(X[cols40].to_numpy(np.float64)).all(1)
    at_h = Hall == score_h
    keep = at_h if panel is None else np.isin(Hall, np.asarray(panel, float))
    rv, sig, sig_c, cs = [], [], [], []
    for f in folds:
        m_fit = f["train"] & fin & keep
        m_te = f["test"] & fin & at_h
        m_ca = f["calib"] & fin & at_h
        if m_fit.sum() < 2000 or m_te.sum() < 100 or m_ca.sum() < 100:
            continue
        bl = B.fit_vol_baselines(X[m_fit], yall[m_fit], S.sample_weights(ep, m_fit))
        sq = np.sqrt(float(score_h))
        s_te = np.exp(np.asarray(bl[TEACHER].predict(X[m_te]), np.float64)) * sq
        s_ca = np.exp(np.asarray(bl[TEACHER].predict(X[m_ca]), np.float64)) * sq
        rv_ca = ep["RV"].to_numpy()[m_ca]
        c = float(np.sqrt(max(np.nanmean(rv_ca ** 2 / np.maximum(s_ca, 1e-12) ** 2),
                              1e-12)))
        rv.append(ep["RV"].to_numpy()[m_te])
        sig.append(s_te)
        sig_c.append(c * s_te)
        cs.append(c)
    if not rv:
        return None, None, None, None
    return (np.concatenate(rv), np.concatenate(sig), np.concatenate(sig_c), cs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-pool-composition")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/pool_composition.json"))
    a = ap.parse_args(argv)

    ep4, X4 = build_h4_table(a.artifacts)
    folds4 = S.walk_forward_folds(ep4)
    alpha = 0.05 / N_FAMILY
    print(f"P2-pool-composition  teacher={TEACHER}  table=episodes_h4 (fixed)")
    print(f"family {N_FAMILY} -> {100*(1-alpha):.2f}% intervals\n")

    out = {"family_size": N_FAMILY, "alpha": alpha, "horizons": {}}
    for H in SCORE_H:
        got = {name: panel_scores(ep4, X4, folds4, H, p_)
               for name, p_ in PANELS.items()}
        base_rv = got[REFERENCE][0]
        if base_rv is None:
            continue
        L = block_len_for(H, len(base_rv))
        print("=" * 96)
        print(f"scored at H = {H}h   {len(base_rv):,} test episodes   blocks of {L}")
        print("=" * 96)
        row = {}
        for mode, si in (("RAW", 1), ("RESCALED, each panel with its own "
                                      "calib-fitted c  <- THE VERDICT COMES "
                                      "FROM HERE", 2)):
            qb = qlike_vec(base_rv, got[REFERENCE][si])
            print(f"  {mode}")
            print(f"  {'fit panel':>22} {'QLIKE':>9} {'vs artifact':>12} "
                  f"{'rel %':>8}   paired CI")
            for name in PANELS:
                rv, sg, sgc, cs = got[name]
                if rv is None:
                    continue
                if not np.array_equal(rv, base_rv):
                    raise SystemExit(
                        f"REFUSING: panel {name!r} scored a different set of "
                        f"episodes than the reference ({len(rv):,} vs "
                        f"{len(base_rv):,}). The contrast would not be paired.")
                q = qlike_vec(rv, (sg, sgc)[si - 1])
                key = f"{name}|{'raw' if si == 1 else 'rescaled'}"
                if name == REFERENCE:
                    print(f"  {name:>22} {np.nanmean(q):9.5f} {'-':>12} "
                          f"{'-':>8}   (reference)"
                          + (f"   c={', '.join(f'{c:.3f}' for c in cs)}"
                             if si == 2 else ""))
                    row[key] = {"qlike": float(np.nanmean(q)),
                                "c_per_fold": [float(c) for c in cs]}
                    continue
                d = qb - q
                g = np.isfinite(d)
                ci = mean_ci(d[g], alpha=alpha, block_len=L)
                rel = 100 * float(np.nanmean(d)) / float(np.nanmean(qb))
                print(f"  {name:>22} {np.nanmean(q):9.5f} {np.nanmean(d):+12.5f} "
                      f"{rel:+8.2f}   [{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]"
                      + ("  CLEARS" if ci["ci95"][0] > 0 else ""))
                row[key] = {"qlike": float(np.nanmean(q)),
                            "vs_artifact": float(np.nanmean(d)), "rel_pct": rel,
                            "ci": list(ci["ci95"]),
                            "clears": bool(ci["ci95"][0] > 0),
                            "c_per_fold": [float(c) for c in cs]}
            print()
        out["horizons"][str(H)] = row

    # The observation that prompted this. A DIFFERENT TABLE, so it is reported
    # and never compared against the panels above -- it varies two things.
    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    print("-" * 84)
    print("the prompting observation, on the MAIN table (horizons {6,12,19,24}).")
    print("NOT a panel: it changes the table too, so it isolates nothing.")
    for H in (6, 24):
        rv, sig, sigc, cs = panel_scores(ep, X, folds, H, (6, 12, 19, 24))
        if rv is None:
            continue
        q = float(np.nanmean(qlike_vec(rv, sig)))
        qc = float(np.nanmean(qlike_vec(rv, sigc)))
        print(f"   main-table har_short at H={H:>3}: raw {q:.5f}  rescaled "
              f"{qc:.5f}  over {len(rv):,} episodes")
        out.setdefault("main_table", {})[str(H)] = {
            "qlike_raw": q, "qlike_rescaled": qc, "n": int(len(rv)),
            "c_per_fold": [float(c) for c in cs]}

    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

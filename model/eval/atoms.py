"""
eval/atoms.py
=====================================================================
Does collapsing Stage B's 32 sigma atoms to one atom actually help, or did
BENCHMARK.md section 9 find a small-sample fluke worth removing a safety
margin for?

THE FINDING BEING TESTED

Stage B is conditioned on `log_sigma` and integrated over 32 sigma ATOMS
drawn from Stage A's predictive distribution -- the mixing integral in
`noctua/infer.py`. On 769 held-out production episodes (H=19, anchor 17:00
UTC, the test split) section 9 measured that swapping those 32 atoms for a
SINGLE atom at Stage A's median sigma improved the barrier Brier score at 4
of 5 barriers and roughly halved calibration error (2% barrier: 3.09pp
against 5.81pp; 3%: 1.47pp against 4.59pp). The suspected mechanism is
Jensen's inequality: P(touch) is convex in sigma over the relevant range, so
averaging the touch probability over a SPREAD of sigmas exceeds the
probability at one representative sigma -- inflating touch probabilities,
which would be a mechanism for the over-forecast bias documented in section
7a and papered over nightly by `serve/adaptive.py`'s shrink factor.

THE PRE-REGISTERED DECISION RULE (BENCHMARK.md section 9, fixed before this
file was run, reproduced verbatim -- this file does not get to renegotiate
it after seeing the numbers):

    Collapse the atoms only if, on the WIDE slice (every H=19 anchor hour,
    ~24x the episodes) split by year:
      (a) the single-atom arm improves mean Brier at the 2% barrier with a
          block-bootstrap CI excluding zero,
      (b) it does not worsen mean Brier at ANY of 0.5/1/3/5%,
      (c) it wins the 2% barrier in at least 5 of 6 years.

Condition (b) is deliberately strict: the atom spread exists to represent
genuine uncertainty, and an average-case win bought by removing a safety
margin at one barrier is the wrong trade for an option seller. If (b) fails
at even ONE barrier, the verdict is DO NOT ADOPT, full stop, even if (a) and
(c) both pass. This file reports the verdict the rule dictates, not the one
that would be more flattering to the mechanism story in the paragraph above.

WHAT A NEGATIVE RESULT LOOKS LIKE

If (a)/(b)/(c) do not all clear -- e.g. the single-atom arm helps at 2% but
costs something at 0.5% or 5%, or the win at 2% does not hold up across
years -- then section 9's headline (roughly halved calibration error) was
real on its own 769 episodes but was bought by narrowing the model's stated
uncertainty in a way that does not generalize to the full anchor grid. That
is a complete, useful answer: it means the 32-atom spread is doing real work
that one small, single-anchor-hour slice was too small (or too lucky) to
show, and the correct move is to leave `serve/` exactly as shipped. A
negative result here is not a failure of this file -- it is the file having
done its job.

A DEVIATION FROM THE LITERAL RULE, FOUND IMMEDIATELY AND REPORTED RATHER
THAN PAPERED OVER

The rule says "5 of 6 years". The task brief pins "the wide slice" to every
H=19 anchor hour INTERSECTED WITH THE TEST SPLIT from `noctua.splits`
(consistent with where section 9's own 769-episode number came from: that
figure is exactly this same test split restricted to anchor_hour=17). But
`noctua.splits.time_splits`'s TEST split begins 2024-07-01 and the episodes
artifact currently ends 2026-08-09 -- so the wide test slice spans exactly
THREE calendar years (2024 partial, 2025 full, 2026 partial), not six. Six
is the walk-forward fold count used elsewhere in this benchmark (2021-2026,
see `noctua.splits.walk_forward_folds` and `eval/direction.py`'s "wide"
slice); it is not a property of what is actually held out for THIS
comparison. Stretching the test split backward to manufacture six years
would mean scoring the shipped artifact against data it was TRAINED on
(2021-2022) or used for its causal calibration reference (2023 through
2024-07) -- not an honest out-of-sample test, and explicitly not what the
task brief asked for ("use noctua.splits -- do not invent your own split").

So this file reports the per-year result for the three years that genuinely
exist in the held-out test split, and treats condition (c) as literally
UNSATISFIABLE on this data: the maximum attainable win count is 3, which can
never reach "5 of 6". That by itself is enough to fail the rule as written.
It is reported as exactly that -- a rule that cannot be met on the data
actually available -- rather than silently rescored against an invented
threshold of "3 of 3" or "2 of 3" to make it pass or fail on the numbers
alone. The per-year wins/losses among the three real years are still
reported in full, because they are informative even though they cannot
satisfy the letter of (c).

THE MECHANISM CHECK

Section 9 only ASSERTED the Jensen's-inequality mechanism; it never verified
it. This file does: for the wide test slice, `noctua.infer.touch_prob`
already computes exactly "the average of the 32 individual atom
probabilities" (it sums each atom's own survival function and divides by 32
-- that is its definition), so it is used directly rather than
reimplemented. That average is compared against the touch probability from
a single atom placed at the ARITHMETIC MEAN of the 32 atoms' sigma values
(not the median used for the decision-rule arm -- the mechanism claim is
specifically about Jensen's gap around the mean). If Jensen's inequality is
the mechanism, the average must exceed the value at the mean everywhere, and
the gap should widen as the barrier moves further from spot (deeper into the
convex part of P(touch) as a function of sigma). If the measured gap does
not behave that way -- shrinks with distance, or is not uniformly positive
-- the stated mechanism is wrong even if the accuracy result upstream turns
out to be real for some other reason, and that is reported plainly too.

REUSE, NOT REIMPLEMENTATION

`stage_b_at_sigma` and `BARRIERS` are imported from `eval.oracle_sigma`,
which already builds the single-atom predictive object averaged over the
seed ensemble the same way `NoctuaV2.predict` does. Reimplementing that here
-- even carefully -- is exactly how a comparison like this quietly stops
being apples-to-apples, so it is not reimplemented. `block_bootstrap_ci` is
imported from `eval.direction` for the same reason: episodes at consecutive
anchor hours overlap (a 19-hour window shares 18 hours with its neighbour
one hour later), so an IID confidence interval would be several times too
narrow, and this project already has one correctly-built moving-block
bootstrap (block length n^(1/3)).

A REPLICATION CHECK, BEFORE THE REAL TEST

Before touching the wide slice, this file recomputes section 9's own
769-episode, anchor-17:00-only numbers through the identical code path used
below, and prints them next to the numbers section 9 published. If they do
not match, the rest of the file's output should not be trusted -- so this
check is not decorative.

    python -m model.eval.atoms
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import block_bootstrap_ci                                # noqa: E402
from eval.oracle_sigma import BARRIERS, stage_b_at_sigma                     # noqa: E402
from noctua import infer as I                                                # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.train import load_all                                            # noqa: E402
from serve import runtime as R                                               # noqa: E402

BOOT_SEED = 42


def episode_brier(pred: dict, u: float, hit_up: np.ndarray, hit_dn: np.ndarray):
    """Per-episode Brier components (not averaged), for both barriers.

    Mirrors `oracle_sigma.score` exactly but returns the per-episode array
    rather than only its mean, because the decision rule needs per-year means
    and a block bootstrap over the episode-level difference, neither of which
    can be recovered from a single aggregate number.
    """
    p_up = I.touch_prob(pred, u, up=True)
    p_dn = I.touch_prob(pred, u, up=False)
    b = 0.5 * ((p_up - hit_up) ** 2 + (p_dn - hit_dn) ** 2)
    return p_up, p_dn, b


def replicate_section9(m, ep, X, fin) -> dict:
    """Recompute section 9's 769-episode, anchor-17:00 numbers.

    Same code path as the wide-slice test below (`stage_b_at_sigma`,
    `episode_brier`), just on the narrow production mask instead of the wide
    one. A sanity check that this file's machinery reproduces a published
    result before it is trusted on a new slice.
    """
    sp = S.time_splits(ep)
    te = sp["test"] & fin & S.production_mask(ep)
    Xte = X[te]
    H = ep["H"].to_numpy(np.float64)[te]
    d = m.prepare(Xte, H)
    M_up = np.abs(ep["M_up"].to_numpy(np.float64))[te]
    M_dn = np.abs(ep["M_dn"].to_numpy(np.float64))[te]

    full = m.predict(d)
    single = stage_b_at_sigma(m, d, full["sigma_med"])

    rows = []
    for pct in BARRIERS:
        u = np.log1p(pct / 100.0)
        hu = (M_up >= u).astype(np.float64)
        hd = (M_dn >= u).astype(np.float64)
        pu_c, pd_c, b_c = episode_brier(full, u, hu, hd)
        pu_s, pd_s, b_s = episode_brier(single, u, hu, hd)
        cal_c = 100 * 0.5 * (abs(pu_c.mean() - hu.mean()) + abs(pd_c.mean() - hd.mean()))
        cal_s = 100 * 0.5 * (abs(pu_s.mean() - hu.mean()) + abs(pd_s.mean() - hd.mean()))
        rows.append({"barrier_pct": float(pct),
                     "brier_committee": float(b_c.mean()),
                     "brier_single": float(b_s.mean()),
                     "single_improves": bool(b_s.mean() < b_c.mean()),
                     "cal_pp_committee": float(cal_c), "cal_pp_single": float(cal_s)})
    return {"n": int(te.sum()), "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered test of the atom-collapse finding")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--model", type=Path, default=Path("model/serve/noctua_v2.npz"))
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/atoms.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    fin = np.isfinite(X.to_numpy()).all(1)
    m = R.load_model(str(a.model))

    # ---- replication check, before anything else --------------------------
    rep = replicate_section9(m, ep, X, fin)
    print(f"replication check -- section 9's own 769-episode slice, n={rep['n']:,}")
    print(f"  published: 2%={{committee 5.81pp / single 3.09pp}}  "
          f"3%={{committee 4.59pp / single 1.47pp}} (calibration error, pp)")
    print(f"  {'barrier':>8} {'brier_committee':>16} {'brier_single':>13} "
          f"{'cal_committee':>14} {'cal_single':>11}")
    for r in rep["rows"]:
        print(f"  {r['barrier_pct']:7.1f}% {r['brier_committee']:16.5f} "
              f"{r['brier_single']:13.5f} {r['cal_pp_committee']:14.2f} "
              f"{r['cal_pp_single']:11.2f}")
    print()

    # ---- the wide slice, as the task brief pins it -------------------------
    # "all H==19 anchors (not just anchor_hour==17) intersected with the test
    # split and finite features" -- noctua.splits, nothing invented.
    sp = S.time_splits(ep)
    wide = (ep["H"] == 19).to_numpy()
    te_wide = sp["test"] & wide & fin
    n_wide = int(te_wide.sum())
    years_present = sorted(set(ep.loc[te_wide, "dt"].dt.year.tolist()))
    print(f"wide slice (H=19, all anchor hours, test split, finite features): "
          f"{n_wide:,} episodes ({n_wide / max(rep['n'], 1):.1f}x the 769-episode "
          f"production slice)")
    print(f"years present in the wide test slice: {years_present}  "
          f"({len(years_present)} of the 6 the decision rule assumes)\n")

    Xte = X[te_wide]
    H = ep["H"].to_numpy(np.float64)[te_wide]
    d = m.prepare(Xte, H)
    M_up = np.abs(ep["M_up"].to_numpy(np.float64))[te_wide]
    M_dn = np.abs(ep["M_dn"].to_numpy(np.float64))[te_wide]
    years = ep["dt"].dt.year.to_numpy()[te_wide]

    full = m.predict(d)                              # committee_32atom: what ships
    single = stage_b_at_sigma(m, d, full["sigma_med"])  # point_forecast: 1 atom, median

    barrier_rows = []
    print(f"{'barrier':>8} {'realized':>9} {'32atom':>9} {'1atom':>9} "
          f"{'delta(c-s)':>11} {'95% CI':>22} {'not-worse':>10}")
    for pct in BARRIERS:
        u = np.log1p(pct / 100.0)
        hu = (M_up >= u).astype(np.float64)
        hd = (M_dn >= u).astype(np.float64)
        pu_c, pd_c, b_c = episode_brier(full, u, hu, hd)
        pu_s, pd_s, b_s = episode_brier(single, u, hu, hd)

        # delta = committee - single, per episode. Positive mean = single atom
        # IMPROVES (lower Brier). This is the sign the decision rule's "improves"
        # and "does not worsen" conditions are stated in.
        delta = b_c - b_s
        lo, hi = block_bootstrap_ci(delta, seed=BOOT_SEED)
        not_worse = bool(delta.mean() >= 0.0)

        cal_c = 100 * 0.5 * (abs(pu_c.mean() - hu.mean()) + abs(pd_c.mean() - hd.mean()))
        cal_s = 100 * 0.5 * (abs(pu_s.mean() - hu.mean()) + abs(pd_s.mean() - hd.mean()))
        mean_p_c = 0.5 * (pu_c.mean() + pd_c.mean())
        mean_p_s = 0.5 * (pu_s.mean() + pd_s.mean())
        realized = 0.5 * (hu.mean() + hd.mean())

        per_year = []
        for y in years_present:
            ym = years == y
            dy = delta[ym]
            per_year.append({"year": int(y), "n": int(ym.sum()),
                             "brier_committee": float(b_c[ym].mean()),
                             "brier_single": float(b_s[ym].mean()),
                             "single_wins": bool(dy.mean() > 0.0)})
        wins = sum(1 for r in per_year if r["single_wins"])

        barrier_rows.append({
            "barrier_pct": float(pct), "n": n_wide,
            "realized_touch_freq": float(realized),
            "brier_committee": float(b_c.mean()), "brier_single": float(b_s.mean()),
            "delta_mean": float(delta.mean()), "delta_ci95": [lo, hi],
            "improves_ci_excludes_zero": bool(lo > 0.0),
            "does_not_worsen": not_worse,
            "cal_pp_committee": float(cal_c), "cal_pp_single": float(cal_s),
            "mean_p_committee": float(mean_p_c), "mean_p_single": float(mean_p_s),
            "per_year": per_year, "years_won_by_single": wins,
            "years_total": len(years_present),
        })
        print(f"{pct:7.1f}% {realized:9.4f} {b_c.mean():9.5f} {b_s.mean():9.5f} "
              f"{delta.mean():+11.5f} [{lo:+.5f},{hi:+.5f}] "
              f"{'yes' if not_worse else 'NO':>10}")

    print("\n  delta = committee_32atom Brier - point_forecast(1atom) Brier, per")
    print("  episode, mean +- block-bootstrap 95% CI. Positive means the single")
    print("  atom is BETTER (lower Brier). 'not-worse' is the point estimate used")
    print("  for decision-rule condition (b): NO means the single atom is worse")
    print("  on average at that barrier, which fails the rule regardless of what")
    print("  happens at 2%.\n")

    print("  calibration error, percentage points, and mean predicted P(touch)")
    print("  against realized frequency, so the bias DIRECTION is visible:")
    print(f"  {'barrier':>8} {'realized':>9} {'p_committee':>12} {'p_single':>9} "
          f"{'cal_committee':>14} {'cal_single':>11}")
    for r in barrier_rows:
        print(f"  {r['barrier_pct']:7.1f}% {r['realized_touch_freq']:9.4f} "
              f"{r['mean_p_committee']:12.4f} {r['mean_p_single']:9.4f} "
              f"{r['cal_pp_committee']:14.2f} {r['cal_pp_single']:11.2f}")

    print("\n  per-year (2% barrier), single-atom vs committee:")
    two_pct = next(r for r in barrier_rows if abs(r["barrier_pct"] - 2.0) < 1e-9)
    print(f"  {'year':>6} {'n':>7} {'brier_committee':>16} {'brier_single':>13} {'single wins':>12}")
    for py in two_pct["per_year"]:
        print(f"  {py['year']:6d} {py['n']:7,} {py['brier_committee']:16.5f} "
              f"{py['brier_single']:13.5f} {('yes' if py['single_wins'] else 'no'):>12}")

    # ---- the mechanism: is it actually Jensen? -----------------------------
    print("\nmechanism check -- average of the 32 atoms' own touch probabilities")
    print("vs the touch probability at a single atom placed at the MEAN of the")
    print("32 atoms' sigma (not the median used above -- Jensen's gap is stated")
    print("around the mean):")
    mean_sigma = full["sigma_atoms"].mean(axis=1)
    mean_atom = stage_b_at_sigma(m, d, mean_sigma)

    print(f"{'barrier':>8} {'avg-of-32':>10} {'at-mean':>9} {'gap':>9} "
          f"{'95% CI':>20} {'gap>0':>6}")
    jensen_rows = []
    for pct in BARRIERS:
        u = np.log1p(pct / 100.0)
        p_avg_up = I.touch_prob(full, u, up=True)
        p_avg_dn = I.touch_prob(full, u, up=False)
        p_mean_up = I.touch_prob(mean_atom, u, up=True)
        p_mean_dn = I.touch_prob(mean_atom, u, up=False)
        gap_up = p_avg_up - p_mean_up
        gap_dn = p_avg_dn - p_mean_dn
        gap = 0.5 * (gap_up + gap_dn)
        lo, hi = block_bootstrap_ci(gap, seed=BOOT_SEED)
        jensen_rows.append({"barrier_pct": float(pct),
                            "avg_of_32_mean_p": float(0.5 * (p_avg_up.mean() + p_avg_dn.mean())),
                            "at_mean_atom_mean_p": float(0.5 * (p_mean_up.mean() + p_mean_dn.mean())),
                            "gap_mean": float(gap.mean()), "gap_ci95": [lo, hi],
                            "gap_up_mean": float(gap_up.mean()),
                            "gap_dn_mean": float(gap_dn.mean())})
        print(f"{pct:7.1f}% {0.5 * (p_avg_up.mean() + p_avg_dn.mean()):10.5f} "
              f"{0.5 * (p_mean_up.mean() + p_mean_dn.mean()):9.5f} "
              f"{gap.mean():+9.5f} [{lo:+.5f},{hi:+.5f}] "
              f"{'yes' if lo > 0 else ('no' if hi < 0 else '?'):>6}")

    gaps = [r["gap_mean"] for r in jensen_rows]
    monotone_growth = all(b >= a - 1e-9 for a, b in zip(gaps, gaps[1:]))
    all_positive = all(g > 0 for g in gaps)
    print(f"\n  all barriers gap > 0: {all_positive}   "
          f"gap widens monotonically with barrier distance: {monotone_growth}")
    if all_positive and monotone_growth:
        print("  -> consistent with Jensen's inequality: the atom spread inflates")
        print("     touch probability, and more so for barriers further from spot.")
    else:
        print("  -> NOT the clean Jensen signature. The average-over-atoms vs")
        print("     value-at-mean gap does not behave the way a convex P(touch)")
        print("     predicts, so the stated mechanism should not be asserted as")
        print("     confirmed even where the accuracy numbers above look good.")

    # ---- the verdict, as the rule dictates ---------------------------------
    two_pct_row = next(r for r in barrier_rows if abs(r["barrier_pct"] - 2.0) < 1e-9)
    cond_a = bool(two_pct_row["improves_ci_excludes_zero"])
    cond_b = all(r["does_not_worsen"] for r in barrier_rows)
    failing_b = [r["barrier_pct"] for r in barrier_rows if not r["does_not_worsen"]]
    cond_c_years_available = two_pct_row["years_total"]
    cond_c_wins = two_pct_row["years_won_by_single"]
    cond_c_literal = cond_c_years_available >= 6 and cond_c_wins >= 5
    adopt = cond_a and cond_b and cond_c_literal

    print("\n" + "=" * 72)
    print("VERDICT (the rule dictates this, not preference)")
    print("=" * 72)
    print(f"  (a) improves mean Brier at 2%, CI excludes zero:  {cond_a}   "
          f"(delta {two_pct_row['delta_mean']:+.5f}, "
          f"CI [{two_pct_row['delta_ci95'][0]:+.5f}, {two_pct_row['delta_ci95'][1]:+.5f}])")
    print(f"  (b) does not worsen mean Brier at ANY barrier:     {cond_b}"
          + (f"   FAILS at: {failing_b}%" if failing_b else ""))
    print(f"  (c) wins the 2% barrier in >=5 of 6 years:         {cond_c_literal}   "
          f"(only {cond_c_years_available} years exist in the test split; "
          f"single atom won {cond_c_wins} of {cond_c_years_available} of them)")
    print(f"\n  ADOPT (collapse the atoms)?  {adopt}")
    if not cond_c_literal and cond_c_years_available < 6:
        print("  Condition (c) as literally written cannot be satisfied on this")
        print("  data -- the held-out test split contains 3 calendar years, not 6.")
        print("  That alone is sufficient to fail the rule, independent of (a)/(b).")
    if not cond_b:
        print(f"  Condition (b) fails at {failing_b}% -- by the rule's own stated")
        print("  logic that is disqualifying by itself, regardless of (a) or (c).")
    print("=" * 72)

    out = {
        "n_test": int(rep["n"]), "n_wide": n_wide, "years_present": years_present,
        "replication_check": rep,
        "barriers": barrier_rows,
        "jensen_mechanism": jensen_rows,
        "jensen_all_positive": all_positive, "jensen_monotone": monotone_growth,
        "decision": {
            "condition_a_improves_at_2pct": cond_a,
            "condition_b_no_worsening_anywhere": cond_b,
            "condition_b_failing_barriers_pct": failing_b,
            "condition_c_years_available": cond_c_years_available,
            "condition_c_years_won": cond_c_wins,
            "condition_c_literal_pass": cond_c_literal,
            "adopt": adopt,
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

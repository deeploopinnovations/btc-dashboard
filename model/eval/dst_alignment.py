"""
eval/dst_alignment.py
=====================================================================
P2-dst-alignment: is the intraday error footprint sharper in US EASTERN local
time than in UTC?

THE HYPOTHESIS, AND WHY IT IS SHARP

Every US macro release and market open is scheduled in EASTERN LOCAL TIME and
therefore moves by one hour in UTC twice a year:

    8:30 ET  CPI / NFP / PPI   ->  13:30 UTC in winter, 12:30 UTC in summer
    9:30 ET  equity cash open  ->  14:30 UTC in winter, 13:30 UTC in summer
    2:00 pm ET  FOMC statement ->  19:00 UTC in winter, 18:00 UTC in summer

If scheduled releases drive the footprint, measuring it in UTC must SPLIT each
event across two adjacent hours and measuring it in Eastern must merge each
pair. `P2-event-footprint`'s UTC lift table is consistent with exactly that,
and the reading was recorded BEFORE this ran: the two largest lifts are
ADJACENT (14:00 = 1.788, 13:00 = 1.201) and the second cluster is an adjacent
PAIR (18:00 = 1.533, 19:00 = 1.465).

WHY THE CONTROL IS THE WHOLE EXPERIMENT, AND WHY THE FIRST ONE WAS VACUOUS

"Eastern time concentrates the errors" is not evidence on its own. The
pre-registration's control was all 23 FIXED hour offsets, on the reasoning that
a fixed offset moves both halves of a DST doublet together and can never merge
them. That reasoning is correct and the control built from it is WORTHLESS: a
fixed offset is a pure RELABELLING of the 24 bins, so the bin counts are a
permutation of themselves and chi-square, every lift and the top-K
concentration are all IDENTICAL to UTC by construction. It could not have
returned another answer. That is R2, committed by me, in the control I wrote
specifically to keep myself honest. It is kept in the output, reported as
invariant, and carries no weight.

The control that works is a PLACEBO SCHEDULE. The Eastern clock is not a fixed
offset -- it is a DATE-DEPENDENT one, -5h in winter and -4h in summer, and that
is exactly what lets it merge a doublet. So the comparison is against clocks
with the SAME two offsets and the SAME two-transitions-a-year structure, whose
transition dates are moved by d days. The true schedule is d = 0. If the effect
is daylight saving, d = 0 must sit at the top of that distribution; if the
placebos do just as well, the Eastern reading is about having two offsets at
all, not about when they switch.

THIS IS A DIAGNOSTIC AND CANNOT DEFINE A FEATURE

The lift tables are computed on TEST-slice errors. Naming a feature after the
hours they pick out would be selection on test. A feature motivated by this
takes its definition from the PUBLISHED RELEASE TIMES, which are structural,
and needs its own pre-registration.

    python -m model.eval.dst_alignment
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import chi2 as chi2_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.teacher_scorecard import YEARS                                     # noqa: E402
from eval.vol_matrix import qlike_vec                                        # noqa: E402

TEACHER = "noctua_v1"
EASTERN = ZoneInfo("America/New_York")
TOP_K = 3            # the concentration statistic, fixed before the run
WORST_Q = 0.95       # "worst 5%"


def eastern_hour(ts: np.ndarray) -> np.ndarray:
    """Local hour in America/New_York, DST transitions included.

    Computed per distinct timestamp rather than per episode: the production
    slices repeat the same anchor hours thousands of times and zoneinfo
    conversion is the expensive part.
    """
    uniq, inv = np.unique(ts, return_inverse=True)
    h = np.array([datetime.fromtimestamp(int(t), tz=timezone.utc)
                  .astimezone(EASTERN).hour for t in uniq], dtype=int)
    return h[inv]


def is_summer(ts: np.ndarray, shift_days: int = 0) -> np.ndarray:
    """True where America/New_York is on -4h, with transitions moved by `shift_days`.

    Read from zoneinfo rather than reimplemented, and evaluated at a shifted
    timestamp so a placebo keeps the real schedule's SHAPE -- two transitions a
    year, the same summer/winter proportions -- and only moves when they happen.
    """
    t = ts - int(shift_days) * 86400
    uniq, inv = np.unique(t // 86400, return_inverse=True)
    off = np.array([datetime.fromtimestamp(int(d) * 86400 + 43200, tz=timezone.utc)
                    .astimezone(EASTERN).utcoffset().total_seconds() / 3600.0
                    for d in uniq])
    return (off[inv] == -4.0)


def local_hour(ts: np.ndarray, utc_h: np.ndarray, shift_days: int) -> np.ndarray:
    """Eastern local hour under a schedule whose transitions are moved d days."""
    summer = is_summer(ts, shift_days)
    return (utc_h + np.where(summer, -4, -5)) % 24


def footprint(hour: np.ndarray, worst: np.ndarray) -> dict:
    """Chi-square of the worst-episode hour distribution against all episodes."""
    obs = np.bincount(hour[worst], minlength=24).astype(np.float64)
    allc = np.bincount(hour, minlength=24).astype(np.float64)
    exp = allc * (obs.sum() / max(allc.sum(), 1.0))
    keep = exp > 0
    chi2 = float(np.sum((obs[keep] - exp[keep]) ** 2 / exp[keep]))
    dof = int(keep.sum() - 1)
    lift = np.divide(obs, np.maximum(exp, 1e-12))
    order = np.argsort(-lift)
    top = order[:TOP_K]
    return {"chi2": chi2, "dof": dof, "p": float(chi2_dist.sf(chi2, dof)),
            "lift": lift.tolist(),
            "top_hours": [int(h) for h in top],
            "top_lift": [float(lift[h]) for h in top],
            # THE PRIMARY: what share of the worst episodes the top-K hours hold
            "concentration": float(obs[top].sum() / max(obs.sum(), 1.0)),
            "expected_share": float(exp[top].sum() / max(exp.sum(), 1.0))}


def load(z, H: int):
    ts, q = [], []
    for y in YEARS:
        k = f"{y}/{H}/test"
        if f"{k}/sigma/{TEACHER}" not in z:
            continue
        s = np.asarray(z[f"{k}/sigma/{TEACHER}"], np.float64)
        r = np.asarray(z[f"{k}/rv"], np.float64)
        a = np.asarray(z[f"{k}/anchor_ts"], np.int64)
        ok = np.isfinite(s) & (s > 0) & np.isfinite(r) & (r > 0)
        ts.append(a[ok]); q.append(qlike_vec(r[ok], s[ok]))
    if not ts:
        return None, None
    return np.concatenate(ts), np.concatenate(q)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-dst-alignment")
    ap.add_argument("--oof", type=Path,
                    default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/dst_alignment.json"))
    a = ap.parse_args(argv)
    z = np.load(a.oof)

    out = {"teacher": TEACHER, "top_k": TOP_K, "horizons": {}}
    for H in (1, 24):
        ts, q = load(z, H)
        if ts is None:
            continue
        worst = q >= np.quantile(q, WORST_Q)
        utc_h = ((ts // 3600) % 24).astype(int)
        et_h = eastern_hour(ts)

        f_utc = footprint(utc_h, worst)
        f_et = footprint(et_h, worst)
        # The vacuous control, kept and labelled. A fixed offset permutes the
        # bins, so every statistic here is identical to UTC by construction.
        offs = {k: footprint((utc_h + k) % 24, worst) for k in range(1, 24)}
        best_off = max(offs, key=lambda k: offs[k]["concentration"])
        fixed_invariant = all(
            abs(offs[k]["concentration"] - f_utc["concentration"]) < 1e-12
            for k in offs)

        # THE CONTROL THAT WORKS: same two offsets, same two transitions a
        # year, transition dates moved by d days. d = 0 is the real schedule.
        shifts = [d for d in range(-175, 176, 7) if d != 0]
        plac = {d: footprint(local_hour(ts, utc_h, d), worst) for d in shifts}
        pc = np.array([plac[d]["concentration"] for d in shifts])
        beat_p = [d for d in shifts
                  if plac[d]["concentration"] >= f_et["concentration"]]

        tag = "PRIMARY" if H == 1 else "CONTROL (must be flat)"
        print("=" * 92)
        print(f"H = {H}h   {len(q):,} episodes   worst {100*(1-WORST_Q):.0f}% "
              f"= {int(worst.sum()):,}   [{tag}]")
        print("=" * 92)
        print(f"  {'clock':>22} {'chi2':>10} {'p':>10} "
              f"{'top-3 share':>12}  top-3 hours (lift)")
        for name, f in (("UTC", f_utc), ("America/New_York", f_et)):
            hh = ", ".join(f"{h:02d}h ({l:.2f})"
                           for h, l in zip(f["top_hours"], f["top_lift"]))
            print(f"  {name:>22} {f['chi2']:10.1f} {f['p']:10.3g} "
                  f"{100*f['concentration']:11.2f}%  {hh}")
        print(f"  {'(chance level)':>22} {'':>10} {'':>10} "
              f"{100*f_utc['expected_share']:11.2f}%")
        print(f"\n  VACUOUS CONTROL, kept and labelled: all 23 fixed hour "
              f"offsets give\n  concentration "
              f"{100*offs[best_off]['concentration']:.2f}% -- identical to UTC "
              f"{'(invariant, as it must be)' if fixed_invariant else '(NOT invariant -- bug)'}. "
              f"A fixed offset\n  permutes the bins, so this control could not "
              f"have returned another answer (R2).")

        print(f"\n  PLACEBO SCHEDULES: same two offsets, transitions moved "
              f"+-7..175 days, {len(shifts)} of them")
        print(f"    real schedule      {100*f_et['concentration']:6.2f}%")
        print(f"    placebo mean       {100*pc.mean():6.2f}%   "
              f"sd {100*pc.std():.2f}pp   max {100*pc.max():6.2f}%")
        z_ = (f_et["concentration"] - pc.mean()) / max(pc.std(), 1e-12)
        print(f"    real vs placebos   z = {z_:+.2f}, rank "
              f"{1 + len(beat_p)} of {len(shifts) + 1}")
        verdict = ("REAL SCHEDULE WINS: no placebo matches it"
                   if not beat_p else
                   f"REFUTED: {len(beat_p)} placebo schedule(s) match or beat "
                   f"the real one (shifts {sorted(beat_p)[:6]} days)")
        print(f"\n  control: {verdict}")
        print(f"  Eastern vs UTC concentration: "
              f"{100*f_et['concentration']:.2f}% vs "
              f"{100*f_utc['concentration']:.2f}%  "
              f"({100*(f_et['concentration']-f_utc['concentration']):+.2f} pp)\n")

        out["horizons"][str(H)] = {
            "n": int(len(q)), "utc": f_utc, "eastern": f_et,
            "fixed_offset_control": {
                "invariant": bool(fixed_invariant),
                "concentration": float(offs[best_off]["concentration"]),
                "note": "vacuous: a fixed offset permutes the bins (R2)"},
            "placebo_schedules": {
                "n": len(shifts),
                "mean": float(pc.mean()), "sd": float(pc.std()),
                "max": float(pc.max()),
                "z": float((f_et["concentration"] - pc.mean())
                           / max(pc.std(), 1e-12)),
                "matching_or_beating": sorted(int(d) for d in beat_p)},
            "eastern_wins": bool(not beat_p)}

    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

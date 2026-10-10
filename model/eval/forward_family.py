"""
eval/forward_family.py
=====================================================================
The forward holdouts on overlapping nights are ONE family. research/DATA_USE.md
("Several holdouts on the same forward nights") requires that when more than one
primary is read, the report states how many were read and gives the
Bonferroni-corrected interval beside the 95% one. Until 2026-10-10 no code did
either, and the locks kept only summary intervals, so the family reading could
not be rebuilt later without rescoring a spent holdout (Codex review, PR #14).

Each scorer now writes into its lock, at its one scoring:
  per_night   anchor_ts and every metric's per-night delta (shipped minus
              candidate: positive favours the candidate), so this family
              reading -- or any later procedure that needs the nights, e.g. a
              step-down on a shared resample -- needs no rescoring;
  family      K, alpha / K, each primary's interval at alpha / K from the SAME
              estimator (eval.ci.mean_ci, the primary's own block length), and
              which members had already been scored.

THE FAMILY, fixed 2026-10-10 before any holdout scored, K = 6: every frozen
holdout on forward nights, scored or not. E2c's (2026-08-28) has no scorer yet;
it is counted anyway, because it will be read and leaving it out would shrink K.
iv1d's PASS needs BOTH of its contrasts (an intersection-union test), so each is
read at alpha / K and no further split is due.

WHAT THIS DOES NOT CHANGE: any holdout's registered PASS rule, primary, N_MIN or
arms. The family interval is reported BESIDE the 95% one.

    python -m model.eval.forward_family      # the family table from the locks
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.ci import mean_ci                                        # noqa: E402

ALPHA = 0.05
RESEARCH = Path(__file__).resolve().parents[1] / "research"
#            member                     freeze        lock file in research/
FAMILY = (("E2c-iv-correction",        "2026-08-28", None),
          ("forward_hour_anchor",      "2026-09-27", "forward_hour_anchor_result.json"),
          ("forward_weekend_anchor",   "2026-09-29", "forward_weekend_anchor_result.json"),
          ("forward_dow_anchor",       "2026-09-29", "forward_dow_anchor_result.json"),
          ("forward_clock_dow_anchor", "2026-09-29", "forward_clock_dow_anchor_result.json"),
          ("forward_iv1d_anchor",      "2026-09-29", "forward_iv1d_anchor_result.json"))
K = len(FAMILY)
ALPHA_EACH = ALPHA / K


def bonferroni_ci(d, block_len: int) -> list:
    lo, hi = mean_ci(np.asarray(d, dtype=np.float64), alpha=ALPHA_EACH, block_len=block_len)["ci95"]
    return [float(lo), float(hi)]


def family_block(member: str, primaries: dict, block_len: int, research: Path = RESEARCH) -> dict:
    """`primaries`: each REGISTERED primary's per-night deltas (two for iv1d)."""
    names = [n for n, _, _ in FAMILY]
    if member not in names:
        raise ValueError(f"{member!r} is not in the forward-holdout family {names}")
    before = [n for n, _, f in FAMILY
              if f is not None and n != member and (Path(research) / f).exists()]
    prim = {}
    for k, d in primaries.items():
        ci = bonferroni_ci(d, block_len)
        prim[k] = {"diff": float(np.mean(d)), "ci_family": ci, "clears_at_family_level": ci[0] > 0}
    return {"member": member, "K": K, "members": names, "alpha_each": ALPHA_EACH,
            "block_len": int(block_len),
            "primaries": prim,
            "all_primaries_clear_at_family_level": all(p["clears_at_family_level"] for p in prim.values()),
            "scored_before_this": before, "primaries_read": len(before) + 1,
            "note": ("Bonferroni across the six forward holdouts (research/DATA_USE.md), reported "
                     "BESIDE this holdout's registered 95% rule, which is unchanged. Positive "
                     "favours the candidate.")}


def per_night(rows, deltas: dict) -> dict:
    return {"anchor_ts": [int(e["anchor_ts"]) for e, _ in rows],
            **{k: [float(x) for x in np.asarray(v, dtype=np.float64)] for k, v in deltas.items()}}


def lock_problems(res: dict, member: str, primaries95: dict) -> list:
    """What is wrong with a scored lock's per-night and family blocks; [] if
    nothing. `primaries95`: per-night key -> the lock's {"diff", "ci95"} for it."""
    pn, fam = res.get("per_night"), res.get("family")
    if not pn or not fam:
        return ["no per_night or family block"]
    out, n, ts = [], res["n_nights"], pn.get("anchor_ts", [])
    if len(ts) != n or any(b <= a for a, b in zip(ts, ts[1:])):
        out.append("anchor_ts is not one increasing timestamp per night")
    if fam.get("member") != member or fam.get("K") != K:
        out.append(f"family block is for {fam.get('member')!r} with K {fam.get('K')}")
    if fam.get("block_len") != res.get("block_len"):
        out.append("family block length differs from the primary's")
    if set(fam.get("primaries", {})) != set(primaries95):
        out.append(f"family primaries {sorted(fam.get('primaries', {}))} != {sorted(primaries95)}")
        return out
    for k, s in primaries95.items():
        d = pn.get(k)
        if d is None or len(d) != n:
            out.append(f"{k}: per-night deltas missing or not one per night")
            continue
        if abs(float(np.mean(d)) - s["diff"]) > 1e-12:
            out.append(f"{k}: per-night mean {np.mean(d)} != recorded diff {s['diff']}")
        cf = fam["primaries"][k]["ci_family"]
        if not (cf[0] <= s["ci95"][0] and cf[1] >= s["ci95"][1]):
            out.append(f"{k}: family interval {cf} does not contain the 95% one {s['ci95']}")
        if cf != bonferroni_ci(d, fam["block_len"]):
            out.append(f"{k}: family interval not rebuilt from the stored deltas")
    return out


def report(research: Path = RESEARCH) -> list:
    """Every member, scored or not; each family interval rebuilt from the lock's
    own per-night deltas and compared with what the lock recorded."""
    out = []
    for name, freeze, f in FAMILY:
        p = Path(research) / f if f else None
        if p is None or not p.exists():
            out.append({"member": name, "freeze": freeze, "scored": False,
                        "note": "no scorer yet" if f is None else "not scored yet"})
            continue
        lock = json.loads(p.read_text())
        fam, pn = lock.get("family"), lock.get("per_night")
        if not fam or not pn:
            out.append({"member": name, "freeze": freeze, "scored": True, "matches_lock": None,
                        "note": "lock has no per-night deltas: the family interval cannot be rebuilt"})
            continue
        rebuilt = {k: bonferroni_ci(pn[k], fam["block_len"]) for k in fam["primaries"]}
        out.append({"member": name, "freeze": freeze, "scored": True,
                    "scored_on": lock.get("scored_on"), "primaries_read": fam["primaries_read"],
                    "rebuilt": rebuilt,
                    "matches_lock": all(rebuilt[k] == fam["primaries"][k]["ci_family"] for k in rebuilt),
                    "clears_at_family_level": fam["all_primaries_clear_at_family_level"],
                    "note": ""})
    return out


def main() -> int:
    rows = report()
    n = sum(r["scored"] for r in rows)
    print(f"forward-holdout family, K = {K}, each primary at alpha = {ALPHA_EACH:.5f}; {n} scored")
    for r in rows:
        if not r["scored"]:
            print(f"  {r['member']:<26} frozen {r['freeze']}  {r['note']}")
        elif r.get("matches_lock") is None:
            print(f"  {r['member']:<26} frozen {r['freeze']}  {r['note']}")
        else:
            iv = "  ".join(f"{k} {v[0]:+.6f} .. {v[1]:+.6f}" for k, v in r["rebuilt"].items())
            print(f"  {r['member']:<26} scored {r['scored_on']} (primary #{r['primaries_read']})  {iv}  "
                  f"clears: {r['clears_at_family_level']}  rebuilt == lock: {r['matches_lock']}")
    return 0 if all(r.get("matches_lock") is not False for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

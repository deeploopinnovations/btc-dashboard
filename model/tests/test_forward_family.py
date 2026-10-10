"""
tests/test_forward_family.py
=====================================================================
The forward holdouts on overlapping nights are ONE family (research/DATA_USE.md,
"Several holdouts on the same forward nights"): when more than one primary is
read, the report gives the Bonferroni interval beside the 95% one. Before this
gate, no code computed it, and the lock files kept only summary intervals, so
it could not be rebuilt later without rescoring a spent holdout (Codex review,
PR #14).

THE CONTRACT (eval/forward_family.py)
  THE FAMILY IS THE SIX FROZEN HOLDOUTS, and each scorer's lock is a member.
  THE FAMILY INTERVAL IS mean_ci AT alpha / K with the primary's own block
  length -- the registered estimator, only the level changes.
  IT CONTAINS THE 95% INTERVAL, AND IT CAN CHANGE THE READING (R2): a primary
  that clears at 95% but not at the family level is reported as not clearing.
  THE LOCK RECORDS WHICH OTHER PRIMARIES WERE ALREADY READ.
  THE PER-NIGHT DELTAS ARE STORED, and the report rebuilds every family
  interval from them, bit for bit, without rescoring.
  THE PASS RULE REGISTERED FOR EACH HOLDOUT IS UNCHANGED.

    python model/tests/test_forward_family.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import forward_family as FF                              # noqa: E402
from eval.ci import mean_ci                                        # noqa: E402

CHECKS: list = []


def check(name: str, good: bool, info: str = "") -> None:
    CHECKS.append((name, bool(good), info))


def main() -> int:
    from eval import (forward_clock_dow_anchor, forward_dow_anchor, forward_hour_anchor,
                      forward_iv1d_anchor, forward_weekend_anchor)
    scorers = (forward_hour_anchor, forward_weekend_anchor, forward_dow_anchor,
               forward_clock_dow_anchor, forward_iv1d_anchor)
    locks = {n: f for n, _, f in FF.FAMILY}
    check("family-is-the-six-frozen-holdouts",
          FF.K == 6 and len(locks) == 6 and locks.get("E2c-iv-correction", "x") is None
          and all(locks.get(m.__name__.split(".")[-1]) == m.LOCK.name for m in scorers),
          f"K {FF.K}, members {list(locks)}")
    check("freeze-dates-match-the-scorers",
          all(dict((n, fz) for n, fz, _ in FF.FAMILY)[m.__name__.split(".")[-1]] == m.FREEZE
              for m in scorers))

    rng = np.random.default_rng(7)
    d = rng.normal(0.0002, 0.003, 400)
    L = 8
    want = mean_ci(d, alpha=0.05 / 6, block_len=L)["ci95"]
    got = FF.bonferroni_ci(d, L)
    check("family-interval-is-mean-ci-at-alpha-over-K",
          np.allclose(got, want, rtol=0, atol=0), f"{got} vs {want}")
    c95 = mean_ci(d, alpha=0.05, block_len=L)["ci95"]
    check("family-interval-contains-the-95-interval",
          got[0] <= c95[0] and got[1] >= c95[1], f"95% {c95}, family {got}")

    # R2: a primary that clears at 95% but not at alpha/6 must be reported as
    # NOT clearing at the family level -- otherwise the block cannot fail
    found = None
    for seed in range(200):
        x = np.random.default_rng(seed).normal(2.3 / np.sqrt(300), 1.0, 300)
        lo95 = mean_ci(x, alpha=0.05, block_len=1)["ci95"][0]
        if lo95 > 0 and FF.bonferroni_ci(x, 1)[0] <= 0:
            found = x
            break
    blk = FF.family_block("forward_weekend_anchor", {"brier": found}, 1, research=Path("/nonexistent"))
    check("correction-can-change-the-reading",
          found is not None and blk["primaries"]["brier"]["clears_at_family_level"] is False
          and blk["all_primaries_clear_at_family_level"] is False)

    clear = np.random.default_rng(3).normal(0.5, 1.0, 300)
    blk2 = FF.family_block("forward_iv1d_anchor", {"a": clear, "b": found}, 1,
                           research=Path("/nonexistent"))
    check("every-registered-primary-must-clear (iv1d's two contrasts)",
          blk2["primaries"]["a"]["clears_at_family_level"] is True
          and blk2["all_primaries_clear_at_family_level"] is False)

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "forward_hour_anchor_result.json").write_text(json.dumps({"scored": True}))
        b = FF.family_block("forward_weekend_anchor", {"brier": d}, L, research=tdp)
        check("lock-records-the-primaries-already-read",
              b["scored_before_this"] == ["forward_hour_anchor"] and b["primaries_read"] == 2
              and b["K"] == 6 and abs(b["alpha_each"] - 0.05 / 6) < 1e-15, json.dumps(b)[:200])

        try:
            FF.family_block("forward_unknown", {"brier": d}, L, research=tdp)
            refused = False
        except ValueError:
            refused = True
        check("an-unknown-member-is-refused", refused)

        # store, then rebuild from the lock alone
        ts = 1790000000 + 86400 * np.arange(len(d))
        rows = [({"anchor_ts": int(t)}, None) for t in ts]
        lock = {"scored": True, "block_len": L, "primary_brier": {"diff": float(d.mean())},
                "per_night": FF.per_night(rows, {"brier": d}),
                "family": FF.family_block("forward_weekend_anchor", {"brier": d}, L, research=tdp)}
        (tdp / "forward_weekend_anchor_result.json").write_text(json.dumps(lock))
        pn = lock["per_night"]
        check("per-night-deltas-stored-with-their-nights",
              pn["anchor_ts"] == [int(t) for t in ts] and len(pn["brier"]) == len(d)
              and abs(np.mean(pn["brier"]) - d.mean()) < 1e-15)
        rep = FF.report(research=tdp)
        row = next(r for r in rep if r["member"] == "forward_weekend_anchor")
        check("report-rebuilds-the-family-interval-from-the-lock",
              row["rebuilt"]["brier"] == lock["family"]["primaries"]["brier"]["ci_family"]
              and row["matches_lock"] is True, json.dumps(row)[:200])
        res = {"n_nights": len(d), "block_len": L, **lock,
               "primary_brier": {"diff": float(d.mean()), "ci95": c95}}
        clean = FF.lock_problems(res, "forward_weekend_anchor", {"brier": res["primary_brier"]})
        tampered = json.loads(json.dumps(res))
        tampered["per_night"]["brier"][0] += 1.0
        short = json.loads(json.dumps(res))
        short["per_night"]["anchor_ts"] = short["per_night"]["anchor_ts"][:-1]
        check("lock-checker-passes-a-good-lock-and-can-fail",
              clean == []
              and FF.lock_problems(tampered, "forward_weekend_anchor", {"brier": res["primary_brier"]})
              and FF.lock_problems(short, "forward_weekend_anchor", {"brier": res["primary_brier"]})
              and FF.lock_problems(res, "forward_dow_anchor", {"brier": res["primary_brier"]})
              and FF.lock_problems({"n_nights": 3}, "forward_weekend_anchor", {"brier": {}}),
              f"clean {clean}")
        unscored = [r["member"] for r in rep if not r["scored"]]
        check("report-lists-every-member, scored or not",
              {r["member"] for r in rep} == set(locks) and "E2c-iv-correction" in unscored)
        # a lock written before per-night deltas existed is reported, not crashed on
        (tdp / "forward_hour_anchor_result.json").write_text(json.dumps({"scored": True}))
        rep2 = FF.report(research=tdp)
        old = next(r for r in rep2 if r["member"] == "forward_hour_anchor")
        check("a-lock-without-per-night-deltas-is-flagged", old["matches_lock"] is None
              and "per-night" in old["note"])

    bad = sum(not g for _, g, _ in CHECKS)
    print("forward-holdout family gate")
    for name, good, info in CHECKS:
        print(f"  [{'ok ' if good else 'FAIL'}] {name}{': ' + info if info and not good else ''}")
    print(f"\n{len(CHECKS) - bad}/{len(CHECKS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

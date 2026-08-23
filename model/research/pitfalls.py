"""
research/pitfalls.py
=====================================================================
Execution feedback: the mistakes this project actually made, as runnable checks.

WHY THIS IS CODE AND NOT A CHECKLIST IN A DOCUMENT

Every function below encodes a specific error that was committed here, shipped,
and later caught -- in several cases only because someone re-derived a number by
hand. A prose checklist did not prevent any of them; BENCHMARK.md already said
"every score is strictly proper, or it is a falsification test" while a broken
first-passage diagnostic sat in it for weeks.

So each pitfall is a function that takes the numbers and returns a verdict. An
experiment is validated by running these against its own outputs BEFORE its
result is written down. The list grows when a new class of error is found; it
is append-only, because a mistake that stops being checked is a mistake that
comes back.

The provenance line on each check is not decoration. It says which experiment
was wrong, which is what makes the check credible to the next reader.

    python -m model.research.pitfalls --self-test
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Verdict:
    ok: bool
    check: str
    detail: str
    provenance: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"[{mark}] {self.check}: {self.detail}"


@dataclass
class Report:
    verdicts: list = field(default_factory=list)

    def add(self, v: Verdict) -> "Report":
        self.verdicts.append(v)
        return self

    @property
    def failures(self) -> list:
        return [v for v in self.verdicts if not v.ok]

    def render(self) -> str:
        out = [str(v) for v in self.verdicts]
        if self.failures:
            out.append("")
            out.append(f"{len(self.failures)} CHECK(S) FAILED -- do not write "
                       f"this result down until each is answered:")
            for v in self.failures:
                out.append(f"  - {v.check}: {v.provenance}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# 1. A null centred below zero is not a floor
# ---------------------------------------------------------------------------
def check_skill_sign(skill: float, null_p95: float, name: str = "arm") -> Verdict:
    """Clearing a permutation null centred below zero is not evidence of skill.

    PROVENANCE: eval/shape.py reported "predictable and useful" for
    gradient-boosted arms whose out-of-sample R^2 was -0.016 and -0.021 -- worse
    than the constant they were scored against. They "cleared" only because the
    permutation null sat at -0.047/-0.040. Beating a null centred well below
    zero means "less bad than fitting shuffled targets", not "predictive".
    """
    clears = skill > null_p95
    positive = skill > 0.0
    ok = (not clears) or positive
    if clears and not positive:
        d = (f"skill {skill:+.5f} clears null p95 {null_p95:+.5f} but is BELOW "
             f"ZERO -- it loses to the constant. Not skill.")
    elif clears:
        d = f"skill {skill:+.5f} > null p95 {null_p95:+.5f} and > 0"
    else:
        d = f"skill {skill:+.5f} does not clear null p95 {null_p95:+.5f}"
    return Verdict(ok, f"skill-sign[{name}]", d,
                   "eval/shape.py read a negative R^2 as predictive")


# ---------------------------------------------------------------------------
# 2. An absolute-value relevance metric is passed by noise
# ---------------------------------------------------------------------------
def check_relevance_not_absolute(mean_abs_delta: float, mean_signed_delta: float,
                                 name: str = "impact") -> Verdict:
    """A relevance test on |change| measures perturbation size, not usefulness.

    PROVENANCE: ROADMAP Priority 1 required shape to "move the 2% touch
    probability by >= 1pp" and eval/shape.py implemented it as mean |delta|. It
    reported 1.05-2.53pp while the mean SIGNED delta was 0.01-0.40pp -- a random
    number generator passes that test. The rule meant IMPROVES.
    """
    ratio = abs(mean_signed_delta) / max(abs(mean_abs_delta), 1e-12)
    ok = ratio > 0.5
    return Verdict(
        ok, f"relevance-not-absolute[{name}]",
        f"mean|d| {mean_abs_delta:.4f} vs mean signed d {mean_signed_delta:+.4f} "
        f"(ratio {ratio:.3f}); a symmetric perturbation would score well on the "
        f"first and zero on the second",
        "eval/shape.py's pp-impact test was satisfiable by noise")


# ---------------------------------------------------------------------------
# 3. Score the floor, always
# ---------------------------------------------------------------------------
def check_beats_base_rate(score: float, base_rate_score: float,
                          lower_is_better: bool = True,
                          name: str = "forecaster") -> Verdict:
    """Any forecaster must be compared against ignoring every input.

    PROVENANCE: the Gaussian first-passage law in eval/firstpassage.py, fed
    PERFECT volatility and a fitted shape constant, is 1.62x/2.46x/6.55x WORSE
    than quoting the historical base rate at the 2/3/5% barriers. Nobody had
    scored the floor, so an entire error decomposition (the "92% is shape"
    figure) was computed through an instrument with negative skill.
    """
    better = score < base_rate_score if lower_is_better else score > base_rate_score
    ratio = score / max(base_rate_score, 1e-12)
    return Verdict(better, f"beats-base-rate[{name}]",
                   f"score {score:.5f} vs base-rate {base_rate_score:.5f} "
                   f"({ratio:.2f}x)" + ("" if better else "  -- LOSES TO A CONSTANT"),
                   "firstpassage.py's Gaussian law loses to the base rate by up to 6.55x")


# ---------------------------------------------------------------------------
# 4. Arms must differ only in the treatment
# ---------------------------------------------------------------------------
def check_arms_matched(arm_sizes: dict, tolerance: float = 0.10,
                       what: str = "calibration episodes") -> Verdict:
    """A nuisance parameter differing between arms is measured as the treatment.

    PROVENANCE: BENCHMARK §6l vetoed the rolling refresh on deep-tail
    miscalibration while giving the frozen arm 52,359 calibration episodes and
    the refreshed arm 17,511. Matching them (§6m) moved the frozen arm's tail
    MCB by +0.005013 on an untouched network, training set, test set and seeds
    -- the entire margin the veto rested on.
    """
    vals = list(arm_sizes.values())
    lo, hi = min(vals), max(vals)
    spread = (hi - lo) / max(hi, 1e-12)
    ok = spread <= tolerance
    return Verdict(ok, f"arms-matched[{what}]",
                   f"{arm_sizes} -- spread {100*spread:.1f}% "
                   f"(tolerance {100*tolerance:.0f}%)",
                   "6l's veto was entirely the calibration-slice size")


# ---------------------------------------------------------------------------
# 5. A win count on a small sample with large scatter is a coin flip
# ---------------------------------------------------------------------------
def check_not_a_coin_flip(per_unit_deltas, name: str = "statistic") -> Verdict:
    """Warn when a win-count rule is being applied to an effect it cannot resolve.

    PROVENANCE: three six-window designs produced 2/6, 5/6 and 3/6 on the SAME
    deep-tail effect, because the effect (~0.003) was small against per-window
    scatter (~0.013). The mean favoured the refresh in all three. The count was
    what the rule read, so the rule flipped on a nuisance choice.
    """
    d = np.asarray(per_unit_deltas, dtype=np.float64)
    n = len(d)
    eff, scat = abs(d.mean()), d.std(ddof=1) if n > 1 else np.inf
    resolvable = eff > scat / np.sqrt(max(n, 1))
    return Verdict(
        bool(resolvable), f"not-a-coin-flip[{name}]",
        f"n={n}, |mean| {eff:.5f} vs sd {scat:.5f} (se {scat/np.sqrt(max(n,1)):.5f}); "
        + ("resolvable" if resolvable else
           "NOT resolvable at this n -- decide on the mean with a CI, not a win count"),
        "6l/6m/6n gave 2/6, 5/6 and 3/6 on one effect")


# ---------------------------------------------------------------------------
# 6. A correction is not self-verifying because it is humbler
# ---------------------------------------------------------------------------
def check_correction_verified(claim: str, verified_against: str) -> Verdict:
    """A correction must be checked against the data, not against its own logic.

    PROVENANCE: the Brownian benchmark was corrected from 1.5958 to 1.5218 on
    the sound-sounding argument that discrete sampling understates a running
    maximum. The premise was false -- the data's extremes come from 1-minute bar
    HIGHS, not 5-minute closes -- and the "correction" hid a third of the gap.
    The second attempt was more careful than the first and still wrong, because
    it never checked which column the extremes came from.
    """
    ok = bool(verified_against.strip())
    return Verdict(ok, "correction-verified",
                   f"'{claim}' verified against: "
                   f"{verified_against or 'NOTHING -- reasoning only'}",
                   "the 1.5218 benchmark was a wrong correction of a wrong number")


# ---------------------------------------------------------------------------
# 7. A guard that cannot return True is not a guard
# ---------------------------------------------------------------------------
def check_guard_is_reachable(guard_fn, positive_case, negative_case,
                             name: str = "guard") -> Verdict:
    """A compatibility guard must be exercised in BOTH directions by a test.

    PROVENANCE: runtime.has_mx() tested for `b.q_mx.weight`, a key that cannot
    exist -- q_mx is a MonotoneQuantileHead with median/up/dn sub-keys and no
    bare .weight. It returned False for every artifact and would have silently
    dropped the head from serving on any artifact that carried it. Harmless
    only by luck.
    """
    try:
        pos, neg = bool(guard_fn(positive_case)), bool(guard_fn(negative_case))
    except Exception as e:                                   # noqa: BLE001
        return Verdict(False, f"guard-reachable[{name}]", f"raised {e!r}",
                       "has_mx() checked a key that could never exist")
    ok = pos and not neg
    return Verdict(ok, f"guard-reachable[{name}]",
                   f"positive case -> {pos}, negative case -> {neg}"
                   + ("" if ok else "  -- the guard is a constant"),
                   "has_mx() returned False for every artifact")


# ---------------------------------------------------------------------------
# 8. The thing measured must be the thing that ships
# ---------------------------------------------------------------------------
def check_measured_is_shipped(measured: dict, shipped: dict,
                              keys=("train_end", "calib_end", "hidden",
                                    "seeds", "feature_extra_lag_hours")) -> Verdict:
    """Diagnostics on a configuration that is not deployed describe nothing.

    PROVENANCE: the first learning-dynamics pass measured a 40-epoch model, but
    train.py restores `best_state` at epoch 6, so the numbers described weights
    that never ship. Separately, features.py's default lag had to be flipped or
    training and serving would have disagreed by an hour.
    """
    diffs = {k: (measured.get(k), shipped.get(k)) for k in keys
             if k in measured and k in shipped and measured[k] != shipped[k]}
    return Verdict(not diffs, "measured-is-shipped",
                   "configurations agree" if not diffs
                   else f"MISMATCH {diffs}",
                   "learning diagnostics first measured a checkpoint that never ships")


ALL_CHECKS = [check_skill_sign, check_relevance_not_absolute,
              check_beats_base_rate, check_arms_matched,
              check_not_a_coin_flip, check_correction_verified,
              check_guard_is_reachable, check_measured_is_shipped]


def self_test() -> int:
    """Every check must fire on the historical case that motivated it."""
    r = Report()
    # each pair is (should_fail_case, should_pass_case) from the real record
    r.add(check_skill_sign(-0.016, -0.047, "gbdt-range (historical FAIL)"))
    r.add(check_skill_sign(+0.0611, +0.0042, "a genuine positive"))
    r.add(check_relevance_not_absolute(1.0547, 0.0954, "shape-pp (historical FAIL)"))
    r.add(check_relevance_not_absolute(0.0100, 0.0095, "a real signed effect"))
    r.add(check_beats_base_rate(0.40217, 0.24762, name="gauss-2pct (historical FAIL)"))
    r.add(check_beats_base_rate(0.19607, 0.21666, name="noctua-2pct"))
    r.add(check_arms_matched({"frozen": 52359, "refreshed": 17511},
                             what="6l calibration (historical FAIL)"))
    r.add(check_arms_matched({"frozen": 17511, "refreshed": 17511},
                             what="6m calibration"))
    r.add(check_not_a_coin_flip([0.0078, 0.0019, -0.0047, 0.0007, 0.0029, -0.0108],
                                "6l tail MCB (historical FAIL)"))
    r.add(check_correction_verified("benchmark is 1.5218", ""))
    r.add(check_correction_verified("benchmark is 1.5831",
                                    "episodes.build_hourly uses 1-min bar HIGHS; "
                                    "simulated at 4 resolutions"))
    r.add(check_guard_is_reachable(lambda k: "b.q_mx.weight" in k,
                                   ["b.q_mx.median.weight"], ["b.q_up.median.weight"],
                                   "has_mx-original (historical FAIL)"))
    r.add(check_guard_is_reachable(lambda k: any("b.q_mx.median.weight" in x for x in k),
                                   ["b.q_mx.median.weight"], ["b.q_up.median.weight"],
                                   "has_mx-fixed"))
    r.add(check_measured_is_shipped({"train_end": "2026-02-09"},
                                    {"train_end": "2023-01-01"}))
    print(r.render())
    expect_fail = 8   # the historical cases, which MUST still be caught
    got = len(r.failures)
    print(f"\nself-test: {got} failures, expected {expect_fail} "
          f"(the historical errors these checks exist to catch)")
    return 0 if got == expect_fail else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Execution-feedback checks")
    ap.add_argument("--self-test", action="store_true",
                    help="assert every check still fires on its historical case")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    print(f"{len(ALL_CHECKS)} checks available:")
    for c in ALL_CHECKS:
        print(f"  {c.__name__:34} {c.__doc__.strip().splitlines()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

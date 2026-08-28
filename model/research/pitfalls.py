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

import math

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


# ---------------------------------------------------------------------------
# 9. A rule whose condition cannot be satisfied is not a rule
# ---------------------------------------------------------------------------
def check_rule_satisfiable(required: int, available: int,
                           unit: str = "folds") -> Verdict:
    """A pre-registered threshold must be reachable with the data that exists.

    PROVENANCE: BENCHMARK §9 pre-registered "wins the 2% barrier in at least 5
    of 6 years" for the sigma-atom test. The test split begins 2024-07-01 and
    the data ends 2026-08-09, so only THREE calendar years exist -- 5 of 6 is
    unsatisfiable without scoring the model on data it trained on. The rule was
    written by the same person who had just been burned twice by rule design,
    and the defect was found by the agent executing it, not by its author.

    Check this when the rule is WRITTEN, not when it is applied: at application
    time an unsatisfiable condition looks identical to a failed one, and gets
    reported as evidence against the hypothesis.
    """
    ok = available >= required
    return Verdict(ok, "rule-satisfiable",
                   f"rule needs {required} {unit}, data provides {available}"
                   + ("" if ok else "  -- UNSATISFIABLE BY CONSTRUCTION; the "
                                    "condition can only ever fail"),
                   "§9's '5 of 6 years' when only 3 years exist")


def check_eval_matches_trainer(eval_kwargs: dict, trainer_kwargs: dict,
                               keys=("sigma_ref", "shape_cols", "hidden",
                                     "seeds", "lam_r", "lam_mx")) -> Verdict:
    """The EVALUATION path must train the same configuration production does.

    PROVENANCE: `eval/benchmark.py:514` called `run_fold` without
    `sigma_ref_fn`, so Stage B trained against realized RV -- which
    `prepare()`'s own docstring documents as a train/serve skew that
    manufactures Spearman -0.4331 from arithmetic alone. `train_v2.py`
    meanwhile builds the causal reference and stamps
    `stage_b_sigma_ref: causal_har_1d_clipped` into the shipped artifact, and
    BENCHMARK.md 6b records that retargeting as ADOPTED. Every headline number
    therefore described a model that was not the shipped model.

    `check_measured_is_shipped` did not catch this, and could not: it compares
    ARTIFACT METADATA. A harness that trains a different configuration leaves
    the artifact untouched. This check compares the two call sites instead.
    """
    diffs = {k: (eval_kwargs.get(k), trainer_kwargs.get(k)) for k in keys
             if eval_kwargs.get(k) != trainer_kwargs.get(k)}
    return Verdict(not diffs, "eval-matches-trainer",
                   "evaluation and production trainer agree" if not diffs
                   else f"MISMATCH {diffs}",
                   "benchmark.py trained Stage B against realized RV while "
                   "train_v2.py used the causal reference")


# ---------------------------------------------------------------------------
# 11. A condition decided by a NaN was never decided by the data
# ---------------------------------------------------------------------------
def check_ci_is_defined(ci, name: str = "") -> Verdict:
    """A pre-registered condition must be evaluated against a REAL interval.

    PROVENANCE: `eval/anchor_freshness.py` fed six yearly fold deltas to
    `direction.block_bootstrap_ci`, which returns (nan, nan) below n = 20 by
    design -- a correct guard for its intended argument, a per-episode loss
    difference, where n < 20 means the caller erred. The unit here was a FOLD.
    The rule then asked

        primary = sp_hi < 0.0

    and `nan < 0.0` is False, so the script printed "PRIMARY ... : False" and
    "DO NOT ADOPT" without any statistic having looked at the data. The verdict
    was right by luck -- all six folds moved the wrong way -- which is the
    worst version of this failure, because a correct answer is not evidence
    that the machinery works.

    NaN is uniquely dangerous in a decision rule: it silently satisfies every
    "did not clear the bar" branch. A rule that returns the SAME answer whether
    the effect is huge, zero, or unmeasured is not testing anything. Assert the
    interval is finite BEFORE comparing it, and prefer a helper that raises --
    `direction.ci_excludes_zero` does -- over one that returns a bool.
    """
    lo, hi = (float(ci[0]), float(ci[1])) if ci is not None else (float("nan"),) * 2
    ok = math.isfinite(lo) and math.isfinite(hi)
    tag = f"[{name}]" if name else ""
    return Verdict(ok, f"ci-is-defined{tag}",
                   f"interval [{lo:+.5g}, {hi:+.5g}]" if ok else
                   f"UNDEFINED interval [{lo}, {hi}] -- every 'did not clear "
                   f"the bar' branch would answer False without consulting "
                   f"the data",
                   "anchor_freshness compared a (nan, nan) CI and printed a verdict")


# ---------------------------------------------------------------------------
# 12. A corruption that cannot move the value is not a test
# ---------------------------------------------------------------------------
def check_corruption_bites(before, after, name: str = "") -> Verdict:
    """A leakage test corrupts a series and checks the corruption is DETECTED.
    If the corruption leaves some values unchanged, those rows were never
    tested at all -- and the test reports a clean pass for them.

    PROVENANCE: `eval/ivfeatures.py` corrupts DVOL with a pure multiplier,
    `v -> v * uniform(2, 5)`. DVOL's minimum is 19.17, so multiplication always
    moves it and the style is sound there. `eval/fundingfeatures.py` inherited
    the same style for the funding rate, where **17.8 % of `interest_1h` values
    are exactly 0.0** -- and `0 * anything == 0`. The positive-control decoy
    went uncaught on a boundary episode whose underlying value happened to be
    zero, giving 9/10 instead of 10/10. `eval/leakage.py`'s own docstring names
    this family ("a sufficiently unlucky multiplier could leave a value
    numerically close to unchanged"); an exact zero is its limiting case.

    The fix was an affine corruption, `v * factor + offset`, which restored
    10/10. The general rule: **check the corruption actually moved every row it
    claims to have corrupted**, rather than assuming the transform is faithful
    on the data at hand. A style that is safe for one series is not thereby
    safe for another.
    """
    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    if b.shape != a.shape:
        return Verdict(False, f"corruption-bites{f'[{name}]' if name else ''}",
                       f"shape mismatch {b.shape} vs {a.shape}",
                       "a corruption that changed the array's shape is not the "
                       "corruption the test intended")
    both_finite = np.isfinite(b) & np.isfinite(a)
    unmoved = int((both_finite & (b == a)).sum())
    ok = unmoved == 0
    tag = f"[{name}]" if name else ""
    return Verdict(ok, f"corruption-bites{tag}",
                   f"all {b.size} corrupted values moved" if ok else
                   f"{unmoved} of {b.size} values UNCHANGED by the corruption -- "
                   f"those rows were never tested and will report a clean pass",
                   "a pure multiplier is a no-op on the 17.8% of funding rates "
                   "that are exactly zero")


ALL_CHECKS = [check_skill_sign, check_relevance_not_absolute,
              check_beats_base_rate, check_arms_matched,
              check_not_a_coin_flip, check_correction_verified,
              check_guard_is_reachable, check_measured_is_shipped,
              check_rule_satisfiable, check_eval_matches_trainer,
              check_ci_is_defined, check_corruption_bites]


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
    r.add(check_rule_satisfiable(5, 3, "years"))          # §9 (historical FAIL)
    r.add(check_rule_satisfiable(5, 6, "folds"))          # the usual rule
    r.add(check_eval_matches_trainer({"sigma_ref": None},
                                     {"sigma_ref": "causal_har_1d_clipped"}))  # historical FAIL
    r.add(check_eval_matches_trainer({"sigma_ref": "causal_har_1d_clipped"},
                                     {"sigma_ref": "causal_har_1d_clipped"}))
    r.add(check_ci_is_defined((float("nan"), float("nan")),
                              "E-anchor spike QLIKE (historical FAIL)"))
    r.add(check_ci_is_defined((0.02151, 0.05408), "the same delta, re-derived"))
    # the funding rate's exact zeros, before and after a PURE multiplier
    _v = np.array([0.0, 1e-5, 0.0, 3e-5])
    r.add(check_corruption_bites(_v, _v * 3.0, "pure multiplier (historical FAIL)"))
    r.add(check_corruption_bites(_v, _v * 3.0 + 1e-5, "affine, as fixed"))
    print(r.render())
    expect_fail = 12  # the historical cases, which MUST still be caught
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

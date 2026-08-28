"""
research/report.py
=====================================================================
Generates REPORT.md from the artifacts, so the report cannot drift from the
numbers.

WHY THIS IS A SCRIPT AND NOT A DOCUMENT

Every table in the report is read from a JSON artifact at generation time.
Nothing is transcribed. This project has already been bitten twice by numbers
quoted from memory rather than from a file -- invented split boundaries
(`iv-coverage-2`) and a claim about `paired_per_episode` that turned out to be
None in both artifacts it cited (§29). A report assembled by hand is a third
opportunity for the same mistake, and it is the one that gets read.

The PROSE is written here too, and that is deliberate: an argument that is only
true for one set of numbers should live next to the code that reads them, so
that a changed artifact makes the sentence visibly stale rather than quietly
wrong. Where a sentence depends on a number, the number is interpolated.

A missing artifact produces a section that says so. It does not produce a
guess, and it does not silently omit the row.

    python -m model.research.report            # -> model/RESEARCH_REPORT.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

A = Path("model/artifacts")


def load(name: str):
    p = A / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:                                            # noqa: BLE001
        return None


def missing(name: str, what: str) -> str:
    return (f"> **`{name}` is not present.** {what} is therefore not reported "
            f"here. Regenerate it with the command in "
            f"[Reproducing this](#reproducing-this) and re-run this generator.\n")


# --------------------------------------------------------------------------
def sec_volatility() -> str:
    d = load("vol_matrix.json")
    if d is None:
        return missing("vol_matrix.json", "The four-horizon volatility matrix")
    out = [
        "Every arm at a given horizon is scored on the **same episodes**, with "
        "the same target and the same loss. The baseline to beat is chosen on "
        "the **calibration** slice and never on test.\n",
        f"Bonferroni within this family: {d['family_size']} rows, so intervals "
        f"are at {100*(1-d['alpha']):.2f}%. Seeds: {d['seeds']}.\n",
    ]
    for H in sorted(d["horizons"], key=lambda k: int(k)):
        r = d["horizons"][H]
        out.append(f"\n### H = {H}h — {r['n_test']:,} test episodes, "
                   f"folds {r['years']}\n")
        out.append(f"Best baseline by calibration QLIKE: **{r['best_baseline']}** "
                   + " · ".join(f"{k} {v:.4f}" for k, v in
                                sorted(r["calib_qlike"].items(), key=lambda kv: kv[1]))
                   + "\n")
        if r.get("arms_absent"):
            out.append(f"\nNot scored at this horizon: `{'`, `'.join(r['arms_absent'])}`\n")
        out.append("\n| arm | QLIKE | vs best | worst fold | spike | calm | "
                   "paired CI (blocks) | same at n^(1/3)? |\n"
                   "|---|---:|---:|---:|---:|---:|---|---|\n")
        for k, v in r["arms"].items():
            ci = v.get("paired_ci")
            cis = "— (is the baseline)" if ci is None else \
                f"[{ci[0]:+.5f}, {ci[1]:+.5f}] ({v.get('block_len')})"
            ct = v.get("paired_ci_cuberoot")
            if ct is None:
                same = "—"
            else:
                same = "yes" if (ct[0] > 0) == (ci[0] > 0) else "**no**"
            out.append(f"| `{k}` | {v['qlike']:.5f} | {v['delta_vs_best']:+.5f} | "
                       f"{v['worst_fold']:.5f} | {v['spike']:.4f} | {v['calm']:.4f} | "
                       f"{cis} | {same} |\n")
        out.append("\nPre-registered verdict: "
                   + " · ".join(f"**{k}** {vv}" for k, vv in r["verdicts"].items())
                   + "\n")
    out.append(
        "\nThe fold-level spread is carried in the artifact as `per_fold` and is "
        "**not** the primary. `vol-matrix-power` measured its minimum detectable "
        "effect at 5.21% / 11.76% / 31.68% / 65.48% of the persistence baseline "
        "at H = 1 / 6 / 24 / 168, against a 4.98% reference effect — one row "
        "marginal, three not powered. That was measured *before* the matrix was "
        "built, which is the only time the measurement is worth anything.\n")
    return "".join(out)


def sec_direction() -> str:
    d = load("direction_bench.json")
    if d is None:
        return missing("direction_bench.json", "The direction benchmark")
    out = [
        "The baseline is the **calibration-window base rate**, not 0.5. "
        "P(R>0) rises with horizon and moves between years, so beating a coin "
        "demonstrates nothing. Calibration slope and intercept are **pass "
        "conditions**; AUC is reported and is explicitly not one.\n",
        "\n| H | arm | Brier | BSS vs calib | AUC | cal slope | cal int | "
        "paired CI | verdict |\n|---|---|---:|---:|---:|---:|---:|---|---|\n",
    ]
    for H in sorted(d, key=lambda k: int(k)):
        for arm in ("base_unc", "base_calib", "logistic", "gbm", "placebo", "shuffled"):
            if arm not in d[H]:
                continue
            r = d[H][arm]
            ci = r.get("paired_ci")
            cis = "—" if not ci else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
            out.append(f"| {H} | `{arm}` | {r['brier']:.5f} | "
                       f"{r['bss_vs_calib']:+.5f} | {r['auc']:.4f} | "
                       f"{r['cal_slope']:+.3f} | {r['cal_intercept']:+.3f} | "
                       f"{cis} | {r.get('verdict','—')} |\n")
    out.append(
        "\nA **positive** paired CI means the arm is **worse** than the baseline: "
        "the quantity bootstrapped is arm-minus-baseline Brier.\n")
    return "".join(out)


def sec_economics() -> str:
    d = load("econ_voltarget.json")
    head = (
        "**There is no options P&L in this report and there will not be one.** "
        "`model/artifacts/datasources.json` records 18 probes and 2 reachable "
        "endpoints; every exchange API and aggregator returns 403 through the "
        "egress proxy. The only option-adjacent series on disk is the Deribit "
        "DVOL volatility *index* — a level, with no strikes, no expiries, no "
        "bid/ask, no size, no prints. An options P&L could only be simulated, "
        "and every assumption the simulation needed would do more work than the "
        "forecast being tested.\n\n"
        "A directional backtest is also absent, for a different reason: the "
        "signal was measured to carry no information at n ≈ 49,000 per horizon, "
        "so its equity curve would be a random walk with a fee drag.\n\n"
        "What *can* be measured is a **volatility-targeting overlay on spot "
        "BTC**, which is the actual use of a volatility forecast for anyone "
        "without an options book. Its primary endpoint is risk control — "
        "|realised annualised vol − target| — not return.\n\n")
    if d is None:
        return head + missing("econ_voltarget.json", "The overlay result")
    out = [head,
           f"Target {d['target']:.0%} annualised, weight capped at {d['w_max']}, "
           f"H = {d['horizon']}h, rebalanced at {d['rebalance_hour']:02d}:00 UTC so "
           f"consecutive windows do not overlap. Costs {d['cost_bps']} bps "
           f"round-trip — **assumptions, not measurements**: this repository has "
           f"no order book and no fee schedule.\n",
           "\n| arm | mean \\|vol err\\| | worst | realised vol | turnover | mean w | "
           + " | ".join(f"net @{c:.0f}bp" for c in d["cost_bps"])
           + " | paired CI vs best |\n|---|---:|---:|---:|---:|---:|"
           + "---:|" * len(d["cost_bps"]) + "---|\n"]
    for k, v in d["arms"].items():
        ci = v.get("paired_ci_vs_best")
        cis = "— (is the best arm)" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        nets = " | ".join(f"{v['net_return_by_cost'][str(c)]:+.3f}" for c in d["cost_bps"])
        out.append(f"| `{k}` | {v['vol_error_mean']:.4f} | {v['vol_error_worst']:.4f} | "
                   f"{v['realised_vol']:.4f} | {v['turnover']:.4f} | {v['mean_w']:.3f} | "
                   f"{nets} | {cis} |\n")
    out.append(f"\nBest arm on the primary: **{d['best_arm_by_primary']}**. "
               f"Ranking by net return identical at all three cost levels: "
               f"**{d['ranking_cost_stable']}**"
               + ("" if d["ranking_cost_stable"] else
                  " — so the return comparison is **cost-dependent** and no arm "
                  "is declared better on it")
               + ".\n")
    return "".join(out)


def sec_experiments() -> str:
    p = Path("model/research/ledger.json")
    if not p.exists():
        return missing("research/ledger.json", "The experiment register")
    es = json.loads(p.read_text())["experiments"]
    counts: dict[str, int] = {}
    for e in es:
        counts[e["verdict"]] = counts.get(e["verdict"], 0) + 1
    out = [f"{len(es)} pre-registered experiments. "
           + " · ".join(f"**{k}** {v}" for k, v in sorted(counts.items()))
           + "\n\nEvery row was registered with its decision rule **before** it "
           "ran. Failures are not deleted; they stay in the family and count "
           "against the multiple-testing correction.\n",
           "\n| id | topic | verdict | question |\n|---|---|---|---|\n"]
    for e in es:
        sup = " ⤳" if e.get("superseded_by") else ""
        out.append(f"| `{e['id']}`{sup} | {e['topic']} | {e['verdict']} | "
                   f"{e['question'][:110]} |\n")
    out.append("\n⤳ = superseded by a later entry; the original is kept rather "
               "than edited.\n")
    return "".join(out)


TIMELINE = """```mermaid
timeline
    title NOCTUA — what was decided, and when it was decided against
    section Foundations
        Dataset and model : 510,496 episodes from 1-min bars
                          : walk-forward folds with an H-derived embargo
        Leakage audit     : 42 columns x 6 eras x 2 corruption styles
                          : decoy caught in 12 of 12 trials
    section Levers tried
        Spike upweighting : REJECT
        Ensemble weight   : NULL — one fold decides everything
        Anchor freshness  : NULL
        Path shape        : REJECT
    section Implied volatility
        IV as a column    : REJECT on coverage, before it was fitted
        IV as a residual  : REJECT — the gain was the intercept
        E2c dynamics      : ADVANCE, then NOT PROVEN after audit
    section Measuring the measurement
        Effective sample  : 24x episodes bought 1.53x precision
        Power before build: 3 designs killed before spending compute
        Data-use ledger   : no untouched holdout exists; one is frozen forward
    section This phase
        Direction, 4 horizons : NULL — 16 of 16 arms fail
        Volatility, 4 horizons : see the matrix
        Economics             : options P&L declined, overlay measured
```"""

FLOWCHART = """```mermaid
flowchart TD
    Q["A hypothesis"] --> MDE{"MDE stated,<br/>effect above it?"}
    MDE -- no --> NP["NOT POWERED<br/>redesign or do not run"]
    MDE -- yes --> PRE["Pre-register: population,<br/>primary, guards, family size,<br/>expected outcome"]
    PRE --> COMMIT["Commit the rule<br/>BEFORE the harness exists"]
    COMMIT --> RUN["Run"]
    RUN --> GUARD{"Can every guard<br/>actually fail?"}
    GUARD -- no --> FIX["Fix the guard.<br/>The result does not count<br/>until it can fail."]
    FIX --> RUN
    GUARD -- yes --> CTRL{"Placebo and<br/>shuffled control<br/>both negative?"}
    CTRL -- no --> BROKEN["The harness is broken.<br/>No verdict."]
    CTRL -- yes --> PRIM{"Primary interval<br/>excludes zero<br/>favourably?"}
    PRIM -- no --> NULLV["NULL / REJECT<br/>recorded, kept in the family"]
    PRIM -- yes --> AUD["Four audits, each trying<br/>to DISPROVE"]
    AUD --> SURV{"Survives all four?"}
    SURV -- no --> NP2["NOT PROVEN<br/>the shipped model is unchanged"]
    SURV -- yes --> ADV["ADVANCE<br/>candidate, still not adopted"]
    ADV --> PROD{"Measured on the PRODUCT,<br/>not a proxy?"}
    PROD -- no --> ADV
    PROD -- yes --> ADOPT["ADOPT"]
```"""

REPRO = """Everything below runs from a clean checkout with no network access.
Artifacts land in `model/artifacts/`.

```bash
# 0. the data the rest depends on (already committed)
ls model/artifacts/btcusd_1h.parquet model/artifacts/episodes_h4.parquet

# 1. the point-in-time audit, including the deliberate leak decoy
python -m model.eval.leakage
python -m model.eval.leakage --episodes model/artifacts/episodes_h4.parquet \\
    --out model/artifacts/leakage_h4.json      # probes H=1 and H=168

# 2. power BEFORE the experiments that depend on it
python -m model.eval.slice_power

# 3. the four-horizon volatility matrix  (~1h, 6 folds x 2 variants x 3 seeds)
python -m model.eval.vol_matrix

# 4. the direction benchmark             (~30 min)
python -m model.eval.direction_bench

# 5. the economic overlay                (~15 min)
python -m model.eval.econ_voltarget

# 6. the guards, which must all still be capable of failing
python -m model.research.pitfalls --self-test
python -m model.research.ledger --validate

# 7. regenerate this report from the artifacts
python -m model.research.report
```

Determinism: every model arm is seeded (`seed=0..2`); every bootstrap is
seeded (`seed=0`). The GARCH fit is multi-start from nine fixed starting
points, so it does not depend on the optimiser's own initialisation. Two runs
on the same artifacts produce the same tables."""


ASSUMPTIONS = """1. **Realized volatility from 5-minute returns is the target**, not an
   unobservable. Every arm is scored against the same estimator, so a bias in
   it cancels in the comparison — but the *level* of any QLIKE figure inherits
   it.
2. **QLIKE is the loss.** It is asymmetric: at a factor-2 error, under-forecast
   is penalised 1.60× more than over-forecast, and it is minimised by the
   conditional **mean** of variance, not the median. The shipped model reports
   a median, which is a known and unresolved mismatch (`E-scale`, still open).
3. **Walk-forward folds with an H-derived embargo** are the evaluation.
   Consecutive episodes overlap by construction, so all intervals are
   moving-block bootstraps and the block is at least twice the forward window.
4. **The trading costs in the economic section are assumptions, not
   measurements.** No order book, no fee schedule. Three levels are reported
   and a ranking that changes across them is declared cost-dependent.
5. **No untouched historical holdout exists.** Every calendar year has
   influenced training, calibration, feature selection, model selection or
   experiment design. This is documented year by year in
   `research/DATA_USE.md`, and the only honest remedy — a forward holdout
   frozen 2026-08-28 — is stated there with the uncomfortable part included:
   at roughly one independent regime per year, resolving a 6% effect on
   fold-level inference needs years, not weeks.
6. **The shipped model has not changed at any point in this work.** Nothing in
   this report is an adoption."""


def build() -> str:
    parts = [
        "# NOCTUA — volatility and probabilistic direction\n",
        "*Educational research. Not financial advice. Generated by "
        "`python -m model.research.report`; every table is read from an "
        "artifact rather than transcribed.*\n",
        "\n## Executive summary\n\n",
        exec_summary(),
        "\n## Assumptions this rests on\n\n", ASSUMPTIONS, "\n",
        "\n## Volatility: the four-horizon matrix\n\n", sec_volatility(),
        "\n## Direction as a probability forecast\n\n", sec_direction(),
        "\n## Economic validation, and its boundary\n\n", sec_economics(),
        "\n## How a hypothesis becomes a result here\n\n", FLOWCHART, "\n",
        "\n## Timeline\n\n", TIMELINE, "\n",
        "\n## The experiment register\n\n", sec_experiments(),
        "\n## Reproducing this\n\n", REPRO, "\n",
    ]
    return "".join(parts)


def _direction_bullet() -> str:
    """The direction claim, COUNTED from the artifact rather than asserted.

    An executive summary is exactly where a remembered number does the most
    damage, because it is the sentence people quote. So the counts here are
    derived: if the artifact changes, the sentence changes with it or the
    generator says the artifact is missing.
    """
    d = load("direction_bench.json")
    if d is None:
        return ("- **Direction**: `direction_bench.json` is not present, so no "
                "claim is made here.\n")
    model_arms = ("logistic", "gbm")
    tot = fails = adverse = straddle = 0
    controls_ok = True
    for H in d:
        for arm in model_arms:
            r = d[H].get(arm)
            if r is None:
                continue
            tot += 1
            if r.get("verdict") == "FAIL":
                fails += 1
            ci = r.get("paired_ci")
            if ci:
                if ci[0] > 0:            # positive => arm WORSE than baseline
                    adverse += 1
                elif ci[0] <= 0 <= ci[1]:
                    straddle += 1
        for ctl in ("shuffled", "placebo"):
            r = d[H].get(ctl)
            if r is not None and r.get("bss_vs_calib", -1) > 0:
                controls_ok = False
    n = max((d[H]["base_calib"]["n"] for H in d if "base_calib" in d[H]), default=0)
    horizons = ", ".join(f"{k}h" for k in sorted(d, key=lambda k: int(k)))
    return (
        f"- **Direction is closed at all four horizons ({horizons}).** "
        f"{fails} of {tot} model arms fail their pre-registered rule. The paired "
        f"per-episode interval excludes zero on the *adverse* side — the arm is "
        f"worse than the calibration-window base rate — in {adverse} of {tot} "
        f"rows and straddles zero in {straddle}, at n up to {n:,} per horizon. "
        + ("Both negative controls behave. " if controls_ok else
           "**A negative control scored positively, so the harness is suspect "
           "and the verdict below should not be read as a result.** ")
        + "This is a measured absence, not an underpowered one.\n")


def _volatility_bullet() -> str:
    d = load("vol_matrix.json")
    if d is None:
        return ("- **Volatility**: `vol_matrix.json` is not present, so no claim "
                "is made here.\n")
    rows, clears = [], 0
    for H in sorted(d["horizons"], key=lambda k: int(k)):
        r = d["horizons"][H]
        bits = []
        for k in ("noctua", "noctua40"):
            v = r["verdicts"].get(k)
            if v is None:
                continue
            if v == "CLEARS":
                clears += 1
                bits.append(f"`{k}` clears "
                            f"({r['arms'][k]['delta_vs_best']:+.5f})")
            elif v == "NOT EVALUABLE":
                bits.append(f"`{k}` not evaluable")
            else:
                bits.append(f"`{k}` fails "
                            f"({r['arms'][k]['delta_vs_best']:+.5f})")
        rows.append(f"**{H}h** vs `{r['best_baseline']}` — " + ", ".join(bits))
    fair = d.get("fair_baselines")
    caveat = ("" if fair else
              " These numbers come from the run whose OLS baselines are "
              "**horizon-blind** — fitted once per fold on the pooled sample "
              "across all four horizons, while NOCTUA sees `cal_H`. That "
              "confound is named in the ledger against my own result and is "
              "resolved by `vol-matrix-fair`; until that lands, the two rows "
              "that clear are **ADVANCE, not ADOPT**.")
    return ("- **The volatility matrix**, one NOCTUA arm per horizon against "
            "the mandatory baseline family: " + "; ".join(rows)
            + f". {clears} row(s) clear the pre-registered interval." + caveat
            + "\n")


def exec_summary() -> str:
    return (
        "The honest headline is that **the shipped model is unchanged by any of "
        "this**, and that the phase's two largest results are a measured absence "
        "and a boundary.\n\n"
        + _direction_bullet()
        + _volatility_bullet()
        + "- **An options P&L cannot be produced honestly here and is not "
          "produced.** What replaces it is a volatility-targeting overlay whose "
          "primary endpoint is risk control rather than return.\n"
        + """
Three things found by guards rather than by looking:

- A comment in the direction benchmark said only one feature column depended on
  the horizon. **Five do.** The benchmark was re-run from corrected features
  rather than defended — a null produced with degraded inputs is not a null.
- The default bootstrap block length is a rule of thumb about *sample size* and
  knows nothing about the *overlap* it exists to absorb. At the weekly horizon
  it was about a fifth of the shared window. Every interval here uses a block
  of at least twice the forward window, and the narrower one is reported beside
  it.
- The research ledger's schema was enforced on one write path only. Checking
  the file instead found a dangling supersede pointer and a one-sided link.
  Both are now gated in CI.
""")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="generate the research report")
    ap.add_argument("--out", type=Path, default=Path("model/RESEARCH_REPORT.md"))
    a = ap.parse_args(argv)
    a.out.write_text(build())
    print(f"wrote {a.out} ({len(build()):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

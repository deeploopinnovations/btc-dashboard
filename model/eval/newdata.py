"""
eval/newdata.py
=====================================================================
What data, obtainable through a GitHub Actions runner, would most plausibly
raise ONSET prediction above the measured ceiling in BENCHMARK.md §12?

THE CEILING THIS FILE IS AIMED AT

§12 measured a spike-tomorrow classifier on the current causal feature set,
walk-forward, production population, causal 180-day trailing 95th-percentile
spike flag (303 spike days: 177 onset, 126 continuation):

    all spike days                    AUC 0.8055   (n positives 76)
    continuation (already underway)   AUC 0.9293   (n positives 28)
    onset (first day of a cluster)    AUC 0.7332   (n positives 48)
                                       95% CI [0.6547, 0.8052]

Persistence does almost all of the work of the 0.78-0.81 headline. Genuine
anticipation of a cluster's FIRST day sits at 0.733, and that is where §7a's
measured cost lives: 45% under-forecasting on spike nights carrying 25.8% of
total loss. Every feature in the current set is a trailing statistic on BTC's
own past price -- onset, by definition, is the day that trailing statistic has
not yet moved. So the candidates below are restricted to information classes
that are NOT derived from BTC's own price history.

WHAT THIS FILE IS AND IS NOT

This file does not fetch anything -- `eval/harvest_newdata.py` and
`.github/workflows/harvest-newdata.yml` do that, for the single top-ranked
candidate. This file (a) documents the full candidate set and how each was
assessed, (b) is explicit about what is VERIFIED (probed and got a response),
ASSERTED (general knowledge, not tested this session), or REPO-DOCUMENTED
(this repo's own prior research, e.g. `OPTION_BUYER_ALPHA.md`, ran a real
fetch outside this container and recorded a number -- treated as stronger
than an assertion but weaker than something verified in this session, since
it was neither performed nor re-checked here), and (c) fixes, BEFORE any of
this data exists in the repo, the rule that decides whether it earns a place
in the model. That ordering matters: §12 itself is a story about a plausible
mechanism (a lag ceiling) that turned out to be an untested assumption, and
§10 is on record for a clean single-fold result reversing sign on more data.
A decision rule written after seeing results is not a decision rule.

    python -m model.eval.newdata      # reports on what has landed in
                                       # data/newdata/, if anything, and
                                       # otherwise explains what is missing
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ======================================================================
# THE CANDIDATES
# ======================================================================
#
# "verified" below is never True in this session -- everything here was
# assessed from a container that reaches only raw.githubusercontent.com and
# gist.github.com (`eval/datasources.py`). It is one of:
#
#   REPO-DOCUMENTED  this repo's own markdown research (OPTION_BUYER_ALPHA.md,
#                     SELLER_DIRECTIONAL_ALPHA.md, DAILY_EXPIRY_ALPHA.md,
#                     AUDIT.md, README.md, src/data.js) already records a
#                     real fetch or a real production code path against this
#                     exact endpoint -- stronger evidence than general
#                     knowledge, but from a *different* fetch process
#                     (client-side browser calls, or a one-off research pull
#                     dated 2026-06-12), not re-checked in this session.
#   ASSERTED         stated from training knowledge (e.g. "Deribit
#                     BTC-PERPETUAL launched August 2016"), not backed by
#                     anything in this repo or this session.
#   UNVERIFIED-FROM-CONTAINER  the honest default: could not be tested from
#                     here, full stop.

@dataclass(frozen=True)
class Candidate:
    name: str
    measures: str
    endpoint: str
    granularity: str
    history_depth: str
    depth_basis: str          # REPO-DOCUMENTED / ASSERTED / UNVERIFIED-FROM-CONTAINER
    onset_mechanism: str
    reachability: str
    rank: int
    verdict: str


CANDIDATES: list[Candidate] = [

    Candidate(
        name="Perpetual funding rate (Binance BTCUSDT / Deribit BTC-PERPETUAL)",
        measures="Positioning stress: what leveraged longs pay leveraged "
                 "shorts (or vice versa) every settlement.",
        endpoint="Binance: GET fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT "
                 "(paginated, start/end ms). "
                 "Deribit: GET www.deribit.com/api/v2/public/get_funding_rate_history"
                 "?instrument_name=BTC-PERPETUAL (paginated, start/end ms).",
        granularity="8-hour settlements (Binance); Deribit computes continuously "
                     "and the history endpoint's field names (interest_1h, "
                     "interest_8h) suggest a finer native cadence, but the "
                     "cadence actually returned is UNVERIFIED-FROM-CONTAINER.",
        history_depth="Binance: 2019-09 -> present (~6.9y). REPO-DOCUMENTED: "
                       "SELLER_DIRECTIONAL_ALPHA.md reports a real prior pull "
                       "of '7,402 8-hour records' spanning 2019-09->2026-06, "
                       "consistent with that start date. Deribit: launched the "
                       "BTC-PERPETUAL contract in Aug 2016 (ASSERTED), which "
                       "would predate NOCTUA's TRAIN_END (2023-01-01) entirely, "
                       "but how much of that history the get_funding_rate_history "
                       "endpoint actually retains is UNVERIFIED-FROM-CONTAINER.",
        depth_basis="REPO-DOCUMENTED (Binance number) / ASSERTED (Deribit launch date)",
        onset_mechanism="Funding extremes mark crowded, over-levered positioning; "
                         "the unwind of that crowding (a liquidation cascade) is a "
                         "well-known trigger for a volatility cluster's first day, "
                         "not just its continuation. This repo's own "
                         "SELLER_DIRECTIONAL_ALPHA.md documents the mechanism from "
                         "the other side: 'washed-out positioning' (funding<0) "
                         "precedes calm, and by implication crowded positioning "
                         "(funding high and rising) precedes the flush. AUDIT.md "
                         "§5.4 calls funding 'a genuine institutional signal' but "
                         "flags it there as a carry/risk-premium signal for "
                         "*direction* -- that caveat is about a different question "
                         "(which way price moves) than the one asked here (whether "
                         "a cluster starts).",
        reachability="Binance's SPOT api.binance.com is documented in this repo "
                      "(serve/fetch.py) as returning HTTP 451 from a GH Actions "
                      "runner (US-region geo-block). fapi.binance.com (futures) "
                      "was never itself tested from a runner in this repo, but "
                      "Binance derivatives carry a stricter US posture than spot, "
                      "so extending the 451 finding is a reasonable inference, "
                      "not a re-verified fact. Deribit has no documented block "
                      "anywhere in this repo; src/data.js and README.md describe "
                      "a live browser client calling Deribit's public API "
                      "directly ('Deribit | Unlimited public'), which is weaker "
                      "evidence for a server-side GH-runner call than a direct "
                      "test would be, but is the best evidence available.",
        rank=1,
        verdict="TOP RANK. Best-corroborated history depth of any candidate "
                "(Binance number is a real number from this repo's own prior "
                "work, not a guess), a concrete and previously-documented onset "
                "mechanism, and -- via Deribit -- a plausible route around the "
                "one reachability risk that is actually documented in this repo. "
                "harvest_newdata.py tries Deribit first and falls back to "
                "Binance, logging which venue actually answered.",
    ),

    Candidate(
        name="Deribit DVOL (30-day BTC implied-volatility index)",
        measures="The options market's own forward-looking volatility forecast "
                 "-- the single input class the model has never had, since "
                 "everything else in the feature set is backward-looking.",
        endpoint="GET www.deribit.com/api/v2/public/get_historical_volatility"
                  "?currency=BTC (no time-range parameters; returns whatever "
                  "window Deribit's backend currently retains).",
        granularity="Sub-daily native ticks (UNVERIFIED-FROM-CONTAINER exact "
                     "cadence); this repo's own prior research consumed it "
                     "resampled to one point per day.",
        history_depth="~2.7 years, 2023-09 -> present. REPO-DOCUMENTED: both "
                       "OPTION_BUYER_ALPHA.md ('DVOL history is only ~2.7 "
                       "years') and SELLER_DIRECTIONAL_ALPHA.md ('Deribit DVOL "
                       "2023-09->2026-06 (1,000 daily closes)') report the same "
                       "figure from a real prior pull. This is the single most "
                       "consequential number in this table: TRAIN_END is "
                       "2023-01-01 (noctua/splits.py), so DVOL's own history "
                       "starts a full 8 MONTHS AFTER training ends. It has ZERO "
                       "overlap with the training window, full coverage of most "
                       "of calib (2023-09->2024-07) and all of test.",
        depth_basis="REPO-DOCUMENTED",
        onset_mechanism="Textbook: implied vol is supposed to price in "
                          "anticipated risk before it realizes, which is exactly "
                          "the onset gap. AUDIT.md §5.4's skepticism about "
                          "'options skew' is about *directional* content, a "
                          "different hypothesis than 'does the IV *level* rise "
                          "ahead of a vol cluster' -- that caveat does not carry "
                          "over to this use.",
        reachability="Same venue and same reasoning as the funding-rate row "
                       "above: no documented block, best evidence is the live "
                       "browser client already using this exact API.",
        rank=2,
        verdict="STRONG SIGNAL, SHORT HISTORY -- worth doing despite the gap, "
                "but the gap has a real, specific cost: it cannot inform the "
                "base sigma model's training AT ALL (zero overlap with "
                "2017-08->2023-01), and any onset classifier that uses it must "
                "be fit and evaluated on the ~2.7-year DVOL-covered window "
                "only. §12's own onset test already has a thin positive count "
                "(48 in nearly 9 years); restricting to 2.7 years shrinks that "
                "further, which is exactly the kind of small-sample trap this "
                "session's own §10 already caught once for a different result. "
                "See DEPTH_COST below for the concrete accounting. Harvested by "
                "the same script as funding rate (one reachable venue, two "
                "series) so there is no separate reachability cost to trying.",
    ),

    Candidate(
        name="Scheduled macro-event calendar (FOMC, CPI, NFP)",
        measures="Whether a known, pre-scheduled macro print falls inside the "
                  "prediction horizon.",
        endpoint="No live API needed for the historical record -- FOMC and CPI "
                  "release dates are public record back to well before 2017 and "
                  "can be assembled as a static calendar. A recurring workflow "
                  "would only need to append newly-announced future dates "
                  "(e.g. from the Federal Reserve's own published schedule "
                  "page, or a maintained public calendar feed), which is a "
                  "much smaller and lower-risk task than the harvests above.",
        granularity="Daily (a date either carries a scheduled print or does not).",
        history_depth="Full, by construction -- these dates are historical "
                       "public record, not a live feed with a retention "
                       "window, so there is no depth constraint at all for the "
                       "training era. ASSERTED (the dates themselves are "
                       "well-known public facts; no specific source was probed "
                       "this session).",
        depth_basis="ASSERTED",
        onset_mechanism="This repo's own OPTION_BUYER_ALPHA.md already treats "
                          "a scheduled macro print as a real conditioning "
                          "variable ('AMBER (event tilt): DVOL 40-50 and a "
                          "scheduled macro print inside your horizon (CPI/NFP "
                          "12:30 UTC, FOMC 18:00 UTC)'), which is repo-internal "
                          "corroboration that this class of information is "
                          "already believed to matter for BTC vol, just not "
                          "yet wired into NOCTUA's own feature set. The "
                          "honest caveat: most of BTC's largest onset days in "
                          "this model's own history are crypto-idiosyncratic "
                          "(exchange failures, regulatory shocks, leverage "
                          "cascades) rather than macro-calendar events, so the "
                          "expected AUC contribution of this one feature is "
                          "plausibly smaller than its zero acquisition risk "
                          "makes it look.",
        reachability="Not a live-fetch reachability question for the "
                       "historical portion; a small ongoing workflow to append "
                       "newly-announced dates would need whatever calendar "
                       "source is chosen to be reachable from a runner, which "
                       "was not tested.",
        rank=3,
        verdict="CHEAPEST CANDIDATE BY FAR, MODEST EXPECTED PAYOFF. Zero "
                "depth risk is a genuine advantage given how badly depth "
                "bites the two options-market candidates above, and it "
                "deserves a place in a full evaluation, but the mechanism is "
                "the weakest of the top three for a BTC-specific onset "
                "problem. Not the one this session's workflow harvests (a "
                "static calendar and a scheduled-fetch workflow are different "
                "shapes of work from the parquet-bundle harvesters above), but "
                "flagged as easy enough to be worth a follow-up session.",
    ),

    Candidate(
        name="Cross-asset risk regime (VIX, DXY, gold)",
        measures="Whether a broad risk-off move in traditional markets is "
                  "already under way, on the hypothesis that it spills into "
                  "crypto with a lag.",
        endpoint="No confirmed no-key public source was found for this repo. "
                  "Plausible candidates (stooq.com CSV endpoints, which do not "
                  "require a key; FRED's API, which is free but requires a "
                  "registered key -- a credential this session does not have "
                  "and per the task's rules is not something to pursue) were "
                  "not tested and are not documented anywhere in this repo.",
        granularity="Daily, typically, for the plausible sources.",
        history_depth="Decades, if a working free endpoint exists -- these "
                        "indices have long public histories. ASSERTED, and "
                        "the weakest-supported ASSERTED claim in this table: "
                        "unlike the funding-rate and DVOL rows, there is no "
                        "repo-internal precedent of anyone actually having "
                        "pulled VIX, DXY, or gold data into this project.",
        depth_basis="ASSERTED, no repo precedent",
        onset_mechanism="Plausible but unmeasured anywhere in this repo. "
                          "Crypto/macro spillover is a real, published "
                          "phenomenon in the general literature, but nothing "
                          "here establishes it operates at the daily onset "
                          "horizon this model needs, versus at longer regime "
                          "horizons.",
        reachability="UNVERIFIED-FROM-CONTAINER and untested even at the "
                       "level of 'does a free endpoint exist that a GH runner "
                       "can reach' -- this is a gap in this investigation, "
                       "stated plainly rather than papered over with an "
                       "assumed stooq/FRED URL that has never been tried "
                       "against this repo's actual network path.",
        rank=4,
        verdict="PLAUSIBLE, LEAST-VERIFIED. Long history if reachable at all, "
                "but reachability itself is an open question here in a way it "
                "is not for the two Deribit-hosted candidates, and the onset "
                "mechanism has no repo-internal corroboration the way funding "
                "and DVOL both have. Would need its own reachability probe "
                "(a runner-side workflow that just curls a candidate stooq URL "
                "and reports the status code, cheaply, before committing to a "
                "harvester) before it could be ranked with any confidence.",
    ),

    Candidate(
        name="On-chain exchange flows / stablecoin supply",
        measures="Whether coins are moving toward exchanges (sell pressure, "
                  "and often a precursor to deleveraging) or stablecoin supply "
                  "is expanding (fresh buying power).",
        endpoint="The metrics with the clearest onset story -- LABELED "
                  "exchange in/outflows -- come from Glassnode, CryptoQuant, "
                  "or Nansen, all of which require a paid API tier for "
                  "anything beyond a handful of free basic series. Raw "
                  "on-chain data (blockchain.com API, mempool.space) is free "
                  "and has full history back to 2009, but does not carry "
                  "exchange labels, so it cannot answer the specific question "
                  "without a labeling effort this project has no data for. "
                  "Stablecoin total supply is technically obtainable "
                  "keylessly via a chain explorer's ERC-20 total-supply call, "
                  "but the well-known convenience endpoints (Etherscan) "
                  "require a free registered API key -- a credential this "
                  "session does not have, per the task's explicit rule to "
                  "report and move on rather than pursue.",
        granularity="Would be daily at best for most free on-chain "
                     "aggregates.",
        history_depth="Full blockchain history if using raw (unlabeled) data; "
                        "materially shorter and paywalled for the labeled "
                        "exchange-flow series that would actually answer the "
                        "onset question. ASSERTED.",
        depth_basis="ASSERTED",
        onset_mechanism="Real in the general literature (exchange inflow "
                          "spikes precede sell-driven volatility), but this "
                          "project cannot obtain the labeled version freely, "
                          "and the free unlabeled version does not carry the "
                          "signal the mechanism depends on.",
        reachability="Not pursued further: the informative version of this "
                       "candidate needs credentials this session does not "
                       "have, which is exactly the case the task instructions "
                       "say to report and move past rather than work around.",
        rank=5,
        verdict="DEPRIORITIZED ON CREDENTIALS, NOT MECHANISM. The mechanism is "
                "plausible; the free, keyless version of this data does not "
                "carry it. Revisit only if the user is willing to add a "
                "Glassnode/CryptoQuant/Etherscan key as a GitHub Actions "
                "secret -- which is within the security constraints here "
                "(secrets read from Actions, never inline) but was not "
                "assumed unilaterally.",
    ),

    Candidate(
        name="Order-book depth / imbalance snapshots",
        measures="Live buy/sell pressure in the limit-order book -- the class "
                  "AUDIT.md §5.4 itself calls 'the only source with measured, "
                  "replicated directional content in crypto.'",
        endpoint="Any major exchange's depth/order-book endpoint (Binance, "
                  "Coinbase, Bitstamp all have one), but for a SNAPSHOT, not a "
                  "history.",
        granularity="Sub-second to 1-minute, if collected live.",
        history_depth="Essentially ZERO for anything before collection "
                        "starts. Free exchange APIs do not archive historical "
                        "order-book depth; it exists only as a live feed. This "
                        "is the sharpest history-depth problem of any "
                        "candidate in this table -- not 'short', but "
                        "nonexistent until this repo starts collecting it.",
        depth_basis="ASSERTED (general knowledge that free order-book history "
                     "is not retained anywhere) -- and the one claim in this "
                     "table that is closest to certain, since 'no free vendor "
                     "sells historical L2 book data' is a standing, widely "
                     "known market fact rather than a guess about a specific "
                     "endpoint's retention window.",
        onset_mechanism="Real and well-established for short-horizon "
                          "DIRECTION per AUDIT.md §5.4. Less clearly "
                          "established for volatility ONSET specifically -- "
                          "the two are related but not the same question, and "
                          "AUDIT.md's endorsement was scoped to direction.",
        reachability="Would be reachable going forward (this repo already "
                       "fetches order-book-adjacent data client-side per "
                       "src/data.js), which is exactly the problem: 'going "
                       "forward' does not help retrain on the past. At §12's "
                       "measured onset rate (~48 onset days in the production "
                       "population across nearly 9 years, roughly 5-6/year), "
                       "accumulating a usable sample of onset days WITH "
                       "order-book context would take years of forward "
                       "collection before it could inform a retrain.",
        rank=6,
        verdict="LOWEST NEAR-TERM PRIORITY DESPITE THE STRONGEST GENERAL "
                "REPUTATION. Zero history is a harder constraint than DVOL's "
                "short history -- DVOL at least has 2.7 years to evaluate on "
                "today; order-book depth has none. Worth starting to collect "
                "now if the collection is cheap (it would ride the existing "
                "fetch-data.yml cron cheaply), but not worth building a "
                "dedicated backfill harvester for, because there is nothing "
                "to backfill.",
    ),
]


# ======================================================================
# THE PRE-REGISTERED DECISION RULE
# ======================================================================
#
# Written now, before funding-rate or DVOL data exists anywhere in this repo,
# for the same reason §12's lever rule was written before either lever was
# run "properly": this session has already watched a clean single-slice
# result reverse sign on more data (§10), and the check that would have
# caught §7a's wrong mechanism claim -- "is the information actually absent,
# or did we merely fail to use it?" -- is precisely a check that has to be
# specified before the answer is known, or it is not a check.
#
# The rule below deliberately mirrors §12's own adopt-a-lever bar (walk-
# forward, multi-fold, a guard against helping one slice at another's
# expense) rather than inventing a new standard, because BENCHMARK.md's own
# §12 already established that standard is the one this project trusts.

MIN_FOLDS = 6                  # standard 6-fold walk-forward, matching the
                                # fold count already used for lever adoption
                                # in §12
MIN_SEEDS = 3                  # matching §12's "3 seeds"
MIN_PASSING_FOLDS = 5          # of 6 -- matching §12's "in >= 5 of 6 folds"
MIN_ONSET_POSITIVES_IN_COVERAGE = 20
# §12's own onset AUC (0.733, CI width ~0.15) was measured on 48 onset
# positives. A feature class whose usable history covers materially fewer
# onset events than that produces a materially less informative CI -- 20 is
# not a theoretically derived number, it is a floor below which this project
# has no precedent for trusting an AUC CI at all (§12's own count is the only
# precedent that exists), stated explicitly so it can be argued with rather
# than silently assumed later.
MAX_CONTINUATION_AUC_DROP = 0.02
# A new onset feature must not be purchased at continuation's expense --
# continuation is the one part of this problem already solved (0.929), and
# an option seller who stops trusting an already-reliable signal has not
# been helped. Mirrors §12's separate guard on the two upweighting levers
# ("worsens calm-episode QLIKE by no more than 3%") applied to the AUC
# analogue of the same principle.


@dataclass
class FoldResult:
    """One fold x seed's worth of onset-classifier comparison: baseline
    (current causal features only) vs candidate (baseline + new feature)."""
    fold: int
    seed: int
    onset_auc_baseline: float
    onset_auc_candidate: float
    onset_auc_null_p95: float          # shuffled-label null, THIS fold
    continuation_auc_baseline: float
    continuation_auc_candidate: float
    n_onset_positives_in_coverage: int


def decision_rule(results: list[FoldResult]) -> dict:
    """Apply the pre-registered bar above to a completed set of fold results.

    Returns a dict with a boolean `adopt` and the itemized reasons, so a
    rejection is as legible as an acceptance -- the same discipline §12 used
    to report the trade-off on the upweighting levers rather than just a
    verdict.

    This function is deliberately literal and un-clever: it does not try to
    be a general statistics library, it encodes exactly the four conditions
    argued for above, on exactly the fold/seed grid this project already
    uses, so that changing the bar later requires editing a number here in
    plain sight rather than re-deriving it from a paragraph.
    """
    if not results:
        return {"adopt": False, "reason": "no results supplied"}

    folds = sorted({r.fold for r in results})
    seeds = sorted({r.seed for r in results})
    coverage_n = max(r.n_onset_positives_in_coverage for r in results)

    checks: dict[str, bool] = {}
    notes: list[str] = []

    checks["enough_folds"] = len(folds) >= MIN_FOLDS
    checks["enough_seeds"] = len(seeds) >= MIN_SEEDS
    checks["enough_onset_coverage"] = coverage_n >= MIN_ONSET_POSITIVES_IN_COVERAGE
    if not checks["enough_onset_coverage"]:
        notes.append(f"only {coverage_n} onset positives fall inside this "
                      f"feature's history coverage, below the "
                      f"{MIN_ONSET_POSITIVES_IN_COVERAGE}-positive floor; the "
                      f"CI on any AUC delta here should be treated as too "
                      f"wide to act on regardless of the point estimate")

    # per (fold, seed): does the candidate beat baseline AND clear the null?
    passing = 0
    total = 0
    worst_continuation_drop = 0.0
    for r in results:
        total += 1
        beats_baseline = r.onset_auc_candidate > r.onset_auc_baseline
        clears_null = r.onset_auc_candidate > r.onset_auc_null_p95
        if beats_baseline and clears_null:
            passing += 1
        drop = r.continuation_auc_baseline - r.continuation_auc_candidate
        worst_continuation_drop = max(worst_continuation_drop, drop)

    checks["passes_enough_fold_seeds"] = passing >= MIN_PASSING_FOLDS * len(seeds)
    checks["continuation_not_damaged"] = worst_continuation_drop <= MAX_CONTINUATION_AUC_DROP
    if not checks["continuation_not_damaged"]:
        notes.append(f"worst continuation-AUC drop {worst_continuation_drop:.4f} "
                      f"exceeds the {MAX_CONTINUATION_AUC_DROP} tolerance")

    adopt = all(checks.values())
    return {
        "adopt": adopt,
        "checks": checks,
        "passing_fold_seeds": f"{passing}/{total}",
        "onset_positive_coverage": coverage_n,
        "worst_continuation_auc_drop": worst_continuation_drop,
        "notes": notes,
    }


# ======================================================================
# WHAT TESTING THIS LOOKS LIKE, ONCE THE DATA EXISTS
# ======================================================================
#
# 1. `eval.harvest_newdata` lands `data/newdata/funding_btc.parquet` and
#    `data/newdata/dvol_btc.parquet`.
# 2. Join onto `noctua.train.load_all`'s episode/feature frame by anchor
#    timestamp (`ep["anchor_ts"]`), same convention `firstpassage.py` and
#    `direction.py` already use, forward-filled to the anchor since funding
#    and DVOL update on their own cadence, not the model's.
# 3. Re-run the §12-style onset/continuation split (`noctua.splits`
#    production mask, causal 180-day trailing 95th-percentile spike flag) on
#    the SAME 6-fold x 3-seed walk-forward grid `eval.direction` already
#    implements (`block_bootstrap_ci`, `shuffled_dsc_null`), once with the
#    current causal feature set alone (baseline) and once with the new
#    feature(s) appended (candidate), producing exactly the `FoldResult`
#    rows `decision_rule` above consumes.
# 4. For DVOL specifically: restrict BOTH arms of that comparison to the
#    DVOL-covered window (2023-09 onward per the REPO-DOCUMENTED figure
#    above, or whatever this repo's own harvested bundle actually shows once
#    it exists) so the comparison is apples-to-apples, not "candidate on a
#    short window vs baseline on the full window" -- the latter would let a
#    feature take credit for the window being easier, not for the feature
#    being informative.
# 5. Apply `decision_rule`. Report the full itemized dict either way, adopt
#    or not, the same way §12 reported the levers it declined to adopt.


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/newdata"))
    a = ap.parse_args(argv)

    print("CANDIDATES, ranked for the ONSET problem (BENCHMARK.md §12):\n")
    for c in sorted(CANDIDATES, key=lambda c: c.rank):
        print(f"  {c.rank}. {c.name}")
        print(f"     depth: {c.history_depth}")
        print(f"     basis: {c.depth_basis}")
        print()

    print("PRE-REGISTERED DECISION RULE:")
    print(f"  - >= {MIN_FOLDS} folds x {MIN_SEEDS} seeds")
    print(f"  - candidate beats baseline AND clears the shuffled-label null "
          f"in >= {MIN_PASSING_FOLDS}/{MIN_FOLDS} folds (per seed)")
    print(f"  - >= {MIN_ONSET_POSITIVES_IN_COVERAGE} onset positives inside "
          f"the feature's own history coverage")
    print(f"  - continuation AUC must not drop by more than "
          f"{MAX_CONTINUATION_AUC_DROP}")
    print()

    if not a.data_dir.exists():
        print(f"{a.data_dir} does not exist yet -- nothing has landed. Run "
              f".github/workflows/harvest-newdata.yml (workflow_dispatch, or "
              f"wait for its daily schedule) to produce it.")
        return 0

    import pandas as pd
    found = sorted(a.data_dir.glob("*.parquet"))
    if not found:
        print(f"{a.data_dir} exists but is empty -- the harvest workflow has "
              f"not committed anything yet.")
        return 0

    print(f"found in {a.data_dir}:")
    for f in found:
        d = pd.read_parquet(f)
        if "ts" in d.columns and len(d):
            lo = pd.Timestamp(int(d["ts"].min()), unit="s", tz="UTC")
            hi = pd.Timestamp(int(d["ts"].max()), unit="s", tz="UTC")
            span_days = (d["ts"].max() - d["ts"].min()) / 86400
            print(f"  {f.name}: {len(d):,} rows, {lo.date()} -> {hi.date()} "
                  f"({span_days:.0f} days)")
        else:
            print(f"  {f.name}: {len(d):,} rows (no usable ts column)")
    print("\nStep 2 onward above (join, walk-forward, decision_rule) is not "
          "yet implemented here -- this prints what has landed so that work "
          "can start from a measured, not assumed, coverage window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

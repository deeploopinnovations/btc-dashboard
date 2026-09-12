"""
noctua/regenerate.py
=====================================================================
Rebuild the research corpus from its public source, and REFUSE if the rebuild
is not the same corpus the committed results were measured on.

WHY THIS EXISTS

On 2026-09-12 `model/artifacts/` was lost with the container: the 476,362-episode
corpus, `teacher_oof.npz`, every result artifact, and the 7.68M-minute source
(DATA_LOSS_2026-09-12.md). Two adversarial audits had to report four of their
eight attacks as NOT TESTABLE, including every attack on
`P2-scorecard-rescaled-result` -- the single load-bearing number in Phase 2, the
one that overturned the Phase 1 diagnosis and ended teacher mining. It has never
been independently recomputed.

Re-downloading the source is easy. Knowing you re-downloaded the SAME source is
the part that needs code, for two reasons that both bite here.

REASON ONE: THE SOURCE MOVES FORWARD AND THE CORPUS MUST NOT

`ff137/bitstamp-btcusd-minute-data` updates daily. The lost corpus ended
2026-08-09 14:37 UTC at 7,681,837 minutes; a clone taken on 2026-09-12 runs to
2026-09-12 01:36 and carries 48,179 extra minutes -- of which **21,697 sit at or
after 2026-08-28, inside the forward holdout that gets exactly one evaluation
(R26)**. A rebuild that simply ingests whatever the source currently holds
silently spends the holdout and silently breaks comparability with every
committed number. So the research corpus is pinned to a DATE, the row count is
the cross-check on that date, and the remainder is written to a separate file
whose name says what it is.

Extending the corpus to the holdout boundary (2026-08-27) is available and would
add 18 days. It is NOT done here, because it would change every committed result
while leaving every committed result's filename unchanged -- the most expensive
kind of cheap win.

REASON TWO: A REBUILD THAT DIFFERS SILENTLY IS WORSE THAN NO REBUILD

`P2-dataset-audit` published integrity statistics BEFORE the loss, which makes
them usable as out-of-sample targets for the rebuild now (R46: validate a
rebuilt component against something it could not have been fitted to). Three
reproduce to the precision they were quoted at:

    filled minutes      0          published: 0 across all 7,681,837 minutes
    zero-volume         17.0846%   published: 17.08%
    bad prints          0.1505%    published: 0.15%

Note the third is a TIGHTER test than it looks: on the FULL rebuilt corpus the
zero-volume rate is 16.98%, not 17.08%, because 33 extra days of 2026 (0.48%
zero-volume) dilute it. The published figure is only recovered on the correctly
truncated slice, so this statistic distinguishes the right truncation from a
near-miss rather than merely confirming the download.

    python -m model.noctua.regenerate --repo <clone> --out model/artifacts
    python -m model.noctua.regenerate --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ------------------------------------------------------- the pinned corpus
# The date is the definition; the row count is the cross-check on it.
CORPUS_END = pd.Timestamp("2026-08-09 14:37:00", tz="UTC")
CORPUS_ROWS = 7_681_837
HOLDOUT_START = pd.Timestamp("2026-08-28", tz="UTC")

# Published by P2-dataset-audit before the artifacts were lost. (value, tol)
# Tolerances are the quoting precision of the published figure, not a margin
# chosen to make the check pass.
TARGETS = {
    "filled_minutes": (0.0, 0.0),
    "zero_volume_pct": (17.08, 0.005),
    "bad_print_pct": (0.15, 0.005),
}

RESEARCH = "btcusd_1min.parquet"
UNSEEN = "btcusd_1min_AFTER_CORPUS_END_do_not_use_in_research.parquet"
MANIFEST = "corpus_manifest.json"


def content_sha256(d: pd.DataFrame) -> str:
    """Hash the DATA, not the file.

    Parquet bytes depend on the writer's version and compression settings, so a
    file hash would report a spurious mismatch on a pyarrow bump. This hashes
    the canonical column contents in a fixed order instead.
    """
    h = hashlib.sha256()
    for c in sorted(d.columns):
        col = d[c]
        a = (col.to_numpy(np.float64) if col.dtype.kind == "f"
             else col.to_numpy(np.int64) if col.dtype.kind in "iu"
             else col.to_numpy(np.uint8) if col.dtype.kind == "b"
             else col.astype(str).to_numpy().astype("S32"))
        h.update(c.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def corpus_stats(d: pd.DataFrame) -> dict:
    return {
        "rows": int(len(d)),
        "filled_minutes": float(d["filled"].sum()),
        "zero_volume_pct": 100.0 * float((d["volume"] == 0).mean()),
        "bad_print_pct": (100.0 * float(d["bad_print"].mean())
                          if "bad_print" in d else float("nan")),
    }


def check_fidelity(stats: dict) -> list[tuple[str, bool, str]]:
    out = []
    for k, (want, tol) in TARGETS.items():
        got = stats.get(k, float("nan"))
        ok = np.isfinite(got) and abs(got - want) <= tol
        out.append((k, bool(ok), f"{got:.4f} against published {want} "
                                 f"(tol {tol})"))
    return out


def truncate(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split at the pinned corpus end. Raises if the pin is not consistent."""
    ts = pd.to_datetime(d["timestamp"], unit="s", utc=True)
    keep = ts <= CORPUS_END
    research, rest = d.loc[keep].reset_index(drop=True), d.loc[~keep].reset_index(drop=True)
    info = {
        "rows_research": int(len(research)),
        "rows_after_end": int(len(rest)),
        "rows_in_holdout": int((ts >= HOLDOUT_START).sum()),
        "research_end_utc": str(ts.loc[keep].iloc[-1]) if len(research) else None,
        "source_end_utc": str(ts.iloc[-1]) if len(d) else None,
    }
    if len(research) != CORPUS_ROWS:
        raise SystemExit(
            f"REFUSING: the pinned date {CORPUS_END} yields "
            f"{len(research):,} minutes, not the {CORPUS_ROWS:,} the committed "
            f"results were measured on. The source's history has been revised "
            f"upstream -- the two pins disagree and one of them is wrong. Do "
            f"not proceed by relaxing whichever is inconvenient.")
    if (pd.to_datetime(research["timestamp"], unit="s", utc=True)
            >= HOLDOUT_START).any():
        raise SystemExit("REFUSING: research corpus crosses the holdout start.")
    return research, rest, info


def regenerate(repo: Path, out: Path, skip_ingest: bool = False) -> int:
    from noctua.ingest import ingest

    out.mkdir(parents=True, exist_ok=True)
    raw = out / RESEARCH
    if not skip_ingest:
        print(f"[regen] stage 1: ingest from {repo}")
        ingest(repo, out)
    if not raw.exists():
        print(f"[regen] MISSING {raw}", file=sys.stderr)
        return 2

    d = pd.read_parquet(raw)
    research, rest, info = truncate(d)
    stats = corpus_stats(research)
    rows = check_fidelity(stats)

    print(f"\n[regen] pinned corpus end {CORPUS_END}  ({CORPUS_ROWS:,} minutes)")
    print(f"[regen] source runs to     {info['source_end_utc']}")
    print(f"[regen] set aside          {info['rows_after_end']:,} minutes, of "
          f"which {info['rows_in_holdout']:,} are inside the holdout")
    print(f"\n[regen] fidelity against P2-dataset-audit's published statistics:")
    bad = 0
    for k, ok, detail in rows:
        if not ok:
            bad += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {k}: {detail}")
    if bad:
        raise SystemExit(
            f"REFUSING: {bad} published statistic(s) did not reproduce. This "
            f"rebuild is NOT the corpus the committed results were measured "
            f"on, and running research on it would silently compare numbers "
            f"from two different datasets.")

    research.to_parquet(raw, index=False, compression="zstd")
    if len(rest):
        rest.to_parquet(out / UNSEEN, index=False, compression="zstd")
    digest = content_sha256(research)
    man = {"corpus_end_utc": str(CORPUS_END), "corpus_rows": CORPUS_ROWS,
           "holdout_start_utc": str(HOLDOUT_START),
           "content_sha256": digest, "stats": stats, "split": info,
           "source": "ff137/bitstamp-btcusd-minute-data (MIT)",
           "published_targets": {k: v[0] for k, v in TARGETS.items()}}
    (Path("model/research") / MANIFEST).write_text(
        json.dumps(man, indent=1, default=float) + "\n")
    print(f"\n[regen] content sha256 {digest}")
    print(f"[regen] wrote {raw} ({len(research):,} rows)")
    print(f"[regen] wrote model/research/{MANIFEST}  <- committed, so the next")
    print(f"        rebuild can be compared against this one byte for byte")
    return 0


def selftest() -> int:
    """The gates must refuse the failure modes they exist for."""
    rng = np.random.default_rng(5)

    def frame(n, t0=0, zero_frac=0.1708, bad_frac=0.0015):
        d = pd.DataFrame({
            "timestamp": np.arange(t0, t0 + 60 * n, 60, dtype=np.int64),
            "close": 100 + rng.normal(0, 1, n),
            "volume": np.where(rng.random(n) < zero_frac, 0.0, 1.0),
            "filled": np.zeros(n, bool),
            "bad_print": rng.random(n) < bad_frac,
        })
        return d

    checks = []

    # 1. a corpus matching the published stats passes
    n = CORPUS_ROWS
    end = int(CORPUS_END.timestamp())
    good = frame(n, t0=end - 60 * (n - 1))
    good["volume"] = 1.0
    good.iloc[:int(round(n * 0.170846)),
              good.columns.get_loc("volume")] = 0.0
    good["bad_print"] = False
    good.iloc[:int(round(n * 0.001505)),
              good.columns.get_loc("bad_print")] = True
    st = corpus_stats(good)
    ok_rows = check_fidelity(st)
    checks.append(("faithful-corpus-passes", all(o for _, o, _ in ok_rows),
                   "; ".join(f"{k}={d.split(' ')[0]}" for k, _, d in ok_rows)))

    # 2. the untruncated corpus must FAIL -- this is the check that the
    #    zero-volume target actually discriminates the right truncation
    extra = frame(48_179, t0=end + 60, zero_frac=0.0048)
    wide = pd.concat([good, extra], ignore_index=True)
    st_w = corpus_stats(wide)
    zv = [r for r in check_fidelity(st_w) if r[0] == "zero_volume_pct"][0]
    checks.append(("untruncated-corpus-is-REJECTED", not zv[1],
                   f"zero-volume {st_w['zero_volume_pct']:.4f}% on the wide "
                   f"corpus vs {st['zero_volume_pct']:.4f}% truncated -- the "
                   f"statistic separates them"))

    # 3. truncate() refuses when the date pin and the row pin disagree
    try:
        truncate(pd.concat([good.iloc[:-10], extra], ignore_index=True))
        refused = False
    except SystemExit:
        refused = True
    checks.append(("pin-disagreement-is-REFUSED", refused,
                   "a revised upstream history cannot pass silently"))

    # 4. truncate() keeps the holdout out, and says how much it set aside
    res, rest, info = truncate(wide)
    checks.append(("holdout-excluded-from-research",
                   len(res) == CORPUS_ROWS and info["rows_after_end"] == 48_179,
                   f"{len(res):,} kept, {info['rows_after_end']:,} set aside"))

    # 5. the content hash is insensitive to file encoding but sensitive to data
    h1 = content_sha256(good)
    h2 = content_sha256(good.copy())
    bumped = good.copy()
    bumped.loc[bumped.index[7], "close"] += 1e-9
    checks.append(("content-hash-stable-and-sensitive",
                   h1 == h2 and content_sha256(bumped) != h1,
                   f"identical frames agree; a 1e-9 change in one cell does not"))

    print("regenerate selftest")
    bad = 0
    for name, ok, detail in checks:
        if not ok:
            bad += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n{len(checks) - bad}/{len(checks)} checks passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="rebuild the pinned research corpus")
    p.add_argument("--repo", type=Path, help="clone of the source dataset")
    p.add_argument("--out", type=Path, default=Path("model/artifacts"))
    p.add_argument("--skip-ingest", action="store_true",
                   help="corpus parquet already written; verify and pin it")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.repo and not a.skip_ingest:
        p.error("--repo is required unless --skip-ingest")
    return regenerate(a.repo, a.out, a.skip_ingest)


if __name__ == "__main__":
    sys.exit(main())

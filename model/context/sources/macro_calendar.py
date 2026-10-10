#!/usr/bin/env python3
"""
Build data/context/macro_calendar.parquet: US macro release calendar, 2014 on.

Sources (free, no key, no account):
  fomc : federalreserve.gov FOMC history pages (2014-2020, headings mark
         scheduled / unscheduled / cancelled meetings), the calendars page
         (2021 onward statements, plus the scheduled meetings still to come
         in the current year).
  bls  : ALFRED vintages of CPIAUCSL (CPI-U, SA) and PAYEMS (total nonfarm).
         The ALFRED vintage date on which a new reference month first appears
         is that month's release date; the first-release value is read from
         that same vintage (MoM % for CPI, level change in thousands for NFP).

Series written (source / series):
  fomc / fomc_statement       scheduled statement, 14:00 ET; pit_quality scheduled
  fomc / fomc_unscheduled     unscheduled rate statement; exact or lagged
  fomc / fomc_upper_target    upper bound of the target range at each decision
  bls  / cpi_release          CPI release (scheduled; available Jan 1 of year)
  bls  / cpi_mom_first        CPI-U SA MoM % as first released; 08:30 ET
  bls  / nfp_release          Employment Situation (scheduled)
  bls  / nfp_first            total nonfarm payroll change, thousands, first release

Re-run:  python3 model/context/sources/macro_calendar.py
"""
from __future__ import annotations

import csv
import html
import io
import re
import sys
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "model"))
from context import pit  # noqa: E402  (validator from model/context/pit.py)

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
FED = "https://www.federalreserve.gov"
ALFRED = "https://alfred.stlouisfed.org"
START = "2014-01-01"          # first release date kept
VINTAGE_FROM = "2013-01-01"   # baseline vintages before START are only used to seed "seen" months
RETRIEVED = int(time.time())
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# Announcement times of unscheduled rate moves. The Fed press-release pages do
# not print a time; these are from the Fed's public record, not parsed from the
# source, so they are flagged in the report.
UNSCHEDULED_TIME = {
    "2020-03-03": (10, 0),   # intermeeting 50 bp cut, Tuesday morning
    "2020-03-15": (17, 0),   # emergency cut to 0-1/4, Sunday evening
}


def fetch(url: str, tries: int = 6) -> str:
    """GET via curl (default user agent: ALFRED drops custom UA strings; urllib is reset by the proxy)."""
    err = None
    for i in range(tries):
        res = subprocess.run(
            ["curl", "-sSLf", "--http1.1", "--max-time", "90", url],
            capture_output=True,
        )
        if res.returncode == 0:
            return res.stdout.decode("utf-8", "replace")
        err = res.stderr.decode("utf-8", "replace").strip()
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"fetch failed: {url}: {err}")


def clean(s: str) -> str:
    s = re.sub(r"<script.*?</script>|<style.*?</style>", " ", s, flags=re.S)
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    for bad in ("‑", "–", "−"):
        s = s.replace(bad, "-")
    return " ".join(s.replace("\xa0", " ").split())


def et_ts(y: int, m: int, d: int, hh: int = 0, mm: int = 0) -> int:
    return int(datetime(y, m, d, hh, mm, tzinfo=ET).timestamp())


def jan1_utc(year: int) -> int:
    return int(datetime(year, 1, 1, tzinfo=UTC).timestamp())


# ---------------------------------------------------------------- FOMC ----

RATE_PATTERNS = [
    # "decided (today) to raise the target range ... by 1/4 percentage point, to 4 to 4-1/4 percent"
    r"decided (?:today )?to \w+ the target range for the federal funds rate(?: by [^.;]{1,30}?)?,? (?:to|at) ([^.;]{1,40}?) percent",
    # "target range for the federal funds rate at 0 to 1/4 percent"
    r"target range for the federal funds rate (?:to|at|of),? ([^.;]{1,40}?) percent",
    # "the current 0 to 1/4 percent target range for the federal funds rate"
    r"(?:current|the) ([0-9][^.;]{0,20}?) percent target range for the federal funds rate",
]
# Only sentences that name the target range for the federal funds rate count.
# Notation-vote and implementation text ("...maintain the federal funds rate in a
# target range of X") is not a rate decision and is skipped.
TOKEN = re.compile(
    r'<h5[^>]*>(?P<h>.*?)</h5>|'
    r'<a href="(?P<href>/newsevents/pressreleases/monetary(?P<d>\d{8})(?P<s>[a-z])\.htm)"',
    re.S,
)


def num(s: str) -> float:
    s = s.strip()
    m = re.fullmatch(r"(?:(\d+)-)?(\d+)/(\d+)", s)          # "1-1/2", "1/4"
    if m:
        return int(m.group(1) or 0) + int(m.group(2)) / int(m.group(3))
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)
    raise ValueError(f"cannot parse rate '{s}'")


def target_range(text: str):
    t = clean(text)
    for p in RATE_PATTERNS:
        m = re.search(p, t)
        if m:
            parts = re.split(r"\s+to\s+", m.group(1).strip())
            if len(parts) == 2:
                lo, hi = num(parts[0]), num(parts[1])
                if lo > hi:
                    raise ValueError(f"inverted range in {m.group(0)!r}")
                return lo, hi
    return None


def statement_candidates():
    """url -> (YYYYMMDD, kind, page). Only the 'a' statement of each release."""
    cands = {}
    for y in range(2014, 2021):
        page = f"{FED}/monetarypolicy/fomchistorical{y}.htm"
        t = fetch(page)
        head = None
        for m in TOKEN.finditer(t):
            if m.group("h") is not None:
                head = clean(m.group("h"))
                continue
            if m.group("s") != "a" or (head and "cancel" in head.lower()):
                continue
            kind = "unscheduled" if head and re.search(r"unscheduled|notation", head, re.I) else "scheduled"
            cands[FED + m.group("href")] = (m.group("d"), kind, page)
    page = f"{FED}/monetarypolicy/fomccalendars.htm"
    t = fetch(page)
    for m in TOKEN.finditer(t):
        if m.group("h") is None and m.group("s") == "a":
            cands.setdefault(FED + m.group("href"), (m.group("d"), "scheduled", page))
    return cands


def future_meetings(today: date):
    """Meetings still to come on the calendars page: (sorted statement dates, page url)."""
    t = fetch(f"{FED}/monetarypolicy/fomccalendars.htm")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    lines = [html.unescape(re.sub(r"<[^>]+>", "", l)).strip()
             for l in re.sub(r"<br\s*/?>|</(p|div|li|tr|td|h\d)>", "\n", t).split("\n")]
    lines = [l for l in lines if l]
    year = month = None
    out = []
    for l in lines:
        m = re.fullmatch(r"(\d{4}) FOMC Meetings", l)
        if m:
            year, month = int(m.group(1)), None
            continue
        if l in MONTHS:
            month = MONTHS.index(l) + 1
            continue
        m = re.fullmatch(r"(\d{1,2})-(\d{1,2})\*?", l)
        if m and year and month:
            d1, d2 = int(m.group(1)), int(m.group(2))
            end_m, end_y = (month, year) if d2 >= d1 else ((month % 12) + 1, year + (month == 12))
            sd = date(end_y, end_m, d2)
            if sd > today and sd.year <= today.year:   # not yet released; schedule known
                out.append(sd)
    return sorted(set(out)), f"{FED}/monetarypolicy/fomccalendars.htm"


def build_fomc(rows: list, today: date):
    cands = statement_candidates()
    stmts = []
    skipped = []
    with ThreadPoolExecutor(6) as ex:
        urls = sorted(cands, key=lambda u: cands[u][0])
        texts = list(ex.map(fetch, urls))
    for url, text in zip(urls, texts):
        ymd, kind, _page = cands[url]
        tr = target_range(text)
        if tr is None:
            skipped.append(url)
            continue
        stmts.append((ymd, kind, url, tr))

    for ymd, kind, url, (lo, hi) in stmts:
        d = date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
        iso = d.isoformat()
        if kind == "scheduled":
            t_ev = et_ts(d.year, d.month, d.day, 14, 0)      # FOMC statements: 2:00 p.m. ET
            q = "exact"
            rows.append(dict(source="fomc", series="fomc_statement", event_ts=t_ev,
                             available_ts=jan1_utc(d.year), value=1.0,
                             text=f"FOMC statement, scheduled meeting ending {iso}",
                             source_url=url, pit_quality="scheduled"))
            # Upper bound of target range decided at this meeting: known at release.
            rows.append(dict(source="fomc", series="fomc_upper_target", event_ts=t_ev,
                             available_ts=t_ev, value=float(hi),
                             text=f"target range {lo:g}-{hi:g} percent",
                             source_url=url, pit_quality="exact"))
        else:
            hh, mm = UNSCHEDULED_TIME[iso]                  # KeyError if an unknown move appears
            t_ev = et_ts(d.year, d.month, d.day, hh, mm)
            q = "exact"
            rows.append(dict(source="fomc", series="fomc_unscheduled", event_ts=t_ev,
                             available_ts=t_ev, value=1.0,
                             text=f"unscheduled FOMC statement {iso}; target range {lo:g}-{hi:g} percent",
                             source_url=url, pit_quality=q))
            rows.append(dict(source="fomc", series="fomc_upper_target", event_ts=t_ev,
                             available_ts=t_ev, value=float(hi),
                             text=f"target range {lo:g}-{hi:g} percent (unscheduled)",
                             source_url=url, pit_quality=q))

    known = {date(int(s[0][:4]), int(s[0][4:6]), int(s[0][6:])) for s in stmts}
    future, page = future_meetings(today)
    for sd in future:
        if sd in known:
            continue
        rows.append(dict(source="fomc", series="fomc_statement",
                         event_ts=et_ts(sd.year, sd.month, sd.day, 14, 0),
                         available_ts=jan1_utc(sd.year), value=1.0,
                         text=f"FOMC meeting ending {sd.isoformat()} (scheduled, statement not yet released)",
                         source_url=page, pit_quality="scheduled"))
    print(f"fomc: {len(stmts)} rate statements, skipped non-rate pages: {len(skipped)}")
    return stmts


# ---------------------------------------------------------- ALFRED / BLS ---

def prev_month(m: str) -> str:
    y, mo = int(m[:4]), int(m[5:7])
    return f"{y - 1}-12" if mo == 1 else f"{y}-{mo - 1:02d}"


def alfred_first_releases(series: str):
    """Yield (vintage_date, csv_url, [new months], levels) for each release."""
    page = f"{ALFRED}/series/downloaddata?seid={series}"
    vints = sorted(set(re.findall(r'<option value="(\d{4}-\d{2}-\d{2})"', fetch(page))))
    vints = [v for v in vints if v >= VINTAGE_FROM]

    def get(v):
        url = f"{ALFRED}/graph/alfredgraph.csv?id={series}&vintage_date={v}"
        levels = {}
        for row in csv.reader(io.StringIO(fetch(url))):
            if not row or not row[0][:4].isdigit():
                continue
            try:
                levels[row[0][:7]] = float(row[1])     # '.' (missing) is skipped
            except (ValueError, IndexError):
                continue
        return url, levels

    with ThreadPoolExecutor(3) as ex:
        data = dict(zip(vints, ex.map(get, vints)))
    seen: set = set()
    out = []
    for i, v in enumerate(vints):
        url, lv = data[v]
        new = sorted(m for m in lv if m not in seen)
        seen |= set(lv)
        if i == 0 or not new or v < START:
            continue
        out.append((v, url, new, lv))
    return out


def build_bls(rows: list):
    specs = [
        ("CPIAUCSL", "cpi", "CPI-U (SA)"),
        ("PAYEMS", "nfp", "Employment Situation (total nonfarm)"),
    ]
    for series, label, title in specs:
        n_rel = n_first = n_skip = n_dropped = 0
        for v, url, new, lv in alfred_first_releases(series):
            y, mo, d = int(v[:4]), int(v[5:7]), int(v[8:10])
            t_ev = et_ts(y, mo, d, 8, 30)                  # BLS releases: 08:30 ET
            rows.append(dict(source="bls", series=f"{label}_release", event_ts=t_ev,
                             available_ts=jan1_utc(y), value=1.0,
                             text=f"{title} release; reference month(s) {', '.join(new)}",
                             source_url=url, pit_quality="scheduled"))
            n_rel += 1
            firsts = []
            for m in new:
                pm = prev_month(m)
                if pm not in lv:
                    n_skip += 1          # prior month never published: no first-release change
                    continue
                if series == "CPIAUCSL":
                    val = (lv[m] / lv[pm] - 1.0) * 100.0
                    s = f"{label}_mom_first"
                    txt = f"CPI-U SA MoM % first release for {m}"
                else:
                    val = lv[m] - lv[pm]
                    s = f"{label}_first"
                    txt = f"total nonfarm payroll change (thousands), first release for {m}"
                firsts.append((m, s, float(val), txt))
            # One release can carry several reference months (BLS published Oct and
            # Nov 2025 payrolls together). Rows with the same timestamp would make
            # pit.asof ambiguous, so keep the latest reference month only.
            if firsts:
                m, s, val, txt = firsts[-1]
                if len(firsts) > 1:
                    n_dropped += len(firsts) - 1
                    print(f"  {series} {v}: joint release, kept {m}, dropped {[f[0] for f in firsts[:-1]]}")
                rows.append(dict(source="bls", series=s, event_ts=t_ev, available_ts=t_ev,
                                 value=val, text=txt, source_url=url, pit_quality="exact"))
                n_first += 1
        print(f"bls {series}: {n_rel} releases, {n_first} first-release values, "
              f"{n_skip} months without prior month, {n_dropped} joint-release months dropped")



# 2025-26 US government funding lapses moved or cancelled BLS releases
# (https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm). The
# original dates were the public schedule; a moved date was public only from
# its notice. Unknown notice dates use actual release - 1 day, which is no
# earlier than the real notice. A cancellation is known by the original
# release time at the latest. (audits/macro_calendar.md F1)
LAPSE_URL = "https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm"
LAPSE = [  # series, original date, actual date or None, notice date or None
    ("cpi_release", date(2025, 10, 15), date(2025, 10, 24), date(2025, 10, 10)),
    ("nfp_release", date(2025, 10, 3), date(2025, 11, 20), None),
    ("cpi_release", date(2025, 11, 13), None, None),
    ("nfp_release", date(2025, 11, 7), None, None),
    ("nfp_release", date(2025, 12, 5), date(2025, 12, 16), None),
    ("cpi_release", date(2025, 12, 10), date(2025, 12, 18), None),
    ("nfp_release", date(2026, 2, 6), date(2026, 2, 11), None),
    ("cpi_release", date(2026, 2, 11), date(2026, 2, 13), None),
]


def apply_lapse(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    extra = []
    for series, orig, actual, notice in LAPSE:
        o = et_ts(orig.year, orig.month, orig.day, 8, 30)
        extra.append(dict(source="bls", series=series, event_ts=o, available_ts=jan1_utc(orig.year),
                          value=1.0, text="original schedule", source_url=LAPSE_URL, pit_quality="scheduled"))
        cancel_av = et_ts(notice.year, notice.month, notice.day, 12, 0) if notice else o
        extra.append(dict(source="bls", series=series + "_cancel", event_ts=o, available_ts=min(cancel_av, o),
                          value=1.0, text="moved or cancelled", source_url=LAPSE_URL, pit_quality="scheduled"))
        if actual is not None:
            a = et_ts(actual.year, actual.month, actual.day, 8, 30)
            av = et_ts(notice.year, notice.month, notice.day, 12, 0) if notice else a - 86400
            m = (df.series == series) & (df.event_ts == a)
            df.loc[m, "available_ts"] = av
            df.loc[m, "source_url"] = LAPSE_URL
    extra = pd.DataFrame(extra)
    extra["retrieved_at"] = df.retrieved_at.iloc[0]
    return pd.concat([df, extra], ignore_index=True)


def main():
    today = datetime.fromtimestamp(RETRIEVED, UTC).date()
    rows: list = []
    build_fomc(rows, today)
    build_bls(rows)

    df = pd.DataFrame(rows)
    df["retrieved_at"] = RETRIEVED
    df = apply_lapse(df)
    df = pit.validate(df, "macro_calendar")
    assert not df.duplicated(["source", "series", "event_ts"]).any(), "duplicate event timestamps"
    df = df.sort_values(["series", "event_ts"])
    df = df.reset_index(drop=True)

    out = pit.CONTEXT / "macro_calendar.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()

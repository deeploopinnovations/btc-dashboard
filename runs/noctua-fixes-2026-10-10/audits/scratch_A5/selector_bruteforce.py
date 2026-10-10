"""Claim 2a-d: brute-force check of serve.adaptive._settled_anchors against its stated contract."""
import sys
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
import serve.adaptive as A
from serve.adaptive import _settled_anchors, STRIDE_HOURS, WINDOW_DAYS, MIN_EPISODES
print("STRIDE_HOURS", STRIDE_HOURS, "WINDOW_DAYS", WINDOW_DAYS, "MIN_EPISODES", MIN_EPISODES)
fails = {"causal": 0, "hod": 0, "most_recent_skipped": 0, "window_bound": 0, "floor": 0, "sorted_unique": 0, "count_too_big": 0}
checked = 0; sizes = []
Hs = [1, 2, 6, 12, 19, 23, 24, 25, 30, 47, 48, 49, 96, 168]
for H in Hs:
    for anchor in list(range(721 + H, 721 + H + 60)) + list(range(1000, 9700, 37)) + [9599, 9600, 9601]:
        rows = _settled_anchors(None, anchor, H, WINDOW_DAYS, STRIDE_HOURS)
        checked += 1
        if len(rows) == 0:
            continue
        sizes.append(len(rows))
        if not np.all(rows + H <= anchor): fails["causal"] += 1
        if not np.all((rows - anchor) % 24 == 0): fails["hod"] += 1
        if rows.min() < max(720, anchor - H - WINDOW_DAYS * 24): fails["window_bound"] += 1
        if rows.min() < 720: fails["floor"] += 1
        if len(np.unique(rows)) != len(rows) or np.any(np.diff(rows) <= 0): fails["sorted_unique"] += 1
        if len(rows) > WINDOW_DAYS: fails["count_too_big"] += 1
        # most recent same-hour row that is settled and inside the window
        last = anchor - H
        cands = [r for r in range(anchor - 24, 720 - 1, -24) if r <= last and r >= max(720, last - WINDOW_DAYS * 24)]
        if cands and rows.max() != max(cands): fails["most_recent_skipped"] += 1
        if (not cands) and len(rows): fails["most_recent_skipped"] += 1
print("anchors/H checked:", checked)
print("failures:", fails)
print("row-count range when non-empty: min", min(sizes), "max", max(sizes))
# gate behaviour: how many anchors are declined by MIN_EPISODES in the selector alone
declined = sum(1 for a in range(720 + 19, 720 + 19 + 24 * 40) if len(_settled_anchors(None, a, 19, WINDOW_DAYS, STRIDE_HOURS)) < MIN_EPISODES)
print("H=19 anchors in first 40 days after floor declined by selector gate:", declined, "of", 24 * 40)
# H > 24 example: the most recent settled same-hour row for H=30 at anchor 5000 (hour-of-day 8)
print("H=30 anchor 5000 newest row:", int(_settled_anchors(None, 5000, 30, 60, 24).max()), "=> offset", 5000 - int(_settled_anchors(None, 5000, 30, 60, 24).max()))
print("H=19 anchor 5000 newest row offset:", 5000 - int(_settled_anchors(None, 5000, 19, 60, 24).max()))
print("H=19 anchor 5000 first row offset:", 5000 - int(_settled_anchors(None, 5000, 19, 60, 24).min()))

"""
tests/test_weekend_column.py
=====================================================================
Pins a KNOWN DEFECT so it cannot be "fixed" silently (P4-weekend-bug), and
pins its FIX: the corrected Sat+Sun column `cal_weekend_frac_ss` (2026-10-10),
which every newly trained artifact reads instead.

noctua/features.py's cal_weekend_frac counts FRIDAY and SATURDAY, not the
weekend: its weekday formula uses offset +4 where the epoch (a Thursday) needs
+3. The shipped network and anchor were trained on that column, and serving
computes it with the same function, so train and serve agree. Editing the
formula in place would feed the shipped network a column it never saw -- a
train/serve skew that no other gate would catch, because every consumer would
move together.

THE CONTRACT
  * the shipped column equals the Fri+Sat fraction, checked against pandas
    dayofweek -- an INDEPENDENT calendar, not a re-implementation of the same
    formula (which is how eval/vol_matrix's check missed it);
  * changing it requires retraining the artifact AND updating this test in the
    same change, with the ledger entry that justified it.

AND THE GATE CAN FAIL (R2): the true Sat+Sun fraction must NOT pass the pin.

    python model/tests/test_weekend_column.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noctua.features import build_features  # noqa: E402

HOUR = 3600


def independent_frac(anchor_ts, H, days) -> np.ndarray:
    """Fraction of forward-window hours whose pandas weekday is in `days`."""
    out = []
    for t, h in zip(anchor_ts, H):
        hrs = pd.to_datetime(int(t) + HOUR * np.arange(int(h)), unit="s", utc=True)
        out.append(np.isin(hrs.dayofweek, days).mean())
    return np.array(out)


def main() -> int:
    from serve.history import load_bundle
    hours = load_bundle()
    hour_ts = hours["hour_ts"].to_numpy(np.int64)
    rng = np.random.default_rng(3)
    rows = np.sort(rng.choice(np.arange(24 * 40, len(hours) - 30), 60, replace=False))
    dt = pd.to_datetime(hour_ts[rows], unit="s", utc=True)
    H = rng.choice([6, 12, 19, 24], len(rows))
    ep = pd.DataFrame({"anchor_ts": hour_ts[rows], "H": H, "row": rows, "dt": dt,
                       "anchor_hour": dt.hour, "dow": dt.dayofweek})
    F = build_features(hours, ep)
    col = F["cal_weekend_frac"].to_numpy(np.float64)
    col_ss = F["cal_weekend_frac_ss"].to_numpy(np.float64)
    fri_sat = independent_frac(hour_ts[rows], H, (4, 5))
    sat_sun = independent_frac(hour_ts[rows], H, (5, 6))
    ok = []
    ok.append(("shipped cal_weekend_frac is the Fri+Sat fraction (KNOWN DEFECT, "
               "P4-weekend-bug)", np.allclose(col, fri_sat, atol=1e-12)))
    ok.append(("the gate can fail: the true Sat+Sun fraction does not pass the pin",
               not np.allclose(sat_sun, fri_sat, atol=1e-12)))
    # THE FIX (2026-10-10): a NEW column, so the legacy one above stays pinned
    # for the v2 artifacts while newly trained artifacts read the correct one.
    ok.append(("cal_weekend_frac_ss is the true Sat+Sun fraction",
               np.allclose(col_ss, sat_sun, atol=1e-12)))
    ok.append(("the fix's gate can fail: the Fri+Sat fraction does not pass it",
               not np.allclose(col_ss, fri_sat, atol=1e-12)))
    from noctua.spec import BASE_COLS, NON_MODEL_COLS, SHAPE_COLS
    ok.append(("new training reads the corrected column, never the legacy one",
               "cal_weekend_frac_ss" in BASE_COLS and "cal_weekend_frac_ss" in SHAPE_COLS
               and "cal_weekend_frac" not in BASE_COLS + SHAPE_COLS
               and "cal_weekend_frac" in NON_MODEL_COLS))
    ok.append(("the sample exercises weekend windows",
               bool((sat_sun > 0).sum() >= 5 and (fri_sat != sat_sun).sum() >= 5)))
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    if bad:
        print("\nIf cal_weekend_frac was changed ON PURPOSE: retrain the artifact in "
              "the same change and update this pin (research/ledger.json P4-weekend-bug).")
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

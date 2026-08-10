#!/usr/bin/env python3
"""
model/tests/test_serving.py
=====================================================================
Self-test for the served model. No market data, no training parquet, no
PyTorch -- exactly the dependency set the GitHub Action and the Hugging Face
Space have.

What it actually checks (each of these is a property that, if it broke
silently, would produce plausible-looking but wrong strike recommendations):

  1. the exported weights load and carry the expected metadata
  2. every quantile head is MONOTONE -- quantile crossing would make the
     survival curve non-invertible and `safe_level` meaningless
  3. m_up and m_dn are non-negative, as excursions must be
  4. barrier survival is DECREASING in the barrier distance -- a further-away
     level can never be more likely to break
  5. `safe_level` actually inverts `touch_prob` to the requested alpha
  6. tail extrapolation stays finite and in [0,1] far outside the quantile grid
  7. the legacy JSON contract still satisfies scripts/smoke.js's constraints,
     including `upside == 50.0`, which src/data.js pipes into strike selection
  8. PyTorch is never imported. The serving path advertises NumPy + SciPy only,
     and both the HF Space and the GitHub Action install exactly that. This
     check exists because the invariant already broke once: noctua/infer.py
     had a module-level `import torch`, the local suite passed anyway (torch is
     installed in the training environment), and only CI caught it. Asserting
     it here means the training environment catches it too.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.predict import to_legacy                      # noqa: E402
from serve.runtime import NumpyNoctua                    # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  -> ' + detail}")
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("NOCTUA serving self-test\n")

    check("torch not imported by the serving path", "torch" not in sys.modules,
          "something on the serve path imports PyTorch")

    m = NumpyNoctua(ROOT / "serve" / "noctua_weights.npz")
    check("weights load", True)
    check("param count is small", m.meta["n_params"] < 1_000_000, str(m.meta["n_params"]))
    check("blend weight in range", 0.0 <= m.blend_w <= 1.0, str(m.blend_w))
    check("calibration shrink in range", 0.0 <= m.cal_shrink <= 1.0, str(m.cal_shrink))

    # ---- synthetic but structurally valid inputs --------------------------
    rng = np.random.default_rng(0)
    n = 64
    d = {
        "Xa": rng.normal(0, 1, (n, len(m.feat_cols))),
        "Xb": rng.normal(0, 1, (n, len(m.base_cols))),
        "Xs": rng.normal(0, 1, (n, len(m.shape_cols))),
        "H": np.full(n, 19.0),
    }
    t0 = time.time()
    pred = m.predict(d)
    dt = (time.time() - t0) / n
    check("inference under 200ms/episode", dt < 0.2, f"{1000*dt:.1f} ms")

    # ---- 2/3: monotonicity and sign --------------------------------------
    check("Stage A quantiles monotone", bool((np.diff(pred["qa"], axis=1) > 0).all()))
    for key in ("q_r", "q_up", "q_dn"):
        check(f"{key} monotone", bool((np.diff(pred[key], axis=2) > 0).all()))
    check("m_up non-negative", bool((pred["q_up"] >= 0).all()))
    check("m_dn non-negative", bool((pred["q_dn"] >= 0).all()))
    check("sigma positive and finite",
          bool(np.all(pred["sigma_med"] > 0) and np.all(np.isfinite(pred["sigma_med"]))))

    # ---- 4: survival decreasing in barrier distance ----------------------
    grid = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25])
    for up in (True, False):
        curves = np.stack([m.touch_prob(pred, np.full(n, u), up) for u in grid], axis=1)
        check(f"touch prob decreasing in distance ({'up' if up else 'dn'})",
              bool((np.diff(curves, axis=1) <= 1e-9).all()))
        check(f"touch prob in [0,1] ({'up' if up else 'dn'})",
              bool(np.all((curves >= 0) & (curves <= 1))))

    # ---- 5: safe_level inverts touch_prob --------------------------------
    for alpha in (0.01, 0.05, 0.20):
        for up in (True, False):
            lvl = m.safe_level(pred, alpha, up)
            got = m.touch_prob(pred, lvl, up)
            check(f"safe_level inverts touch_prob (alpha={alpha}, {'up' if up else 'dn'})",
                  bool(np.abs(got - alpha).max() < 0.02), f"max dev {np.abs(got-alpha).max():.4f}")
            check(f"safe_level positive (alpha={alpha}, {'up' if up else 'dn'})",
                  bool(np.all(lvl > 0)))

    # deeper alpha must give a further level
    l1 = m.safe_level(pred, 0.01, True)
    l5 = m.safe_level(pred, 0.05, True)
    check("safer alpha implies further level", bool(np.all(l1 >= l5 - 1e-9)))

    # ---- 6: far-tail extrapolation stays sane ----------------------------
    far = m.touch_prob(pred, np.full(n, 5.0), True)
    check("far tail finite and in [0,1]",
          bool(np.all(np.isfinite(far)) and np.all((far >= 0) & (far <= 1))))
    check("far tail is small", bool(far.max() < 0.05), f"max {far.max():.4f}")

    p_up = m.prob_up(pred)
    check("prob_up in [0,1]", bool(np.all((p_up >= 0) & (p_up <= 1))))

    # re-check AFTER a full predict: the lazy import lives inside a code path
    # that only a real forecast exercises
    check("torch still not imported after a full predict", "torch" not in sys.modules,
          "predict path pulled in PyTorch")

    # ---- 7: legacy JSON contract ----------------------------------------
    fake = {
        "anchor_utc": "2026-08-10 17:00:00+00:00", "p_up": 0.61,
        "p_vol_amplify": 0.42, "spot": 65000.0,
    }
    legacy = to_legacy(fake)
    check("legacy upside pinned to 50", legacy["upside"] == 50.0, str(legacy["upside"]))
    check("legacy p_up_raw preserved", legacy["p_up_raw"] == 61.0, str(legacy["p_up_raw"]))
    check("legacy upside in smoke range", 0 <= legacy["upside"] <= 100)
    check("legacy volAmp in smoke range", 0 <= legacy["volAmp"] <= 100)
    check("legacy has _updatedMs", bool(legacy.get("_updatedMs")))
    check("legacy flags direction as uninformative",
          legacy["upside_is_informative"] is False)
    check("legacy is JSON-serialisable", isinstance(json.dumps(legacy), str))

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

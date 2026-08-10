"""
noctua/spec.py
=====================================================================
The model's *specification* constants -- quantile grid and column layouts --
with no dependencies beyond NumPy.

This module exists to break a dependency chain, and the chain is worth naming
because it bit twice. Serving advertises NumPy + SciPy only, but:

    serve.runtime -> noctua.infer -> noctua.model -> import torch

`noctua/model.py` legitimately imports PyTorch: it defines the nn.Module
classes. `noctua/infer.py` does not need PyTorch except inside `predict()`, but
it needed LEVELS and MEDIAN_IDX, which lived in model.py -- so importing infer
dragged torch in transitively. The first layer (infer's own module-level
import) was caught by CI; the second (this one) was caught by simulating the
CI environment locally.

Putting the constants here means the numeric layout has exactly one definition,
shared by the training code and the NumPy runtime, with no framework attached.
`model.py` re-exports them so existing imports keep working.
"""
from __future__ import annotations

import numpy as np

# Quantile grid, deliberately dense in the tails: an option seller lives at
# alpha = 1-5%, so that is where resolution has to be.
LEVELS = np.array(
    [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
     0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995],
    dtype=np.float64,
)
MEDIAN_IDX = int(np.argmin(np.abs(LEVELS - 0.5)))

# Stage A's linear base: exactly the Log-HAR cascade plus the two calendar
# terms that the baseline scoreboard showed to matter.
BASE_COLS = ["har_1d", "har_5d", "har_22d", "cal_H", "cal_weekend_frac"]

# Stage B sees only SHAPE information -- not the vol level itself, except
# through the explicit log-sigma conditioning input appended at the end.
SHAPE_COLS = [
    "semi_neg_share_1d", "semi_signed_jump_1d", "semi_neg_share_5d",
    "semi_signed_jump_5d", "jump_share_1d", "jump_share_5d",
    "mom_ret_1d", "mom_ret_5d", "mom_ret_22d", "mom_dist_ma100",
    "mom_drawdown_90d", "vov_5d", "vov_22d", "reg_rv_vs_year",
    "reg_post_etf", "cal_hour_sin", "cal_hour_cos", "cal_dow_sin",
    "cal_dow_cos", "cal_H", "cal_weekend_frac",
]

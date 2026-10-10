import numpy as np
import pandas as pd

RUN = "/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10"
H = 19
TEST_START = pd.Timestamp("2024-07-01", tz="UTC")
TRAIN_END = pd.Timestamp("2024-06-30", tz="UTC")   # claim: anchors before 2024-06-30


def load_E():
    E = pd.read_parquet(f"{RUN}/evals/fresh_test_episodes.parquet").reset_index(drop=True)
    E["y"] = np.log(E.RV)
    E["lnoct"] = np.log(E.noctua_raw_med)
    dow = E.dt.dt.dayofweek
    for d in range(1, 7):
        E[f"dow{d}"] = (dow == d).astype(float)
    return E


def load_hours():
    h = pd.read_parquet(f"{RUN}/evals/hours_through_now.parquet").reset_index(drop=True)
    return h


def qlike(rv, sig):
    r = np.maximum(rv, 1e-12) ** 2 / np.maximum(sig, 1e-12) ** 2
    return r - np.log(r) - 1.0


def block_boot_mean(x, block=20, reps=4000, seed=0):
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, (reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n] % n
    m = x[idx].mean(1)
    return float(x.mean()), [float(np.quantile(m, .025)), float(np.quantile(m, .975))], float((m <= 0).mean())


def fit_eval(E, cols, fit_start, fit_end=TRAIN_END, test_start=TEST_START, seed=0):
    """OLS on log RV, QLIKE-optimal scale on the fit window, QLIKE on test window.
    Returns (qlike_test_per_day array, coef, scale)."""
    fs = pd.Timestamp(fit_start, tz="UTC")
    tr = ((E.dt >= fs) & (E.dt < fit_end)).to_numpy()
    cols_all = ["dow1", "dow2", "dow3", "dow4", "dow5", "dow6"]
    X = np.column_stack([np.ones(len(E))] + [E[c].to_numpy(float) for c in cols + cols_all])
    ok = np.isfinite(X).all(1) & np.isfinite(E.y.to_numpy()) & np.isfinite(E.RV.to_numpy())
    trm = tr & ok
    b, *_ = np.linalg.lstsq(X[trm], E.y.to_numpy()[trm], rcond=None)
    s = np.exp(X @ b)
    k = np.sqrt(np.mean(E.RV.to_numpy()[trm] ** 2 / s[trm] ** 2))
    sig = s * k
    te = ((E.dt >= test_start).to_numpy()) & ok
    q = qlike(E.RV.to_numpy()[te], sig[te])
    return q, b, k, int(trm.sum()), int(te.sum()), te

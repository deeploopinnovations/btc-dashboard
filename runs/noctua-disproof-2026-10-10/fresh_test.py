"""
runs/noctua-disproof-2026-10-10/fresh_test.py
=====================================================================
Iteration 1: an independent, fresh out-of-sample test of the SHIPPED NOCTUA.

Independent: none of eval/benchmark.py's scoring code is reused. Only the
model's own runtime (serve/runtime.py), its feature builder and the serve
adaptive correction are imported, because those ARE the product under test.

Fresh: the pinned artifact serve/noctua_v2.npz was fit before 2024-07-01, and
the price history is extended to 2026-10-10 with live 5-minute bars
(evals/hours_through_now.parquet, built by policy/paper.fetch_tail).

    python runs/noctua-disproof-2026-10-10/fresh_test.py
    python runs/noctua-disproof-2026-10-10/fresh_test.py --artifact model/serve/other.npz --tag other

Writes evals/fresh_test.json and evals/fresh_test_episodes.parquet, or with
--tag NAME evals/fresh_test_NAME.json and evals/fresh_test_episodes_NAME.parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "model"))

from noctua.features import build_features          # noqa: E402
from serve.runtime import load_model, norm_ppf        # noqa: E402
from policy import dataset as DS                      # noqa: E402
from policy import backtest as BT                     # noqa: E402
from policy.runtime import NumpyTrader                # noqa: E402
from serve.adaptive import _settled_anchors           # noqa: E402

H = 19
PROD_HOUR = 17
TEST_START = pd.Timestamp("2024-07-01", tz="UTC")
TRAIN_START = pd.Timestamp("2018-01-01", tz="UTC")
# Splice: data/assets/btc_history.parquet ends 2026-08-15 11:00 UTC,
# data/noctua_history.parquet ends 2026-10-05 17:00 UTC, and live Bitstamp/Coinbase
# bars cover the rest (evals/hours_tail_live.parquet). Label only, not a filter.
FRESH_START = pd.Timestamp("2026-08-15", tz="UTC")
ADAPT_WINDOW_D, ADAPT_STRIDE, ADAPT_MIN = 60, 6, 20
ADAPT_CLIP = (0.70, 1.40)
BLOCK, REPS = 20, 4000
ALPHAS = (0.01, 0.05, 0.10)

PERIODS = {
    "calib_2023H1_2024H1(in-sample ref)": ("2023-01-01", "2024-07-01"),
    "test_all": ("2024-07-01", "2099-01-01"),
    "2024H2": ("2024-07-01", "2025-01-01"),
    "2025H1": ("2025-01-01", "2025-07-01"),
    "2025H2": ("2025-07-01", "2026-01-01"),
    "2026H1": ("2026-01-01", "2026-07-01"),
    "2026-07-01..08-14": ("2026-07-01", "2026-08-15"),
    "fresh_2026-08-15+": ("2026-08-15", "2099-01-01"),
}


def qlike(rv: np.ndarray, sig: np.ndarray) -> np.ndarray:
    r = np.maximum(rv, 1e-12) ** 2 / np.maximum(sig, 1e-12) ** 2
    return r - np.log(r) - 1.0


def auc_mann_whitney(score: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC: P(score[up] > score[down]), ties count half (Mann-Whitney U)."""
    pos = y == 1
    n1 = int(pos.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank(method="average").to_numpy()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def block_boot_mean(x: np.ndarray, block: int = BLOCK, reps: int = REPS, seed: int = 0):
    n = len(x)
    if n < 5:
        return [float("nan"), float("nan")], float("nan")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, (reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n] % n
    m = x[idx].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))], float((m <= 0).mean())


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def trailing_sum(x: np.ndarray, k: int) -> np.ndarray:
    """out[i] = sum(x[i-k:i]), exclusive of i."""
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    out = np.full(len(x), np.nan)
    out[k:] = c[k:len(x)] - c[:len(x) - k]
    return out


def noctua_batch(model, hours: pd.DataFrame, rows: np.ndarray, keep_pred=False):
    ts = hours.hour_ts.to_numpy(np.int64)
    out_med, out_mean, preds, feat_ok = [], [], [], []
    for s in range(0, len(rows), 4096):
        r = rows[s:s + 4096]
        dt = pd.to_datetime(ts[r], unit="s", utc=True)
        ep = pd.DataFrame({"anchor_ts": ts[r], "H": H, "row": r, "dt": dt,
                           "anchor_hour": dt.hour, "dow": dt.dayofweek})
        X = build_features(hours, ep)
        feat_ok.append(np.isfinite(X.to_numpy()).all(1))    # serve/adaptive.py `ok` filter
        p = model.predict(model.prepare(X, np.full(len(r), float(H))))
        out_med.append(np.asarray(p["sigma_med"], np.float64))
        out_mean.append(np.asarray(p["sigma_mean"], np.float64))
        if keep_pred:
            preds.append(p)
    return np.concatenate(out_med), np.concatenate(out_mean), preds, np.concatenate(feat_ok)


def load_hours_through_now() -> pd.DataFrame:
    """Committed history plus the live tail fetched on 2026-10-10.

    The full frame (evals/hours_through_now.parquet, 7 MB) is git-ignored; the
    committed evals/hours_tail_live.parquet holds every bar from 2026-08-15 on,
    which is all that the committed history lacks or might later revise."""
    full = HERE / "evals" / "hours_through_now.parquet"
    if full.exists():
        return pd.read_parquet(full).reset_index(drop=True)
    tail = pd.read_parquet(HERE / "evals" / "hours_tail_live.parquet")
    a = pd.read_parquet(ROOT / "data/assets/btc_history.parquet")
    h = pd.concat([a[a.hour_ts < tail.hour_ts.min()], tail])
    h = h.sort_values("hour_ts").drop_duplicates("hour_ts").reset_index(drop=True)
    h.to_parquet(full)
    return h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fresh out-of-sample test of the NOCTUA artifact.")
    ap.add_argument("--artifact", default="model/serve/noctua_v2.npz",
                    help="NOCTUA artifact, relative to the repo root")
    ap.add_argument("--tag", default="",
                    help="write evals/fresh_test_<tag>.json instead of evals/fresh_test.json")
    ap.add_argument("--served", choices=("all_hour", "same_hour"), default="all_hour",
                    help="served-correction replica: all_hour = the 6 h stride shipped until "
                         "2026-10-10 (what VERDICT.md scored); same_hour = serve/adaptive.py "
                         "as fixed on 2026-10-10 (24 h stride stepping back from the anchor)")
    args = ap.parse_args(argv)
    art_path = ROOT / args.artifact
    art_sha12 = hashlib.sha256(art_path.read_bytes()).hexdigest()[:12]
    stem = f"fresh_test_{args.tag}" if args.tag else "fresh_test"
    ep_stem = f"fresh_test_episodes_{args.tag}" if args.tag else "fresh_test_episodes"

    t0 = time.time()
    hours = load_hours_through_now()
    ts = hours.hour_ts.to_numpy(np.int64)
    assert (np.diff(ts) == 3600).all(), "gap in hourly history"
    dt_all = pd.to_datetime(ts, unit="s", utc=True)
    rv5 = hours.rv5.to_numpy(np.float64)
    high, low, close = (hours[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    n = len(hours)

    # ---------- targets (FUTURE; used only for scoring) ------------------
    fwd = np.full(n, np.nan)
    c = np.concatenate([[0.0], np.cumsum(rv5)])
    fwd[: n - H + 1] = c[H:] - c[: n - H + 1]            # sum rv5[a .. a+H-1]
    RV = np.sqrt(fwd)

    # ---------- production anchors -----------------------------------------
    is17 = (dt_all.hour == PROD_HOUR)
    rows17 = np.flatnonzero(is17 & (dt_all >= TRAIN_START) & np.isfinite(RV))
    rows17 = rows17[rows17 > 24 * 400]

    model = load_model(art_path)
    print(f"[fresh] artifact {args.artifact} sha256[:12]={art_sha12} "
          f"meta: {model.meta.get('name', '?')} seeds={model.n_seeds}")

    # NOCTUA at every 17:00 anchor (keep full predictive objects for test rows)
    sig_med17, sig_mean17, _, _ = noctua_batch(model, hours, rows17)

    # ---------- served correction (serve/adaptive.py, replicated) ---------
    # Needs NOCTUA at every hour in the trailing 60 d of each test anchor.
    first_needed = int(np.searchsorted(ts, int((TEST_START - pd.Timedelta(days=ADAPT_WINDOW_D + 3)).timestamp())))
    rows_all = np.arange(first_needed, n)
    sm_all, _, _, ok_all = noctua_batch(model, hours, rows_all)
    sig_by_row = np.full(n, np.nan)
    sig_by_row[rows_all] = sm_all
    feat_ok_by_row = np.zeros(n, dtype=bool)               # all NOCTUA features finite at that row
    feat_ok_by_row[rows_all] = ok_all
    factor17 = np.full(len(rows17), 1.0)
    n_nonfinite_dropped = 0
    for i, a in enumerate(rows17):
        if dt_all[a] < TEST_START:
            continue
        last = a - H
        first = max(24 * 30, last - ADAPT_WINDOW_D * 24)
        if args.served == "same_hour":
            rr = _settled_anchors(hours, int(a), H, ADAPT_WINDOW_D, 24)
        else:
            rr = np.arange(first, last, ADAPT_STRIDE)
        s, rvv = sig_by_row[rr], RV[rr]
        n_nonfinite_dropped += int((~feat_ok_by_row[rr]).sum())
        g = feat_ok_by_row[rr] & np.isfinite(s) & np.isfinite(rvv) & (s > 0) & (rvv > 0)
        if g.sum() >= ADAPT_MIN:
            factor17[i] = float(np.clip(np.median(rvv[g] / s[g]), *ADAPT_CLIP))
    print(f"[fresh] NOCTUA forecasts done in {time.time() - t0:.0f}s")

    # ---------- baselines: causal inputs at anchor a use rows <= a-1 ------
    def logvol(k):                                       # per-19h scaled
        return 0.5 * np.log(np.maximum(trailing_sum(rv5, k) * H / k, 1e-14))
    Xh = {k: logvol(k) for k in (1, 6, 24, 120, 528)}
    dvol = DS._hourly_series(ROOT / "data/newdata/dvol_btc.parquet", "volatility", "ts", ts)
    dvol_prev = np.concatenate([[np.nan], dvol[:-1]])   # value stamped a-1
    dvol19 = dvol_prev / 100.0 * np.sqrt(H / (24 * 365))
    lam = 0.94 ** (1 / 24)                               # RiskMetrics daily lambda, hourly
    ew = np.full(n, np.nan)
    acc = rv5[:24].mean()
    for i in range(24, n):
        ew[i] = acc                                      # uses rv5 up to i-1
        acc = lam * acc + (1 - lam) * rv5[i]
    ewma19 = np.sqrt(ew * H)
    persist = np.exp(Xh[24])

    E = pd.DataFrame({"row": rows17, "dt": dt_all[rows17], "RV": RV[rows17],
                      "noctua_raw_med": sig_med17, "noctua_raw_mean": sig_mean17,
                      "factor": factor17})
    E["noctua_served_mean"] = E.noctua_raw_mean * E.factor
    E["noctua_served_med"] = E.noctua_raw_med * E.factor
    for k, v in Xh.items():
        E[f"lv{k}"] = v[rows17]
    E["ldvol"] = np.log(dvol19[rows17])
    E["ewma"] = ewma19[rows17]
    E["persistence"] = persist[rows17]
    E["y"] = np.log(E.RV)

    train = (E.dt < TEST_START - pd.Timedelta(hours=H)).to_numpy()   # embargoed
    test = (E.dt >= TEST_START).to_numpy()

    def ols(cols, mask):
        A = np.column_stack([np.ones(mask.sum())] + [E.loc[mask, c].to_numpy() for c in cols])
        b, *_ = np.linalg.lstsq(A, E.loc[mask, "y"].to_numpy(), rcond=None)
        return b

    def pred_ols(cols, b, idx=None):
        Z = E if idx is None else E.iloc[idx]
        A = np.column_stack([np.ones(len(Z))] + [Z[c].to_numpy() for c in cols])
        return np.exp(A @ b)

    def qlike_scale(sig, mask):
        """QLIKE-optimal multiplier fitted on `mask` only: sqrt(mean(RV^2/sig^2))."""
        return float(np.sqrt(np.mean(E.RV.to_numpy()[mask] ** 2 / sig[mask] ** 2)))

    specs = {"log_har3": ["lv24", "lv120", "lv528"],
             "har5": ["lv1", "lv6", "lv24", "lv120", "lv528"]}
    fitted = {}
    for name, cols in specs.items():
        b = ols(cols, train)
        s = pred_ols(cols, b)
        k = qlike_scale(s, train)
        E[f"{name}_frozen"] = s * k
        fitted[name] = {"coef": b.tolist(), "scale": k}

    # expanding refit HAR5, monthly, only on episodes settled before the month
    E["har5_refit"] = np.nan
    months = pd.date_range(TEST_START, E.dt.max() + pd.offsets.MonthBegin(1), freq="MS")
    for m0, m1 in zip(months[:-1], months[1:]):
        fit_mask = ((E.dt + pd.Timedelta(hours=H)) <= m0).to_numpy() & np.isfinite(E[specs["har5"]].to_numpy()).all(1)
        idx = np.flatnonzero(((E.dt >= m0) & (E.dt < m1)).to_numpy())
        if len(idx) == 0:
            continue
        b = ols(specs["har5"], fit_mask)
        s_fit = pred_ols(specs["har5"], b)
        k = qlike_scale(s_fit, fit_mask)
        E.loc[E.index[idx], "har5_refit"] = pred_ols(specs["har5"], b, idx) * k

    dmask = train & np.isfinite(E.ldvol.to_numpy())
    k = qlike_scale(np.exp(E.ldvol.to_numpy()), dmask)
    E["dvol_market"] = np.exp(E.ldvol) * k
    b = ols(specs["har5"] + ["ldvol"], dmask)
    s = pred_ols(specs["har5"] + ["ldvol"], b)
    E["har5_plus_dvol"] = s * qlike_scale(s, dmask)
    E["ewma"] = E.ewma * qlike_scale(E.ewma.to_numpy(), train)
    E["persistence"] = E.persistence * qlike_scale(E.persistence.to_numpy(), train)
    fitted["dvol_scale"] = k
    fitted["har5_plus_dvol_coef"] = b.tolist()

    # weekday-aware HAR (A1): 6 dummies for the 17:00 anchor's day of week, Monday as base.
    # Same embargoed train mask as above; the recent arm starts its fit window in 2023.
    wd_cols = [f"wd{d}" for d in range(1, 7)]
    for d, col in enumerate(wd_cols, start=1):
        E[col] = (E.dt.dt.dayofweek == d).astype(float)
    wk_cols = specs["har5"] + wd_cols
    wk_fits = {"har5_weekday_frozen": train,
               "har5_weekday_recent": train & (E.dt >= pd.Timestamp("2023-01-01", tz="UTC")).to_numpy()}
    for name, fit_mask in wk_fits.items():
        b = ols(wk_cols, fit_mask)
        s = pred_ols(wk_cols, b)
        k = qlike_scale(s, fit_mask)
        E[name] = s * k
        fitted[name] = {"coef": b.tolist(), "scale": k, "cols": wk_cols,
                        "fit_start": str(E.dt[fit_mask].min()), "n_fit": int(fit_mask.sum())}

    noctua_arms = ["noctua_raw_med", "noctua_raw_mean", "noctua_served_mean", "noctua_served_med"]
    base_arms = ["log_har3_frozen", "har5_frozen", "har5_refit", "ewma", "dvol_market",
                 "har5_plus_dvol", "persistence", "har5_weekday_frozen", "har5_weekday_recent"]

    results = {"vol": {}, "meta": {
        "artifact": args.artifact, "artifact_sha256_12": art_sha12,
        "served_replica": args.served,
        "H": H, "anchor_hour_utc": PROD_HOUR,
        "history_end_utc": str(dt_all[-1]), "n_train_episodes": int(train.sum()),
        "n_test_episodes": int(test.sum()), "baseline_fits": fitted,
        "served_replica_nonfinite_dropped": n_nonfinite_dropped,
        "block_days": BLOCK, "reps": REPS,
        "notes": ("Baseline arms (HAR, refit, EWMA, persistence, DVOL, weekday HAR) get an in-sample "
                  "QLIKE rescale fitted on the training window; NOCTUA arms do not. The served arms "
                  "multiply by a causal trailing median factor. The Gaussian barrier arms use the "
                  "mean-type scaled har5_frozen and har5_weekday_frozen; NOCTUA's median-based arms are unscaled.")}}

    for pname, (p0, p1) in PERIODS.items():
        pm = ((E.dt >= pd.Timestamp(p0, tz="UTC")) & (E.dt < pd.Timestamp(p1, tz="UTC"))).to_numpy()
        rv = E.RV.to_numpy()
        res = {"n": int(pm.sum()), "qlike": {}, "paired_vs": {}}
        for a in noctua_arms + base_arms:
            s = E[a].to_numpy()
            ok = pm & np.isfinite(s)
            res["qlike"][a] = float(qlike(rv[ok], s[ok]).mean()) if ok.any() else None
        for na in ("noctua_raw_med", "noctua_served_mean"):
            res["paired_vs"][na] = {}
            for ba in base_arms:
                sb, sn = E[ba].to_numpy(), E[na].to_numpy()
                ok = pm & np.isfinite(sb) & np.isfinite(sn)
                if ok.sum() < 5:
                    continue
                d = qlike(rv[ok], sb[ok]) - qlike(rv[ok], sn[ok])   # >0: NOCTUA better
                ci, p_le0 = block_boot_mean(d)
                res["paired_vs"][na][ba] = {
                    "n": int(ok.sum()), "mean_gain": float(d.mean()),
                    "rel_pct": float(100 * d.mean() / qlike(rv[ok], sb[ok]).mean()),
                    "ci95": ci, "p_gain_le_0": p_le0}
        res["median_RV_over_sigma"] = {a: float(np.nanmedian(rv[pm] / E[a].to_numpy()[pm]))
                                       for a in noctua_arms + ["har5_frozen"]}
        results["vol"][pname] = res

    # rolling 180-day QLIKE gain vs har5_frozen, served and raw (decay check)
    roll = {}
    Et = E[test].reset_index(drop=True)
    for na in ("noctua_raw_med", "noctua_served_mean"):
        d = qlike(Et.RV.to_numpy(), Et.har5_frozen.to_numpy()) - qlike(Et.RV.to_numpy(), Et[na].to_numpy())
        r = pd.Series(d).rolling(180).mean()
        roll[na] = {"dates": Et.dt.dt.strftime("%Y-%m-%d").iloc[179::30].tolist(),
                    "gain": [float(x) for x in r.iloc[179::30]],
                    "frac_windows_positive": float((r.dropna() > 0).mean())}
    results["rolling_180d_gain_vs_har5_frozen"] = roll

    # ---------- deep-tail barriers -----------------------------------------
    te_rows = E.row.to_numpy()[test]
    preds = noctua_batch(model, hours, te_rows, keep_pred=True)[2]
    fac = E.factor.to_numpy()[test]
    s_tau = close[te_rows - 1]
    mup = np.array([max(0.0, np.log(high[a:a + H].max() / s)) for a, s in zip(te_rows, s_tau)])
    mdn = np.array([min(0.0, np.log(low[a:a + H].min() / s)) for a, s in zip(te_rows, s_tau)])
    from serve.adaptive import apply_correction
    tail = {}
    off = 0
    lv_raw = {(a, up): [] for a in ALPHAS for up in (True, False)}
    lv_srv = {(a, up): [] for a in ALPHAS for up in (True, False)}
    for p in preds:
        m = len(p["sigma_med"])
        for a in ALPHAS:
            for up in (True, False):
                lv_raw[(a, up)].append(model.safe_level(p, a, up))
        for j in range(m):
            pj = {k: (v[j:j + 1] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == m else v)
                  for k, v in p.items() if not k.startswith("_pooled_")}
            pj = apply_correction(pj, float(fac[off + j]))
            for a in ALPHAS:
                for up in (True, False):
                    lv_srv[(a, up)].append(model.safe_level(pj, a, up))
        off += m
    gauss_sig = E.har5_frozen.to_numpy()[test]
    gauss_wk_sig = E.har5_weekday_frozen.to_numpy()[test]
    dts = E.dt[test].reset_index(drop=True)
    for a in ALPHAS:
        for up in (True, False):
            side = "up" if up else "dn"
            u_raw = np.concatenate(lv_raw[(a, up)])
            u_srv = np.concatenate(lv_srv[(a, up)])
            u_gau = -gauss_sig * norm_ppf(a / 2.0)       # reflection principle, P(touch)=a
            u_gau_wk = -gauss_wk_sig * norm_ppf(a / 2.0)
            M = mup if up else -mdn
            arms = (("noctua_raw", u_raw), ("noctua_served", u_srv), ("gauss_har5", u_gau),
                    ("gauss_har5wk", u_gau_wk))
            for arm, u in arms:
                hit = M >= np.abs(u)
                fr = (dts >= FRESH_START).to_numpy()
                tail[f"{side}_a{a:.2f}_{arm}"] = {
                    "nominal": a, "n": int(len(hit)), "breaches": int(hit.sum()),
                    "rate": float(hit.mean()), "wilson95": wilson(int(hit.sum()), len(hit)),
                    "fresh_n": int(fr.sum()), "fresh_breaches": int(hit[fr].sum()),
                    "median_level_pct": float(100 * np.median(np.expm1(np.abs(u))))}
    results["tails"] = tail

    # ---------- direction (expected negative) ------------------------------
    pu = np.concatenate([model.prob_up(p) for p in preds])
    R = np.log(close[te_rows + H - 1] / s_tau)
    yup = (R > 0).astype(float)
    # pre-test base rate: 17:00 anchors whose 19h outcome closes before TEST_START (train mask, embargoed)
    tr_row = E.row.to_numpy()[train]
    base_pre = float((np.log(close[tr_row + H - 1] / close[tr_row - 1]) > 0).mean())
    results["direction"] = {
        "n": int(len(yup)), "frac_up": float(yup.mean()),
        "base_rate_pre_test": base_pre, "n_pre_test": int(len(tr_row)),
        "brier_model": float(np.mean((pu - yup) ** 2)),
        "brier_base_rate": float(np.mean((base_pre - yup) ** 2)),
        "brier_half": float(np.mean((0.5 - yup) ** 2)),       # constant 0.5 forecast, not a base rate
        "pearson_r": float(np.corrcoef(pu, yup)[0, 1]),
        "auc": auc_mann_whitney(pu, yup),
        "note": "served upside is pinned to 50.0; this scores the raw prob_up anyway. "
                "brier_half is the constant-0.5 forecast; brier_base_rate uses the pre-test base rate."}

    # ---------- trader v1 --------------------------------------------------
    rows_tr = np.flatnonzero((dt_all.hour == PROD_HOUR) & (dt_all >= TEST_START))
    rows_tr = rows_tr[rows_tr + DS.HOLD_H <= n]
    F = DS.features_at(hours, rows_tr)
    T = DS.targets(hours, rows_tr)
    trader = NumpyTrader()
    pos = trader.position(F)
    ret, fund = T.tgt_ret.to_numpy(), T.tgt_fund.to_numpy()
    sig_ann_trail = np.sqrt(trailing_sum(rv5, 24 * 30)[rows_tr] / (24 * 30) * 24 * 365)
    sig_ann_nx = np.exp(F.nx_logsig_med.to_numpy()) * np.sqrt(24 * 365 / 19.0)
    # matched-exposure controls (M2)
    idx_tr = np.arange(len(pos))
    lo_tr = np.maximum(0, idx_tr - 60)
    cpos = np.concatenate([[0.0], np.cumsum(pos)])
    cnt_tr = idx_tr - lo_tr                                 # days in the prior 60-day window
    trailing60 = np.where(cnt_tr > 0, (cpos[idx_tr] - cpos[lo_tr]) / np.maximum(cnt_tr, 1), 0.0)
    strategies = {
        "trader_v1": pos,
        "buy_and_hold": np.ones(len(pos)),
        "half_long": np.full(len(pos), 0.5),
        "voltarget_trailing30d_cap0.5": np.clip(0.25 / sig_ann_trail, 0, 0.5),
        "voltarget_noctua_cap0.5": np.clip(0.25 / sig_ann_nx, 0, 0.5),
        "constant_matched_exposure": np.full(len(pos), float(pos.mean())),
        "trailing60_mean_position": trailing60,
    }
    strat_notes = {
        "constant_matched_exposure": "diagnostic only: constant position equal to the trader's mean "
                                     "position over the scored days; uses the sample mean, not tradable",
        "trailing60_mean_position": "causal: day i holds the mean of the trader's positions on days "
                                    "i-60..i-1 (excludes day i); expanding mean before 60 days; 0 on day 0",
    }
    dtr = pd.to_datetime(F.anchor_ts.to_numpy(), unit="s", utc=True)
    trd = {"n_days": int(len(pos)), "first": str(dtr[0]), "last": str(dtr[-1]),
           "avg_position_trader": float(pos.mean()), "strategies": {}, "vs_trader": {}}
    pl = {}
    for k, p in strategies.items():
        pl[k] = BT.pnl(p, ret, fund)
        m = BT.metrics(pl[k], p)
        yr = {}
        for y in sorted(set(dtr.year)):
            msk = (dtr.year == y)
            yr[str(y)] = float(np.prod(1 + pl[k][msk]) - 1)
        fr = (dtr >= FRESH_START)
        m["compounded_by_year"] = yr
        m["fresh_total_return"] = float(np.prod(1 + pl[k][fr]) - 1)
        if k in strat_notes:
            m["notes"] = strat_notes[k]
        trd["strategies"][k] = m
    for k in strategies:
        if k != "trader_v1":
            trd["vs_trader"][k] = BT.block_bootstrap_sharpe_diff(pl["trader_v1"], pl[k], reps=4000)
    results["trader"] = trd

    out = HERE / "evals" / f"{stem}.json"
    out.write_text(json.dumps(results, indent=2, default=float) + "\n")
    E.to_parquet(HERE / "evals" / f"{ep_stem}.parquet")
    print(f"[fresh] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

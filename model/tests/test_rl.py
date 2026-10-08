"""
tests/test_rl.py
=====================================================================
The gate for the paper-trading agent (P5-rl-paper; model/rl/). Offline, no
network: synthetic streams for the learners, the committed history bundle for
one real NOCTUA step.

THE CONTRACT
  UTILITY is u = w R - c|w - w_prev| - (gamma/2)(w R)^2, exactly.
  BOTH LEARNERS FIND A PLANTED EDGE and beat holding; NEITHER INVENTS ONE in
  pure noise (they stay near flat, as a learner with nothing to learn must).
  STATE ROUND-TRIPS: an agent restored from its JSON makes the same decisions.
  THE LOOP IS CAUSAL: it decides at slot E only when the hour before E is
  complete, never reads a bar at or after E, and settles only once the hour
  starting E+5h is complete, with R = close(E+5h) / close(E-1h) - 1.
  THE LOOP IS IDEMPOTENT: a restart never decides a slot twice or learns from a
  window twice. A feed outage HOLDS the position (its P&L still counts) and is
  logged as missed. Every settled step logs what FLAT/HALF/HOLD/VT would have
  earned on the same R. AND THE GATE CAN FAIL (R2).

    python model/tests/test_rl.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl import env as E                                            # noqa: E402
from rl.agent import make_agent                   # noqa: E402
from rl.loop import PaperTrader                                     # noqa: E402
from rl.store import HFStore, LocalStore                            # noqa: E402

HOUR = 3600


def synthetic(n, edge, seed, drift=0.0):
    """Steps with s ~ 1.5%, x4 carrying `edge` of the sign of R if edge > 0."""
    rng = np.random.default_rng(seed)
    s = np.exp(rng.normal(np.log(0.015), 0.3, n))
    x = rng.normal(0, 1, (n, 5))
    sign = np.sign(x[:, 3])
    R = drift + edge * sign + s * rng.normal(0, 1, n)
    x[:, 0] = (np.log(s) - np.log(0.015)) / 0.5
    return x, s, R


def run(agent, x, s, R, seed=0):
    rng = np.random.default_rng(seed)
    w, us, ws = 0.0, [], []
    for i in range(len(R)):
        a = agent.act(x[i], s[i], w, rng)
        us.append(E.utility(a, R[i], w))
        agent.update(x[i], s[i], w, a, R[i])
        ws.append(a)
        w = a
    return np.array(us), np.array(ws)


def hold_u(R):
    w = np.ones(len(R)); prev = np.r_[0.0, w[:-1]]
    return np.array([E.utility(1.0, r, p) for r, p in zip(R, prev)])


def main() -> int:
    ok = []

    def check(name, good, detail=""):
        ok.append((name, bool(good), detail))

    # ---- utility --------------------------------------------------------
    u = E.utility(1.0, 0.01, 0.0, cost=0.001, gamma=2.0)
    check("utility-is-exact", abs(u - (0.01 - 0.001 - 1e-4)) < 1e-15, f"{u!r}")
    check("utility-charges-turnover-both-ways",
          abs(E.utility(-1.0, 0.0, 1.0, cost=0.001) + 0.002) < 1e-15)

    # ---- learners: a planted edge, and pure noise -----------------------
    x, s, R = synthetic(3000, edge=0.006, seed=1)
    for kind in ("MV", "TS"):
        us, ws = run(make_agent(kind, seed=3), x, s, R)
        late = slice(1500, None)
        check(f"{kind}-learns-a-planted-edge",
              us[late].mean() > hold_u(R)[late].mean() + 1e-4 and us[late].mean() > 1e-4,
              f"late mean u {us[late].mean():+.5f} vs hold {hold_u(R)[late].mean():+.5f}")
        check(f"{kind}-positions-follow-the-planted-signal",
              np.corrcoef(ws[late], np.sign(x[late, 3]))[0, 1] > 0.5,
              f"corr {np.corrcoef(ws[late], np.sign(x[late, 3]))[0, 1]:+.2f}")
    x0, s0, R0 = synthetic(3000, edge=0.0, seed=2)
    for kind in ("MV", "TS"):
        us, ws = run(make_agent(kind, seed=5), x0, s0, R0)
        late = slice(1500, None)
        flat_share = float(np.mean(ws[late] == 0.0))
        passes = flat_share > (0.80 if kind == "MV" else 0.30) and us[late].mean() > -2e-4
        if kind == "MV":
            check("MV-invents-no-edge-in-noise", passes,
                  f"flat {flat_share:.0%} of late steps, mean u {us[late].mean():+.6f}")
        else:
            # KNOWN, asserted: classic Thompson-sampling RL with forgetting never
            # stops exploring, and in pure noise that exploration is pure cost.
            # It stays in the replay as registered and is barred from the live
            # Space (space/run.py). If this ever passes, TS changed: re-decide.
            check("TS-trades-noise-so-it-is-not-deployed", not passes,
                  f"flat {flat_share:.0%} of late steps, mean u {us[late].mean():+.6f}")

    # ---- state round trip ------------------------------------------------
    for kind in ("MV", "TS"):
        a1 = make_agent(kind, seed=7)
        run(a1, x[:400], s[:400], R[:400])
        a2 = make_agent(kind, seed=7)
        a2.load_state(json.loads(json.dumps(a1.state())))
        r1, r2 = np.random.default_rng(11), np.random.default_rng(11)
        d1 = [a1.act(x[i], s[i], 0.0, r1) for i in range(400, 450)]
        d2 = [a2.act(x[i], s[i], 0.0, r2) for i in range(400, 450)]
        check(f"{kind}-state-round-trips", d1 == d2, f"{sum(p == q for p, q in zip(d1, d2))}/50 identical")

    # ---- the loop, with a fake forecaster and a slice of real history ----
    from serve.history import load_bundle
    hours = load_bundle()
    hts = hours.hour_ts.to_numpy(np.int64)
    slots = [int(t) for t in hts if pd.Timestamp(int(t), unit="s", tz="UTC").hour in E.SLOT_HOURS]
    E0 = slots[-12]                                          # leaves room for two settled windows
    seen_rows = []

    def fake_forecaster(h_upto, Eslot):
        seen_rows.append(int(h_upto.hour_ts.max()))
        return {"sigma": 0.015, "trailing_rv": 0.012, "p_up": 0.5, "p_vol_amplify": 0.5}

    def upto(t):                                             # the feed as it stands at time t
        return hours[hours.hour_ts + HOUR <= t].reset_index(drop=True)

    with tempfile.TemporaryDirectory() as td:
        st = LocalStore(Path(td))
        pt = PaperTrader(st, agent_kind="MV", forecaster=fake_forecaster)
        # 1. too early: the hour before E0 is not complete yet
        ev = pt.tick(E0 - 60, upto(E0 - 60))
        check("no-decision-before-the-slot", not [e for e in ev if e["type"] == "decision"])
        # 2. at the slot: decides, and saw no bar at or after E0
        ev = pt.tick(E0 + 180, upto(E0 + 180))
        dec = [e for e in ev if e["type"] == "decision"]
        check("decides-at-the-slot", len(dec) == 1 and dec[0]["E"] == E0, f"{dec}")
        check("never-reads-a-bar-at-or-after-E", seen_rows and max(seen_rows) < E0,
              f"last bar seen {pd.Timestamp(max(seen_rows), unit='s', tz='UTC')} for E {pd.Timestamp(E0, unit='s', tz='UTC')}")
        # 3. a restart in the same slot: no second decision
        pt2 = PaperTrader(LocalStore(Path(td)), agent_kind="MV", forecaster=fake_forecaster)
        ev = pt2.tick(E0 + 600, upto(E0 + 600))
        check("restart-does-not-decide-twice", not [e for e in ev if e["type"] == "decision"])
        # 4. not settled before E0 + 6h
        ev = pt2.tick(E0 + 6 * HOUR - 60, upto(E0 + 6 * HOUR - 60))
        check("no-settlement-before-the-window-closes", not [e for e in ev if e["type"] == "settled"])
        # 5. settled after, with the right R and the baselines logged
        E1 = E0 + 6 * HOUR
        ev = pt2.tick(E1 + 180, upto(E1 + 180))
        stl = [e for e in ev if e["type"] == "settled"]
        r0 = int(np.searchsorted(hts, E0))
        R_true = hours.close[r0 + 5] / hours.close[r0 - 1] - 1
        check("settles-with-the-right-return",
              len(stl) == 1 and abs(stl[0]["R"] - R_true) < 1e-15, f"{stl[0]['R'] if stl else None} vs {R_true}")
        check("baselines-logged-on-the-same-return",
              stl and set(stl[0]["u_baseline"]) == {"FLAT", "HALF", "HOLD", "VT"}
              and abs(stl[0]["u_baseline"]["HOLD"] - E.utility(1.0, R_true, 0.0)) < 1e-15)
        check("regret-against-the-best-action-logged",
              stl and stl[0]["regret"] >= 0 and len(stl[0]["u_all"]) == len(E.ACTIONS))
        n_upd = pt2.agent.n_updates
        # 6. restart after settlement: no second update
        pt3 = PaperTrader(LocalStore(Path(td)), agent_kind="MV", forecaster=fake_forecaster)
        pt3.tick(E1 + 900, upto(E1 + 900))
        check("restart-does-not-learn-twice", pt3.agent.n_updates == n_upd,
              f"{pt3.agent.n_updates} vs {n_upd}")
        # 7. a feed outage past the grace window: position held, logged missed
        E2 = E1 + 6 * HOUR
        w_before = pt3.w
        stale = upto(E2 - 2 * HOUR)                          # the feed stopped two hours early
        ev = pt3.tick(E2 + 2 * HOUR, stale)
        miss = [e for e in ev if e["type"] == "missed"]
        check("outage-holds-the-position-and-is-logged",
              len(miss) == 1 and pt3.w == w_before and pt3.pending and pt3.pending[-1]["missed"])
        log = st.read_log()
        check("log-is-append-only-jsonl", len(log) >= 1 and all("E" in r for r in log), f"{len(log)} records")

    # ---- robustness (audit G, 2026-10-08): each was a real failure -------
    from huggingface_hub import errors as hf_errors

    class FakeApi:
        token = "x"

        def __init__(self):
            self.uploads = []

        def create_repo(self, *a, **k):
            pass

        def upload_file(self, path_or_fileobj, path_in_repo, **k):
            self.uploads.append(path_in_repo)

    def raiser(exc):
        def f(*a, **k):
            raise exc
        return f

    # 1. a transient download error must ABORT, not look like a first run
    #    (it used to upload a fresh agent / a one-line log over the real ones)
    with tempfile.TemporaryDirectory() as td:
        hs = HFStore(Path(td), "u/d", api=FakeApi())
        hs._download = raiser(ConnectionError("503 from the hub"))
        aborted = False
        try:
            hs.restore()
        except Exception:
            aborted = True
        check("restore-error-aborts-instead-of-starting-fresh", aborted)
        resp = type("R", (), {"status_code": 404, "headers": {}, "request": None})()
        hs._download = raiser(hf_errors.RemoteEntryNotFoundError("missing", response=resp))
        ok_absent = True
        try:
            hs.restore()
        except Exception as e:
            ok_absent = f"raised {type(e).__name__}"
        check("restore-missing-file-is-a-first-run", ok_absent is True, str(ok_absent))

    # a three-slot stretch of real bars, fake forecaster
    E3 = slots[-20]

    def run_ticks(st, plan, forecaster=fake_forecaster):
        out = []
        for t, h in plan:
            out += PaperTrader(st, agent_kind="MV", forecaster=forecaster).tick(t, h)
        return out

    # 2. a crash after a settlement (here: in the next decision) must not
    #    duplicate the log record or the learning on restart
    with tempfile.TemporaryDirectory() as td:
        st = LocalStore(Path(td))
        run_ticks(st, [(E3 + 180, upto(E3 + 180))])
        boom = {"n": 0}

        def flaky(h_upto, Eslot):
            boom["n"] += 1
            raise RuntimeError("planted crash while deciding")
        try:
            PaperTrader(st, agent_kind="MV", forecaster=flaky).tick(E3 + 6 * HOUR + 180, upto(E3 + 6 * HOUR + 180))
        except RuntimeError:
            pass
        pt = PaperTrader(st, agent_kind="MV", forecaster=fake_forecaster)
        pt.tick(E3 + 6 * HOUR + 240, upto(E3 + 6 * HOUR + 240))
        Es = [r["E"] for r in st.read_log()]
        check("crash-after-settlement-no-duplicate-record", Es == [E3], f"log E {Es}")
        check("crash-after-settlement-learns-once", pt.agent.n_updates == 1, f"n_updates {pt.agent.n_updates}")

    # 3. a window whose bars arrive AFTER a later window settled still settles,
    #    and every baseline's cost uses its own previous position. Reproduces
    #    audit G's a3d3: the bar before slot Ea is late, so the window before
    #    it cannot close and Ea becomes a forced hold; a later window Eb then
    #    settles first. A settled-up-to watermark left Ez and Ea pending forever.
    with tempfile.TemporaryDirectory() as td:
        st = LocalStore(Path(td))
        Ez, Ea = E3, E3 + 6 * HOUR
        Eb = Ea + 6 * HOUR
        gap = Ea - HOUR
        holey = lambda t: upto(t)[lambda d: d.hour_ts != gap].reset_index(drop=True)   # noqa: E731
        run_ticks(st, [(Ez + 180, upto(Ez + 180)),                 # decide Ez
                       (Ea + 3100, holey(Ea + 3100)),              # Ea past grace, bar missing: hold
                       (Eb + 180, holey(Eb + 180)),                # decide Eb
                       (Eb + 6 * HOUR + 180, holey(Eb + 6 * HOUR + 180)),   # Eb settles first
                       (Eb + 6 * HOUR + 400, upto(Eb + 6 * HOUR + 400))])   # the bar arrives
        log = st.read_log()
        Es = sorted(r["E"] for r in log if r["type"] == "settled")
        check("late-bar-window-still-settles", {Ez, Ea, Eb} <= set(Es) and len(Es) == len(set(Es)),
              f"settled {[(e - Ez) // HOUR for e in Es]} h after Ez")
        rb = {r["E"]: r for r in log}
        base_ok = Eb in rb and abs(rb[Eb]["u_baseline"]["HOLD"]
                                   - E.utility(1.0, rb[Eb]["R"], 1.0)) < 1e-15
        check("baseline-cost-uses-its-own-previous-position-out-of-order", base_ok,
              f"HOLD u {rb[Eb]['u_baseline']['HOLD'] if Eb in rb else None}")

    # 4. a window whose bars never arrive is logged VOID after a week, not pending forever
    with tempfile.TemporaryDirectory() as td:
        st = LocalStore(Path(td))
        Ea = slots[-60]                                   # 8+ days of real bars after it
        never = lambda t: upto(t)[lambda d: d.hour_ts != Ea + 5 * HOUR].reset_index(drop=True)   # noqa: E731
        run_ticks(st, [(Ea + 180, upto(Ea + 180))])
        t_late = Ea + 6 * HOUR + 8 * 86400
        pt = PaperTrader(st, agent_kind="MV", forecaster=fake_forecaster)
        pt.tick(t_late, never(t_late))
        voids = [r for r in st.read_log() if r.get("type") == "void" and r["E"] == Ea]
        check("never-settling-window-is-logged-void", len(voids) == 1 and all(d["E"] != Ea for d in pt.pending),
              f"voids {len(voids)}, still pending {[d['E'] for d in pt.pending][:3]}")

    # 5. a clock far ahead of the feed must not mint a year of fake holds
    with tempfile.TemporaryDirectory() as td:
        st = LocalStore(Path(td))
        run_ticks(st, [(E3 + 180, upto(E3 + 180))])
        before = st.state_path.read_text()
        pt = PaperTrader(st, agent_kind="MV", forecaster=fake_forecaster)
        refused = False
        try:
            pt.tick(E3 + 365 * 86400, upto(E3 + 180))
        except E.ClockAheadOfFeed:
            refused = True
        check("clock-far-ahead-of-feed-is-refused-state-untouched",
              refused and st.state_path.read_text() == before)

    # ---- one real NOCTUA step through the serving code -----------------
    real = E.inputs_at(None, hours[hours.hour_ts < E0].reset_index(drop=True), E0)
    check("real-noctua-inputs", np.isfinite(real["x"]).all() and 0.001 < real["sigma"] < 0.2,
          f"sigma {real['sigma']:.4f} p_up {real['p_up']} x {np.round(real['x'], 3).tolist()}")

    # ---- R2: the planted-edge check can fail ----------------------------
    us_bad, _ = run(make_agent("MV", seed=3), x, s, np.random.default_rng(9).permutation(R))
    check("gate-can-fail", not (us_bad[1500:].mean() > hold_u(R)[1500:].mean() + 1e-4
                                and us_bad[1500:].mean() > 1e-4),
          f"with R shuffled the edge check reads {us_bad[1500:].mean():+.5f}")

    print("rl paper-trading gate")
    for n, good, m in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {n}: {m}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok) - len(bad)}/{len(ok)} checks passed")
    if not bad:
        print("all checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
